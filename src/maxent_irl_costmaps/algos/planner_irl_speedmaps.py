import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import scipy.spatial
import scipy.interpolate

from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.costmappers.linear_costmapper import LinearCostMapper
from maxent_irl_costmaps.utils import get_state_visitations, get_speedmap
from maxent_irl_costmaps.geometry_utils import apply_footprint

from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *

class PlannerIRLSpeedmaps:
    """
    This is the same as MPPI IRL, but in addition to the IRL, also learn a speed map via MLE to expert speed
    Speedmap Learning:
        1. Run the network to get the per-cell speed distribution
        2. Create a speed label for each cell that the expert visited (have to register speeds/traj onto the map)
        3. Compute a masked MLE for the cells that were visited
        4. backward pass w/ the IRL grad
        5. win

    MPPI IRL:
        Costmap learner that uses expert data + MPPI optimization to learn costmaps.
        The algorithm is as follows:
        1. Get empirical feature counts across the entire dataset for expert
        2. Iteratively
            a. Sample a batch of data from the expert's dataset
            b. Compute a set of costmaps from the current weight vector.
            c. Use MPPI to optimize a trajectory on the costmap
            d. Get empirical feature counts from the MPPI solver (maybe try the weighted trick)
        e. Match feature expectations
    """
    def __init__(self, network, opt, expert_dataset, planner, footprint, batch_size=64, speed_coeff=1.0, reg_coeff=1e-2, grad_clip=1., categorical_speedmaps=False, device='cpu'):
        """
        Args:
            network: the network to use for predicting costmaps
            opt: the optimizer for the network
            expert_dataset: The dataset containing expert demonstrations to imitate
            footprint: "smear" state visitations with this
            mppi: The MPPI object to optimize with
        """
        self.expert_dataset = expert_dataset
        self.footprint = footprint
        self.planner = planner
        self.length_weight = 0.05
        self.categorical_speedmaps = categorical_speedmaps

        self.network = network

        print(self.network)
        print('({} params)'.format(sum([x.numel() for x in self.network.parameters()])))
        print(expert_dataset.feature_keys)
        self.network_opt = opt

        self.batch_size = batch_size
        self.reg_coeff = reg_coeff
        self.speed_coeff = speed_coeff
        self.grad_clip = grad_clip

        self.itr = 0
        self.device = device

    def update(self, n=-1):
        """
        High-level method that runs training for one epoch.
        """
        self.itr += 1
        dl = DataLoader(self.expert_dataset, batch_size=self.batch_size, shuffle=True)
        for i, batch in enumerate(dl):
            if n > -1 and i >= n:
                break

            #skip the last batch in the dataset as MPPI batching forces a fixed size
            if batch['traj'].shape[0] < self.batch_size:
                break

            print('{}/{}'.format(i+1, int(len(self.expert_dataset)/self.batch_size)), end='\r')
            self.gradient_step(batch)

        print('_____ITR {}_____'.format(self.itr))

    def gradient_step(self, batch):
        """
        Apply the MaxEnt update to the network given a batch
        """
        assert batch['metadata']['resolution'].std() < 1e-4, "got mutliple resolutions in a batch, which we currently don't support"

        grads = []
        speed_loss = []

        efc = []
        lfc = []
        rfc = []
        costmap_cache = []

        #first generate all the costmaps
        res = self.network.forward(batch['map_features'])
        costmaps = res['costmap'][:, 0]
        speedmaps = res['speedmap']

        #initialize metadata for cost function
        map_params = batch['metadata']

        #initialize goals for cost function
        expert_traj = batch['traj']
        expert_kbm_traj = torch.stack([
            expert_traj[:, :, 0],
            expert_traj[:, :, 1],
            self.quat_to_yaw(expert_traj[:, :, 3:7]) % (2*np.pi)
        ], dim=-1)

        #have the goal be the last traj point still in map bounds
        initial_pos = expert_kbm_traj[:, 0]

        goal_pos = []
        for bi in range(initial_pos.shape[0]):
            map_params_b = {
                'resolution': batch['metadata']['resolution'].mean().item(),
                'height': batch['metadata']['height'].mean().item(),
                'width': batch['metadata']['width'].mean().item(),
                'origin': batch['metadata']['origin'][bi]
            }
            etraj = expert_kbm_traj[bi]
            goal_pos.append(self.clip_to_map_bounds(etraj, map_params_b))

        goal_pos = torch.stack(goal_pos, dim=0)

        # setup start state
        angles = torch.linspace(0., 2*np.pi, self.planner.primitives['angnum']+1, device=self.planner.device).view(1, -1)

        sgx = ((initial_pos[:, 0] - map_params['origin'][:, 0]) / self.planner.primitives['lindisc']).round().clip(0, self.planner.states.shape[0]-1)
        sgy = ((initial_pos[:, 1] - map_params['origin'][:, 1]) / self.planner.primitives['lindisc']).round().clip(0, self.planner.states.shape[1]-1)
        sga = (initial_pos[:, [2]] - angles).abs().argmin(dim=-1) % self.planner.primitives['angnum']
        start_state = torch.stack([sgx, sgy, sga], dim=-1).long()

        # setup goal state
        ggx = ((goal_pos[:, 0] - map_params['origin'][:, 0]) / self.planner.primitives['lindisc']).round().clip(0, self.planner.states.shape[0]-1)
        ggy = ((goal_pos[:, 1] - map_params['origin'][:, 1]) / self.planner.primitives['lindisc']).round().clip(0, self.planner.states.shape[1]-1)
        gga = (goal_pos[:, [2]] - angles).abs().argmin(dim=-1) % self.planner.primitives['angnum']

        goal_state = torch.stack([ggx, ggy, gga], dim=-1).long()

        #afaik, this op is not batch-able because of torch.bincount
        #so just loop it - this is not the speed bottleneck
        learner_state_visitations = []
        expert_state_visitations = []
        for bi in range(start_state.shape[0]):
            map_params_b = {
                'resolution': batch['metadata']['resolution'].mean().item(),
                'height': batch['metadata']['height'].mean().item(),
                'width': batch['metadata']['width'].mean().item(),
                'origin': batch['metadata']['origin'][bi]
            }

            self.planner.load_costmap(costmaps[bi].T, length_weight=self.length_weight)
            solution = self.planner.solve(goal_states = goal_state[bi].unsqueeze(0), max_itrs=1000)
            traj = self.planner.extract_solution_parallel(solution, start_state[bi].unsqueeze(0))[0].float()
            traj[:, 0] += initial_pos[bi, 0]
            traj[:, 1] += initial_pos[bi, 1]

#            footprint_learner_traj = apply_footprint(traj.unsqueeze(0), self.footprint).view(1, -1, 2)
#            footprint_expert_traj = apply_footprint(expert_kbm_traj[bi].unsqueeze(0), self.footprint).view(1, -1, 2)

            footprint_learner_traj = traj.unsqueeze(0)
            footprint_expert_traj = expert_kbm_traj[bi].unsqueeze(0)

            lsv = get_state_visitations(footprint_learner_traj, map_params_b)
            esv = get_state_visitations(footprint_expert_traj, map_params_b)
            learner_state_visitations.append(lsv)
            expert_state_visitations.append(esv)

            """
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='r')
            axs[0].imshow(lsv.cpu(), origin='lower', extent=(
                map_params_b['origin'][0].item(),
                map_params_b['origin'][0].item() + map_params_b['height'],
                map_params_b['origin'][1].item(),
                map_params_b['origin'][1].item() + map_params_b['width'],
            ))
            axs[0].set_title('learner')

            axs[1].plot(expert_traj[bi][:, 0].cpu(), expert_traj[bi][:, 1].cpu(), c='r')
            axs[1].imshow(esv.cpu(), origin='lower', extent=(
                map_params_b['origin'][0].item(),
                map_params_b['origin'][0].item() + map_params_b['height'],
                map_params_b['origin'][1].item(),
                map_params_b['origin'][1].item() + map_params_b['width'],
            ))
            axs[1].set_title('expert')
            plt.show()
            """

        learner_state_visitations = torch.stack(learner_state_visitations, dim=0)
        expert_state_visitations = torch.stack(expert_state_visitations, dim=0)

        grads = (expert_state_visitations - learner_state_visitations) / initial_pos.shape[0]

        if not torch.isfinite(grads).all():
            import pdb;pdb.set_trace()

        #Speedmaps here:
        expert_speedmaps = []
        for bi in range(initial_pos.shape[0]):
            map_params_b = {
                'resolution': batch['metadata']['resolution'].mean().item(),
                'height': batch['metadata']['height'].mean().item(),
                'width': batch['metadata']['width'].mean().item(),
                'origin': batch['metadata']['origin'][bi]
            }
            #no footprint
            epos = expert_traj[bi][:, :2].unsqueeze(0)
            espeeds = torch.linalg.norm(expert_traj[bi][:, 7:10], dim=-1).unsqueeze(0)
            esm = get_speedmap(epos, espeeds, map_params_b).view(costmaps[bi].shape)

            #footprint
#            epos = apply_footprint(expert_kbm_traj[bi].unsqueeze(0), self.footprint).view(1, -1, 2)
#            espeeds = torch.linalg.norm(expert_traj[bi][:, 7:10], dim=-1).unsqueeze(0)
#            espeeds = espeeds.unsqueeze(2).tile(1, 1, len(self.footprint)).view(1, -1)
#            esm = get_speedmap(epos, espeeds, map_params_b).view(costmaps[bi].shape)

            expert_speedmaps.append(esm)

        expert_speedmaps = torch.stack(expert_speedmaps, dim=0)

        if self.categorical_speedmaps:
            #bin expert speeds
            _sbins = self.network.speed_bins[:-1].to(self.device).view(1, -1, 1, 1)
            sdiffs = expert_speedmaps.unsqueeze(1) - _sbins
            sdiffs[sdiffs < 0] = 1e10
            expert_speed_idxs = sdiffs.argmin(dim=1)

            mask = (expert_speedmaps > 1e-2) #only want the cells that the expert drove in
            ce = torch.nn.functional.cross_entropy(speedmaps, expert_speed_idxs, reduction='none')[mask]

#            speed_loss = ce.mean() * self.speed_coeff


            #try regularizing speeds to zero
            neg_labels = torch.zeros_like(expert_speed_idxs)
            ce_neg = torch.nn.functional.cross_entropy(speedmaps, neg_labels, reduction='none')[~mask]

            speed_loss = ce.mean() * self.speed_coeff + 0.1 * ce_neg.mean()

        else:
            mask = (expert_speedmaps > 1e-2) #only need the cells that the expert drove in
            ll = -speedmaps.log_prob(expert_speedmaps)[mask]
            speed_loss = ll.mean() * self.speed_coeff

#        print('IRL GRAD:   {:.4f}'.format(torch.linalg.norm(grads).detach().cpu().item()))
#        print('SPEED LOSS: {:.4f}'.format(speed_loss.detach().item()))

        #add regularization
        reg = self.reg_coeff * costmaps

        #kinda jank, but since we're multi-headed and have a loss and a gradient,
        # I think we need two backward passes through the computation graph.
        self.network_opt.zero_grad()
        costmaps.backward(gradient=(grads + reg), retain_graph=True)
        speed_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.network_opt.step()

    def visualize(self, idx=-1):
        """
        Create a visualization of MaxEnt IRL inputs/outputs for the idx-th datapoint.
        """
        if idx == -1:
            idx = np.random.randint(len(self.expert_dataset))

        with torch.no_grad():
            data = self.expert_dataset[idx]

            #hack back to single dim
            map_features = data['map_features'].unsqueeze(0)
            metadata = data['metadata']
            xmin = metadata['origin'][0].cpu()
            ymin = metadata['origin'][1].cpu()
            xmax = xmin + metadata['length_x'].cpu()
            ymax = ymin + metadata['length_y'].cpu()

            expert_traj = data['traj']
            expert_kbm_traj = torch.stack([
                expert_traj[:, 0],
                expert_traj[:, 1],
                self.quat_to_yaw(expert_traj[:, 3:7]) % (2*np.pi)
            ], dim=-1)

            #compute costmap
            #resnet cnn
            if hasattr(self.network, 'ensemble_forward'):
                res = self.network.ensemble_forward(map_features)
                costmap = res['costmap'].mean(dim=1)[:, 0]

                if self.categorical_speedmaps:
                    _speeds = self.network.speed_bins[1:].to(self.device).view(1, -1, 1, 1)
                    speedmap_dist = res['speedmap'][0].softmax(dim=1)
                    speedmap_val = (_speeds * speedmap_dist).sum(dim=1).mean(dim=0)
                    speedmap_unc = (-speedmap_dist * speedmap_dist.log()).sum(dim=1).mean(dim=0)
                else:
                    speedmap_val = res['speedmap'].loc[0].mean(dim=0)
                    speedmap_unc = res['speedmap'].scale[0].mean(dim=0)

            else:
                print('non ensemble currently broken')
                exit(1)
                res = self.network.forward(map_features)
                costmap = res['costmap'][:, 0]


                if self.categorical_speedmaps:
                    speedmap_val = None
                    speedmap_unc = None
                else:
                    import pdb;pdb.set_trace()

            #initialize solver
            initial_pos = expert_kbm_traj[0]
            goal_pos = self.clip_to_map_bounds(expert_kbm_traj, metadata)

            # setup start state
            angles = torch.linspace(0., 2*np.pi, self.planner.primitives['angnum']+1, device=self.planner.device)

            sgx = ((initial_pos[0] - metadata['origin'][0]) / self.planner.primitives['lindisc']).round().clip(0, self.planner.states.shape[0]-1)
            sgy = ((initial_pos[1] - metadata['origin'][1]) / self.planner.primitives['lindisc']).round().clip(0, self.planner.states.shape[1]-1)
            sga = (initial_pos[2] - angles).abs().argmin() % self.planner.primitives['angnum']
            start_state = torch.stack([sgx, sgy, sga], dim=-1).long()

            # setup goal state
            ggx = ((goal_pos[0] - metadata['origin'][0]) / self.planner.primitives['lindisc']).round().clip(0, self.planner.states.shape[0]-1)
            ggy = ((goal_pos[1] - metadata['origin'][1]) / self.planner.primitives['lindisc']).round().clip(0, self.planner.states.shape[1]-1)
            gga = (goal_pos[2] - angles).abs().argmin() % self.planner.primitives['angnum']

            goal_state = torch.stack([ggx, ggy, gga], dim=-1).long()

#            plt.show()

            self.planner.load_costmap(costmap[0].T, length_weight=self.length_weight)
            solution = self.planner.solve(goal_states = goal_state.unsqueeze(0), max_itrs=1000)
            traj = self.planner.extract_solution_parallel(solution, start_state.unsqueeze(0))[0]
            traj[:, 0] += initial_pos[0]
            traj[:, 1] += initial_pos[1]

            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            axs = axs.flatten()
            
            fk = None
            fklist = ['height_high', 'step', 'diff']
            for f in fklist:
                if f in self.expert_dataset.feature_keys:
                    fk = f
                    idx = self.expert_dataset.feature_keys.index(fk)
                    break

            img = data['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu()
            traj_viz = torch.clone(traj)
            traj_viz[:, 2] = expert_traj[0, 2]
#            expert_traj_px = self.project_traj_into_fpv(expert_traj.cpu().numpy(), expert_traj[0].cpu().numpy(), img)
#            traj_px = self.project_traj_into_fpv(traj_viz.cpu().numpy(), expert_traj[0].cpu().numpy(), img)

            axs[0].imshow(img)
            axs[1].imshow(map_features[0][idx].cpu(), origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))
#            m1 = axs[2].imshow(costmap.mean(dim=0).cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax), vmin=0., vmax=2.)
            m1 = axs[2].imshow(costmap[0].cpu(), origin='lower', cmap='jet', extent=(xmin, xmax, ymin, ymax))
            m3 = axs[4].imshow(speedmap_val.cpu(), origin='lower', cmap='jet', extent=(xmin, xmax, ymin, ymax), vmin=0., vmax=10.)
            m4 = axs[5].imshow(speedmap_unc.cpu(), origin='lower', cmap='viridis', extent=(xmin, xmax, ymin, ymax))

#            axs[0].plot(expert_traj_px[:, 0], expert_traj_px[:, 1], c='y')
            axs[1].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y', label='expert')
            axs[2].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y')
            axs[3].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y')
            axs[4].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y')
            axs[5].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y')

#            axs[0].plot(traj_px[:, 0], traj_px[:, 1], c='g')
            axs[1].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g', label='learner')
            axs[2].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g')
            axs[3].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g')
            axs[4].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g')
            axs[5].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g')

            for ax in axs[1:]:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

            axs[0].set_title('FPV')
            axs[1].set_title('gridmap {}'.format(fk))
            axs[2].set_title('irl cost (clipped)')
            axs[3].set_title('speedmap lcb (z=-1)')
            axs[4].set_title('speedmap mean')
            axs[5].set_title('speedmap unc')

            for i in [1, 2, 4, 5]:
                axs[i].set_xlabel('X(m)')
                axs[i].set_ylabel('Y(m)')

            axs[1].legend()

            plt.colorbar(m1, ax=axs[2])
            plt.colorbar(m3, ax=axs[4])
            plt.colorbar(m4, ax=axs[5])
        return fig, axs

    def project_traj_into_fpv(self, traj, init_pose, image):
        """
        be fancy and project paths into image frame
        Args:
            traj: the traj to project
            init_pose: the pose [p; q] to transform pose into local frame
        """
        K = np.array(self.expert_dataset.config['intrinsics']['K']).reshape(3, 3)
        I = get_intrinsics(K)
        E = np.eye(4)
        E[:3,:3] = Rotation.from_quat(np.array(self.expert_dataset.config['extrinsics']['q'])).as_matrix()
        E[:3, -1] = np.array(self.expert_dataset.config['extrinsics']['p'])
        E = get_extrinsics(E)
        P = obtain_projection_matrix(I, E)

        init_pose_htm = np.eye(4)
        init_pose_htm[:3, -1] = init_pose[:3]
        init_pose_htm[:3, :3] = Rotation.from_quat(init_pose[3:7]).as_matrix()
        traj_pts = np.concatenate([
            traj[:, :3],
            np.ones([traj.shape[0], 1])
        ], axis=-1)
        local_traj_pts = np.linalg.inv(init_pose_htm).reshape(1, 4, 4) @ traj_pts.reshape(-1, 4, 1)
        local_traj_pts = local_traj_pts[:, :3, 0]

        pixel_coords = get_pixel_from_3D_source(local_traj_pts, P)
        traj_pts_in_frame, pixels_in_frame, ind_in_frame = get_points_and_pixels_in_frame(
            local_traj_pts,
            pixel_coords,
            image.shape[0],
            image.shape[1]
        )

        return pixels_in_frame

    def quat_to_yaw(self, q):
        #quats are x,y,z,w
        qx, qy, qz, qw = q.moveaxis(-1, 0)
        return torch.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))

    def clip_to_map_bounds(self, traj, metadata):
        """
        Given traj, find last point (temporally) in the map bounds
        """
        ox = metadata['origin'][0]
        oy = metadata['origin'][1]
        lx = metadata['height']
        ly = metadata['width']
        xs = traj[:, 0]
        ys = traj[:, 1]

        in_bounds = (xs > ox) & (xs < ox+lx) & (ys > oy) & (ys < oy+ly)
        idx = (torch.arange(in_bounds.shape[0], device=self.device) * in_bounds).argmax(dim=0)

        return traj[idx]

    def to(self, device):
        self.device = device
        self.expert_dataset = self.expert_dataset.to(device)
        self.planner = self.planner.to(device)
        self.network = self.network.to(device)
        self.footprint = self.footprint.to(device)
        return self
