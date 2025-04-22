import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from ros_torch_converter.datatypes.transform import TransformTorch
from ros_torch_converter.datatypes.intrinsics import IntrinsicsTorch

from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelGrid
from physics_atv_visual_mapping.utils import *

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_maps.utils import get_state_visitations, get_speedmap, compute_map_mean_entropy
from maxent_irl_maps.geometry_utils import apply_footprint

class MPPIIRLSpeedmapsTorchCoordinator:
    """
    Same as MPPIIRLSpeedmaps, but instead of gridmap inputs, we're learning through the whole mapping process

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

    def __init__(
        self,
        torch_coordinator,
        bev_network,
        fpv_network,
        bev_opt,
        fpv_opt,
        expert_dataset,
        mppi,
        footprint,
        mppi_itrs=10,
        batch_size=64,
        speed_coeff=1.0,
        reg_coeff=1e-2,
        grad_clip=1.0,
        device="cpu",
    ):
        """
        Args:
            network: the torch_coordinator setup
            opt: the optimizer for the network
            expert_dataset: The dataset containing expert demonstrations to imitate
            footprint: "smear" state visitations with this
            mppi: The MPPI object to optimize with
        """
        self.expert_dataset = expert_dataset
        self.footprint = footprint
        self.mppi = mppi
        self.mppi_itrs = mppi_itrs

        self.torch_coordinator = torch_coordinator

        self.bev_network = bev_network
        self.bev_netopt = bev_opt

        self.fpv_network = fpv_network
        self.fpv_netopt = fpv_opt

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

        sample_idxs = np.random.permutation(len(self.expert_dataset))
        if n != -1 and n < len(sample_idxs):
            sample_idxs = sample_idxs[:n]

        for i in tqdm.tqdm(sample_idxs):
            batch = self.expert_dataset[i]

            # print(
            #     "{}/{}".format(i + 1, int(len(self.expert_dataset) / self.batch_size)),
            #     end="\r",
            # )
            self.gradient_step(batch)

        print("_____ITR {}_____".format(self.itr))

    def gradient_step(self, batch):
        """
        Apply the MaxEnt update to the network given a batch
        """
        grads = []
        speed_loss = []

        self.run_torch_coordinator(batch)
        expert_traj, expert_kbm_traj = self.setup_expert_traj(batch)

        bev_map = self.torch_coordinator.data['bev_map'].bev_grid
        nav_map = self.torch_coordinator.data['nav_map'].bev_grid
        metadata = nav_map.metadata
        xmin = metadata.origin[0].cpu()
        ymin = metadata.origin[1].cpu()
        xmax = xmin + metadata.length[0].cpu()
        ymax = ymin + metadata.length[1].cpu()

        heightmap = bev_map.data[..., bev_map.feature_keys.index('max_elevation')]
        costmap = nav_map.data[..., nav_map.feature_keys.index('cost')]
        speedmap = nav_map.data[..., nav_map.feature_keys.index('speed')]

        speed_logit_idxs = sorted([i for i in range(len(nav_map.feature_keys)) if 'speed_logit' in nav_map.feature_keys[i]])
        speed_logits = nav_map.data[..., speed_logit_idxs]

        # initialize solver
        initial_state = expert_kbm_traj[0]
        x = torch.stack([initial_state] * self.mppi.B, dim=0)

        map_params = {
            "origin": torch.stack([metadata.origin] * self.mppi.B, dim=0),
            "length": torch.stack([metadata.length] * self.mppi.B, dim=0),
            "resolution": torch.stack([metadata.resolution] * self.mppi.B, dim=0),
        }

        goals = [
            self.clip_to_map_bounds(expert_traj[:, :2], metadata).view(1, 2)
        ] * self.mppi.B

        self.mppi.reset()
        self.mppi.cost_fn.data["waypoints"] = goals
        self.mppi.cost_fn.data["local_navmap"] = {
            "data": costmap.unsqueeze(0).unsqueeze(0),
            "metadata": map_params,
            "feature_keys": ["cost"]
        }

        # solve for traj
        for ii in range(self.mppi_itrs):
            with torch.no_grad():
                self.mppi.get_control(x, step=False)

        # weighting version
        trajs = self.mppi.noisy_states.clone()
        weights = self.mppi.last_weights.clone()

        # afaik, this op is not batch-able because of torch.bincount
        # so just loop it - this is not the speed bottleneck
        learner_state_visitations = []
        expert_state_visitations = []

        footprint_learner_traj = apply_footprint(trajs[0], self.footprint).view(
            self.mppi.K, -1, 2
        )
        footprint_expert_traj = apply_footprint(
            expert_kbm_traj.unsqueeze(0), self.footprint
        ).view(1, -1, 2)

        lsv = get_state_visitations(
            footprint_learner_traj, metadata, weights[0]
        )
        esv = get_state_visitations(footprint_expert_traj, metadata)

        # learner_state_visitations.append(lsv)
        # expert_state_visitations.append(esv)
        # learner_state_visitations = torch.stack(learner_state_visitations, dim=0)
        # expert_state_visitations = torch.stack(expert_state_visitations, dim=0)

        """
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(trajs[0][weights[0].argmax(), :, 0].cpu(), trajs[0][weights[0].argmax(), :, 1].cpu(), c='r')
        axs[0].imshow(lsv.T.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
        axs[0].set_title('learner')

        axs[1].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='r')
        axs[1].imshow(esv.T.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
        axs[1].set_title('expert')
        plt.show()
        """

        grads = esv - lsv

        if not torch.isfinite(grads).all():
            import pdb; pdb.set_trace()

        # Speedmaps here:
        expert_speedmaps = []
    
        # footprint
        epos = apply_footprint(
            expert_kbm_traj.unsqueeze(0), self.footprint
        ).view(1, -1, 2)
        espeeds = torch.linalg.norm(expert_traj[:, 7:10], dim=-1).unsqueeze(0)
        espeeds = espeeds.unsqueeze(2).tile(1, 1, len(self.footprint)).view(1, -1)
        esm = get_speedmap(epos, espeeds, metadata)

        expert_speedmaps.append(esm)

        expert_speedmaps = torch.stack(expert_speedmaps, dim=0)

        # speedmap_probs = speed_logits.softmax(axis=-1)
        speed_logits = speed_logits.permute(2,0,1).unsqueeze(0)

        # bin expert speeds
        _sbins = self.bev_network.speed_bins[:-1].to(self.device).view(1, -1, 1, 1)
        sdiffs = expert_speedmaps - _sbins
        sdiffs[sdiffs < 0] = 1e10
        expert_speed_idxs = sdiffs.argmin(dim=1) + 1
        expert_speed_idxs = expert_speed_idxs.clip(0, self.bev_network.speed_nbins-1).long()

        #debug viz
        """
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(expert_kbm_traj[:, 0].cpu(), expert_kbm_traj[:, 1].cpu(), c='r')
        axs[0].imshow(esm.T.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
        axs[0].set_title('expert speedmap')

        axs[1].plot(expert_kbm_traj[:, 0].cpu(), expert_kbm_traj[:, 1].cpu(), c='r')
        axs[1].imshow(expert_speed_idxs.T.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
        axs[1].set_title('expert speedmap idx')
        plt.show()
        """

        mask = (
            expert_speedmaps > 1e-6
        )  # only want the cells that the expert drove in
        ce = torch.nn.functional.cross_entropy(
            speed_logits, expert_speed_idxs, reduction="none"
        )[mask]

        # try regularizing speeds to zero
        neg_labels = torch.zeros_like(expert_speed_idxs)
        ce_neg = torch.nn.functional.cross_entropy(
            speed_logits, neg_labels, reduction="none"
        )[~mask]

        neg_ratio = mask.sum() / (~mask | mask).sum()

        speed_loss = self.speed_coeff * (ce.mean() + 0.1 * neg_ratio * ce_neg.mean())

        # print('IRL GRAD:   {:.4f}'.format(torch.linalg.norm(grads).detach().cpu().item()))
        # print('SPEED LOSS: {:.4f}'.format(speed_loss.detach().item()))

        # add regularization
        reg = self.reg_coeff * costmap

        # kinda jank, but since we're multi-headed and have a loss and a gradient,
        # I think we need two backward passes through the computation graph.
        self.bev_netopt.zero_grad()
        self.fpv_netopt.zero_grad()

        costmap.backward(gradient=(grads + reg), retain_graph=True)
        speed_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.bev_network.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.fpv_network.parameters(), self.grad_clip)

        self.bev_netopt.step()
        self.fpv_netopt.step()

    def run_torch_coordinator(self, data):
        """
        Get torch coordinator outputs
        """
        perception_data = data['perception']
        T = len(perception_data['state'])

        #hack to clear state in torch coordinator
        self.torch_coordinator.data = {}
        self.torch_coordinator.nodes['voxel_mapper'].localmapper.voxel_grid = VoxelGrid(
            self.torch_coordinator.nodes['voxel_mapper'].localmapper.metadata,
            self.torch_coordinator.nodes['voxel_mapper'].localmapper.n_features,
            self.torch_coordinator.nodes['voxel_mapper'].localmapper.device
        )

        #temp hack just to get things working
        intrinsics = (
            torch.tensor([455.7750, 0., 497.1180, 0., 456.3191, 251.8580, 0., 0., 1.]).reshape(3, 3).to(self.device)
        )
        extrinsics = pose_to_htm(
            np.array([0.17265, -0.15227, 0.05708, 0.55940, -0.54718, 0.44603, 0.43442])
        ).to(self.device)

        for t in range(T):
            pose_H = pose_to_htm(perception_data['state'][t].state.cpu().numpy()).to(self.device)

            self.torch_coordinator.data['odometry'] = perception_data['state'][t]
            self.torch_coordinator.data['odometry'].frame_id = "odom"

            self.torch_coordinator.data['tf_odom_to_base'] = TransformTorch.from_torch(pose_H, "vehicle")
            self.torch_coordinator.data['tf_odom_to_base'].stamp = perception_data['state'][t].stamp
            self.torch_coordinator.data['tf_odom_to_base'].frame_id = "odom"

            self.torch_coordinator.data['tf_base_to_odom'] = TransformTorch.from_torch(torch.linalg.inv(pose_H), "odom")
            self.torch_coordinator.data['tf_base_to_odom'].stamp = perception_data['state'][t].stamp
            self.torch_coordinator.data['tf_base_to_odom'].frame_id = "vehicle"

            self.torch_coordinator.data['tf_base_to_cam'] = TransformTorch.from_torch(extrinsics, "camera")
            self.torch_coordinator.data['tf_base_to_cam'].stamp = perception_data['state'][t].stamp
            self.torch_coordinator.data['tf_base_to_cam'].frame_id = "vehicle"

            self.torch_coordinator.data['image_intrinsics'] = IntrinsicsTorch.from_torch(intrinsics)
            self.torch_coordinator.data['image_intrinsics'].stamp = perception_data['state'][t].stamp
            self.torch_coordinator.data['image_intrinsics'].frame_id = ""

            self.torch_coordinator.data['image'] = perception_data["image"][t]
            self.torch_coordinator.data['image'].frame_id = "camera"

            self.torch_coordinator.data['pointcloud_in_odom'] = perception_data["pointcloud"][t]
            self.torch_coordinator.data['pointcloud_in_odom'].frame_id = "odom"

            timing_stats = self.torch_coordinator.run()

    def setup_expert_traj(self, data):
        expert_traj = []
        steer_angles = []
        supervision_data = data["supervision"]
        T = len(supervision_data["state"])

        for t in range(T):
            expert_traj.append(supervision_data["state"][t].state)
            steer_angles.append(supervision_data["steer"][t].data)

        expert_traj = torch.stack(expert_traj, dim=0)
        steers = torch.stack(steer_angles, dim=0)

        X_expert = {
            "state": expert_traj,
            "steer_angle": steers.unsqueeze(-1)
        }
        expert_kbm_traj = self.mppi.model.get_observations(X_expert)

        return expert_traj, expert_kbm_traj

    def visualize(self, idx=-1):
        """
        Create a visualization of MaxEnt IRL inputs/outputs for the idx-th datapoint.
        """
        if idx == -1:
            idx = np.random.randint(len(self.expert_dataset))

        with torch.no_grad():
            data = self.expert_dataset[idx]

            self.run_torch_coordinator(data)
            expert_traj, expert_kbm_traj = self.setup_expert_traj(data)

            bev_map = self.torch_coordinator.data['bev_map'].bev_grid
            nav_map = self.torch_coordinator.data['nav_map'].bev_grid
            metadata = nav_map.metadata
            xmin = metadata.origin[0].cpu()
            ymin = metadata.origin[1].cpu()
            xmax = xmin + metadata.length[0].cpu()
            ymax = ymin + metadata.length[1].cpu()

            heightmap = bev_map.data[..., bev_map.feature_keys.index('max_elevation')]
            costmap = nav_map.data[..., nav_map.feature_keys.index('cost')]
            speedmap = nav_map.data[..., nav_map.feature_keys.index('speed')]

            # initialize solver
            initial_state = expert_kbm_traj[0]
            x = torch.stack([initial_state] * self.mppi.B, dim=0)

            map_params = {
                "origin": torch.stack([metadata.origin] * self.mppi.B, dim=0),
                "length": torch.stack([metadata.length] * self.mppi.B, dim=0),
                "resolution": torch.stack([metadata.resolution] * self.mppi.B, dim=0),
            }

            goals = [
                self.clip_to_map_bounds(expert_traj[:, :2], metadata).view(1, 2)
            ] * self.mppi.B

            self.mppi.reset()
            self.mppi.cost_fn.data["waypoints"] = goals
            self.mppi.cost_fn.data["local_navmap"] = {
                "data": costmap.unsqueeze(0).unsqueeze(0),
                "metadata": map_params,
                "feature_keys": ["cost"]
            }

            # solve for traj
            for ii in range(self.mppi_itrs):
                self.mppi.get_control(x, step=False)

            tidx = self.mppi.last_cost.argmin()
            traj = self.mppi.last_states[tidx]

            footprint_learner_traj = apply_footprint(traj, self.footprint).view(1, -1, 2)
            footprint_expert_traj = apply_footprint(
                expert_kbm_traj.unsqueeze(0), self.footprint
            ).view(1, -1, 2)

            lsv = get_state_visitations(footprint_learner_traj, metadata)
            esv = get_state_visitations(footprint_expert_traj, metadata)

            learner_cost = (lsv * costmap).sum()
            expert_cost = (esv * costmap).sum()

            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            axs = axs.flatten()

            img = self.torch_coordinator.data["image"].image.cpu()
            feat_img = self.torch_coordinator.data["feature_image"].image.cpu()

            #do a pca on feat img for viz
            feat_img_flat = feat_img.view(-1, feat_img.shape[-1])
            feat_img_mean = feat_img_flat.mean(dim=0)
            feat_img_norm = feat_img_flat - feat_img_mean.view(1, -1)
            U, S, V = torch.pca_lowrank(feat_img_norm, q=6)
            feat_img_pca = (feat_img_norm @ V).view(*feat_img.shape[:2], 6)

            fig.suptitle("Expert cost = {:.4f}, Learner cost = {:.4f}".format(expert_cost.item(), learner_cost.item()))

            axs[0].imshow(
                heightmap.T.cpu(),
                origin='lower',
                cmap='gray',
                extent=(xmin, xmax, ymin, ymax)
            )

            m1 = axs[1].imshow(
                costmap.T.cpu(),
                origin="lower",
                cmap="jet",
                extent=(xmin, xmax, ymin, ymax),
            )

            m2 = axs[2].imshow(
                speedmap.T.cpu(),
                origin="lower",
                cmap="jet",
                extent=(xmin, xmax, ymin, ymax),
                vmin=0.0,
                vmax=10.0,
            )

            axs[0].plot(
                expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c="y", label="expert"
            )
            axs[1].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c="y")
            axs[2].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c="y")

            axs[0].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c="g", label="learner")
            axs[1].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c="g")
            axs[2].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c="g")

            axs[3].imshow(img)
            axs[4].imshow(normalize_dino(feat_img[..., :3]))
            axs[5].imshow(normalize_dino(feat_img[..., 3:]))

            for ax in axs[:3]:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

            axs[0].set_title("elevation map")
            axs[1].set_title("irl cost mean")
            axs[2].set_title("speedmap mean")
            axs[3].set_title("fpv")
            axs[4].set_title("feat img 1-3")
            axs[5].set_title("feat img 4-6")

            for i in [1, 2]:
                axs[i].set_xlabel("X(m)")
                axs[i].set_ylabel("Y(m)")

            axs[0].legend()

            plt.colorbar(m1, ax=axs[1])
            plt.colorbar(m2, ax=axs[2])
            
        return fig, axs

    def clip_to_map_bounds(self, traj, metadata):
        """
        Given traj, find last point (temporally) in the map bounds
        """
        ox = metadata.origin[0]
        oy = metadata.origin[1]
        lx = metadata.length[0]
        ly = metadata.length[1]
        xs = traj[:, 0]
        ys = traj[:, 1]

        in_bounds = (xs > ox) & (xs < ox + lx) & (ys > oy) & (ys < oy + ly)
        idx = (torch.arange(in_bounds.shape[0], device=self.device) * in_bounds).argmax(
            dim=0
        )

        return traj[idx]

    def to(self, device):
        self.device = device
        self.expert_dataset = self.expert_dataset.to(device)
        self.mppi = self.mppi.to(device)
        self.torch_coordinator = self.torch_coordinator.to(device)
        self.footprint = self.footprint.to(device)
        return self
