import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import scipy.spatial
import scipy.interpolate

from torch.utils.data import DataLoader

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.costmappers.linear_costmapper import LinearCostMapper
from maxent_irl_costmaps.utils import get_state_visitations, get_speedmap
from maxent_irl_costmaps.geometry_utils import apply_footprint

from maxent_irl_costmaps.networks.mlp import MLP
from maxent_irl_costmaps.networks.resnet import ResnetCostmapCNN

class MPPIIRLSpeedmaps:
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
    def __init__(self, network, opt, expert_dataset, mppi, footprint, mppi_itrs=10, batch_size=64, speed_coeff=1.0, reg_coeff=1e-2, grad_clip=1., device='cpu'):
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
        self.mppi = mppi
        self.mppi_itrs = mppi_itrs

        self.network = network

        print(self.network)
        print(sum([x.numel() for x in self.network.parameters()]))
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
        goals = [traj[[-1], :2] for traj in expert_traj]

        #initialize initial state for MPPI
        initial_states = expert_traj[:, 0]
        x0 = {
            'state': initial_states,
            'steer_angle': batch['steer'][:, [0]] if 'steer' in batch.keys() else torch.zeros(initial_states.shape[0], device=initial_states.device)
        }
        x = self.mppi.model.get_observations(x0)

        #also get KBM states for expert
        X_expert = {
            'state': expert_traj,
            'steer_angle': batch['steer'].unsqueeze(-1) if 'steer' in batch.keys() else torch.zeros(self.mppi.B, self.mppi.T, 1, device=initial_states.device)
        }
        expert_kbm_traj = self.mppi.model.get_observations(X_expert)

        #set up the solver
        self.mppi.reset()
        self.mppi.cost_fn.data['goals'] = goals
        self.mppi.cost_fn.data['costmap'] = costmaps
        self.mppi.cost_fn.data['costmap_metadata'] = map_params

        #TEMP HACK - Sam TODO: clean up once data is on cluster
        self.mppi.cost_fn.data['costmap_metadata']['length_x'] = self.mppi.cost_fn.data['costmap_metadata']['width']
        self.mppi.cost_fn.data['costmap_metadata']['length_y'] = self.mppi.cost_fn.data['costmap_metadata']['height']

        #run MPPI
        for ii in range(self.mppi_itrs):
            with torch.no_grad():
                self.mppi.get_control(x, step=False)

        #weighting version
        trajs = self.mppi.noisy_states.clone()
        weights = self.mppi.last_weights.clone()

        self.mppi.reset()

        #afaik, this op is not batch-able because of torch.bincount
        #so just loop it - this is not the speed bottleneck
        learner_state_visitations = []
        expert_state_visitations = []
        for bi in range(trajs.shape[0]):
            map_params_b = {
                'resolution': batch['metadata']['resolution'].mean().item(),
                'height': batch['metadata']['height'].mean().item(),
                'width': batch['metadata']['width'].mean().item(),
                'origin': batch['metadata']['origin'][bi]
            }

            footprint_learner_traj = apply_footprint(trajs[bi], self.footprint).view(self.mppi.K, -1, 2)
            footprint_expert_traj = apply_footprint(expert_kbm_traj[bi].unsqueeze(0), self.footprint).view(1, -1, 2)

            lsv = get_state_visitations(footprint_learner_traj, map_params_b, weights[bi])
            esv = get_state_visitations(footprint_expert_traj, map_params_b)
            learner_state_visitations.append(lsv)
            expert_state_visitations.append(esv)

            """
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(trajs[bi][weights[bi].argmax(), :, 0].cpu(), trajs[bi][weights[bi].argmax(), :, 1].cpu(), c='r')
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

        grads = (expert_state_visitations - learner_state_visitations) / trajs.shape[0]

        if not torch.isfinite(grads).all():
            import pdb;pdb.set_trace()

        #Speedmaps here:
        expert_speedmaps = []
        for bi in range(trajs.shape[0]):
            map_params_b = {
                'resolution': batch['metadata']['resolution'].mean().item(),
                'height': batch['metadata']['height'].mean().item(),
                'width': batch['metadata']['width'].mean().item(),
                'origin': batch['metadata']['origin'][bi]
            }
            esm = get_speedmap(expert_traj[bi].unsqueeze(0), map_params_b).view(speedmaps.loc[bi].shape)
            expert_speedmaps.append(esm)

        expert_speedmaps = torch.stack(expert_speedmaps, dim=0)

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
            map_features = torch.stack([data['map_features']] * self.mppi.B, dim=0)
            metadata = data['metadata']
            xmin = metadata['origin'][0].cpu()
            ymin = metadata['origin'][1].cpu()
            xmax = xmin + metadata['width'].cpu()
            ymax = ymin + metadata['height'].cpu()
            expert_traj = data['traj']

            #compute costmap
            #resnet cnn
            res = self.network.forward(map_features)
            costmap = res['costmap'][:, 0]
            speedmap = torch.distributions.Normal(loc=res['speedmap'].loc, scale=res['speedmap'].scale)

            #initialize solver
            initial_state = expert_traj[0]
            x0 = {"state":initial_state, "steer_angle":data["steer"][[0]] if 'steer' in data.keys() else torch.zeros(1, device=initial_state.device)}
            x = torch.stack([self.mppi.model.get_observations(x0)] * self.mppi.B, dim=0)

            map_params = {
                'resolution': torch.tensor([metadata['resolution']] * self.mppi.B, device=self.mppi.device),
                'height': torch.tensor([metadata['height']] * self.mppi.B, device=self.mppi.device),
                'width': torch.tensor([metadata['width']] * self.mppi.B, device=self.mppi.device),
                'origin': torch.stack([metadata['origin']] * self.mppi.B, dim=0)
            }

            goals = [expert_traj[[-1], :2]] * self.mppi.B

            self.mppi.reset()
            self.mppi.cost_fn.data['goals'] = goals
            self.mppi.cost_fn.data['costmap'] = costmap
            self.mppi.cost_fn.data['costmap_metadata'] = map_params

            #TEMP HACK - Sam TODO: clean up once data is on cluster
            self.mppi.cost_fn.data['costmap_metadata']['length_x'] = self.mppi.cost_fn.data['costmap_metadata']['width']
            self.mppi.cost_fn.data['costmap_metadata']['length_y'] = self.mppi.cost_fn.data['costmap_metadata']['height']

            #solve for traj
            for ii in range(self.mppi_itrs):
                self.mppi.get_control(x, step=False)

            tidx = self.mppi.last_cost.argmin()
            traj = self.mppi.last_states[tidx]

            metadata = data['metadata']
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            axs = axs.flatten()
            
            idx = self.expert_dataset.feature_keys.index('height_high')
            
            axs[0].imshow(data['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            axs[1].imshow(map_features[tidx][idx].cpu(), origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))
            m1 = axs[2].imshow(costmap[tidx].cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax), vmin=-5., vmax=5.)
#            m1 = axs[2].imshow(costmap[tidx].cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax))
            m2 = axs[4].imshow(speedmap.loc[tidx].cpu(), origin='lower', cmap='bwr', extent=(xmin, xmax, ymin, ymax), vmax=10.)
            m3 = axs[5].imshow(speedmap.scale[tidx].cpu(), origin='lower', cmap='bwr', extent=(xmin, xmax, ymin, ymax), vmax=10.)

            axs[1].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y', label='expert')
            axs[2].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y')
            axs[4].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y')
            axs[5].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y')

            axs[1].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g', label='learner')
            axs[2].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g')
            axs[4].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g')
            axs[5].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g')

            #plot expert speed
            e_speeds = torch.linalg.norm(expert_traj[:, 7:10], axis=-1).cpu()
            l_speeds = traj[:, 3].cpu()
            times = torch.arange(len(e_speeds)) * self.mppi.model.dt
            axs[3].plot(times, e_speeds, label='expert speed', c='y')
            axs[3].plot(times, l_speeds, label='learner speed', c='g')

            axs[0].set_title('FPV')
            axs[1].set_title('heightmap high')
            axs[2].set_title('irl cost (clipped)')
            axs[3].set_title('speed')
            axs[4].set_title('speedmap mean')
            axs[5].set_title('speedmap std')

            for i in [1, 2, 4, 5]:
                axs[i].set_xlabel('X(m)')
                axs[i].set_ylabel('Y(m)')

            axs[3].set_xlabel('T(s)')
            axs[3].set_ylabel('Speed (m/s)')
            axs[3].legend()

            axs[1].legend()

            plt.colorbar(m1, ax=axs[2])
            plt.colorbar(m2, ax=axs[4])
            plt.colorbar(m3, ax=axs[5])
        return fig, axs

    def to(self, device):
        self.device = device
        self.expert_dataset = self.expert_dataset.to(device)
        self.mppi = self.mppi.to(device)
        self.network = self.network.to(device)
        self.footprint = self.footprint.to(device)
        return self
