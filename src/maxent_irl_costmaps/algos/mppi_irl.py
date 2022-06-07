import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import scipy.spatial
import scipy.interpolate

from torch.utils.data import DataLoader

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
from torch_mpc.algos.batch_mppi import BatchMPPI
from torch_mpc.algos.mppi import MPPI
from torch_mpc.cost_functions.waypoint_costmap import WaypointCostMapCostFunction

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.costmappers.linear_costmapper import LinearCostMapper
from maxent_irl_costmaps.utils import get_feature_counts, get_state_visitations

from maxent_irl_costmaps.networks.mlp import MLP
from maxent_irl_costmaps.networks.resnet import ResnetCostmapCNN

class MPPIIRL:
    """
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
    def __init__(self, expert_dataset, mppi, batch_size=64):
        """
        Args:
            expert_dataset: The dataset containing expert demonstrations to imitate
            mppi: The MPPI object to optimize with
        """
        self.expert_dataset = expert_dataset
        self.mppi = mppi
        self.mppi_itrs = 10

#        mlp_hiddens = [128, 128]
#        self.network = MLP(insize = len(expert_dataset.feature_keys), outsize=1, hiddens=mlp_hiddens)

        hiddens = [128,]
#        hiddens = []
        self.network = ResnetCostmapCNN(in_channels=len(expert_dataset.feature_keys), out_channels=1, hidden_channels=hiddens)

        print(sum([x.numel() for x in self.network.parameters()]))
        print(expert_dataset.feature_keys)

#        self.network.layers[-1].weight.fill_(0.)
#        print(list(self.network.parameters()))

#        self.network_opt = torch.optim.SGD(self.network.parameters(), lr=0.01)
        self.network_opt = torch.optim.Adam(self.network.parameters())

        self.batch_size = batch_size
        self.itr = 0

    def update(self):
        self.itr += 1
        dl = DataLoader(self.expert_dataset, batch_size=self.batch_size, shuffle=True)
        for i, batch in enumerate(dl):
            print('{}/{}'.format(i+1, int(len(self.expert_dataset)/self.batch_size)), end='\r')
            self.gradient_step(batch)

        print('_____ITR {}_____'.format(self.itr))

    def gradient_step(self, batch):
        grads = []
        efc = []
        lfc = []
        rfc = []
        contrastive_grads = []
        deep_features_cache = []

        for i in range(batch['traj'].shape[0]):
            map_features = batch['map_features'][i]
            map_metadata = {k:v[i] for k,v in batch['metadata'].items()}
            expert_traj = batch['traj'][i]

            #compute costmap
#            costmap = (map_features * self.weights.view(-1, 1, 1)).sum(dim=0)
#            costmap = self.costmapper.get_costmap(data)[0]

#            #MLP
#            costmap = torch.moveaxis(self.network.forward(torch.moveaxis(map_features, 0, -1)), -1, 0)[0]

            #resnet cnn
            costmap = self.network.forward(map_features.view(1, *map_features.shape))[0, 0]

            #initialize solver
            initial_state = expert_traj[0]
            HACK = {"state":initial_state, "steer_angle":torch.zeros(1)}
            x = self.mppi.model.get_observations(HACK)

            map_params = {
                'resolution': map_metadata['resolution'].item(),
                'height': map_metadata['height'].item(),
                'width': map_metadata['width'].item(),
                'origin': map_metadata['origin']
            }
            self.mppi.reset()
            self.mppi.cost_fn.update_map_params(map_params)
            self.mppi.cost_fn.update_costmap(costmap)
            self.mppi.cost_fn.update_goal(expert_traj[-1, :2])

            #solve for traj
            for ii in range(self.mppi_itrs):
                self.mppi.get_control(x, step=False)

            #regular version
            traj = self.mppi.last_states

            #weighting version
            trajs = self.mppi.noisy_states.clone()
            weights = self.mppi.last_weights.clone()

            self.mppi.reset()

            u_rand = torch.rand(self.mppi.K, self.mppi.T, self.mppi.umin.shape[0])
            u_rand = (u_rand * (self.mppi.umax - self.mppi.umin)) + self.mppi.umin
            traj_rand = self.mppi.model.rollout(x.unsqueeze(0).repeat(self.mppi.K, 1), u_rand)

            learner_state_visitations = get_state_visitations(trajs, map_metadata, weights)
            expert_state_visitations = get_state_visitations(expert_traj.unsqueeze(0), map_metadata)
            random_state_visitations = get_state_visitations(traj_rand, map_metadata)

            lfc.append(learner_state_visitations)
            efc.append(expert_state_visitations)
            rfc.append(random_state_visitations)

            deep_features_cache.append(costmap)
            grads.append((expert_state_visitations - learner_state_visitations))

        lfc = torch.stack(lfc, dim=0)
        efc = torch.stack(efc, dim=0)
        rfc = torch.stack(rfc, dim=0)

        deep_features_cache = torch.stack(deep_features_cache, dim=0)
        grads = torch.stack(grads, dim=0) / len(grads)

        self.network_opt.zero_grad()
        deep_features_cache.backward(gradient=grads)
        self.network_opt.step()

    def visualize(self):
        dl = DataLoader(self.expert_dataset, batch_size=1, shuffle=True)
        with torch.no_grad():
            for i, data in enumerate(dl):
                if i > 20:
                    return

                map_features = data['map_features'][0]
                map_metadata = {k:v[0] for k,v in data['metadata'].items()}
                metadata = data['metadata']
                xmin = metadata['origin'][0, 0]
                ymin = metadata['origin'][0, 1]
                xmax = xmin + metadata['width'][0]
                ymax = ymin + metadata['height'][0]
                expert_traj = data['traj'][0]

                #compute costmap

#                #mlp
#                costmap = torch.moveaxis(self.network.forward(torch.moveaxis(map_features, 0, -1)), -1, 0)[0]

                #resnet cnn
                costmap = self.network.forward(map_features.view(1, *map_features.shape))[0, 0]

                #initialize solver
                initial_state = expert_traj[0]
                HACK = {"state":initial_state, "steer_angle":torch.zeros(1)}
                x = self.mppi.model.get_observations(HACK)

                map_params = {
                    'resolution': map_metadata['resolution'].item(),
                    'height': map_metadata['height'].item(),
                    'width': map_metadata['width'].item(),
                    'origin': map_metadata['origin']
                }
                self.mppi.reset()
                self.mppi.cost_fn.update_map_params(map_params)
                self.mppi.cost_fn.update_costmap(costmap)
                self.mppi.cost_fn.update_goal(expert_traj[-1, :2])

                #solve for traj
                for ii in range(self.mppi_itrs):
                    self.mppi.get_control(x, step=False)

                #regular version
                traj = self.mppi.last_states

                metadata = data['metadata']
                xmin = metadata['origin'][0, 0]
                ymin = metadata['origin'][0, 1]
                xmax = xmin + metadata['width'][0]
                ymax = ymin + metadata['height'][0]
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                

                idx = self.expert_dataset.feature_keys.index('height_high')
                axs[0].imshow(map_features[idx], origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))
                m1 = axs[1].imshow(costmap, origin='lower', cmap='coolwarm', extent=(xmin, xmax, ymin, ymax))
                axs[0].plot(expert_traj[:, 0], expert_traj[:, 1], c='y', label='expert')
                axs[1].plot(expert_traj[:, 0], expert_traj[:, 1], c='y')
                axs[0].plot(traj[:, 0], traj[:, 1], c='g', label='learner')
                axs[1].plot(traj[:, 0], traj[:, 1], c='g')

                axs[0].set_title('heightmap high')
                axs[1].set_title('irl cost')
                axs[0].legend()

                plt.colorbar(m1, ax=axs[1])
                plt.show()

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)

    horizon = 70
    batch_size = 100

    bag_fp = '/home/yamaha/Desktop/datasets/yamaha_maxent_irl/rosbags/'
    pp_fp = '/home/yamaha/Desktop/datasets/yamaha_maxent_irl/torch/'

    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=pp_fp)

    kbm = SteerSetpointKBM(L=3.0, v_target_lim=[3.0, 8.0], steer_lim=[-0.3, 0.3], steer_rate_lim=0.2)

    parameters = {
        'log_K_delta':torch.tensor(10.0)
    }
    kbm.update_parameters(parameters)
    cfn = WaypointCostMapCostFunction(unknown_cost=0., goal_cost=0., map_params=dataset.metadata)
    mppi = MPPI(model=kbm, cost_fn=cfn, num_samples=2048, num_timesteps=horizon, control_params={'sys_noise':torch.tensor([2.0, 0.5]), 'temperature':0.05})

    mppi_irl = MPPIIRL(dataset, mppi)

    for i in range(100):
        mppi_irl.update()
        with torch.no_grad():
            torch.save({'net':mppi_irl.network, 'keys':mppi_irl.expert_dataset.feature_keys}, 'learner_net_2layer/weights_itr_{}.pt'.format(i + 1))

#        if (i % 10) == 0:
            #visualize
    mppi_irl.visualize()
