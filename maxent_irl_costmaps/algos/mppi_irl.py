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
from maxent_irl_costmaps.utils import get_feature_counts

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
    def __init__(self, expert_dataset, mppi, costmapper, opt):
        """
        Args:
            expert_dataset: The dataset containing expert demonstrations to imitate
            mppi: The MPPI object to optimize with
        """
        self.expert_dataset = expert_dataset
        self.mppi = mppi
        self.mppi_itrs = 3
        self.costmapper = costmapper
        self.opt = opt

        self.itr = 0

    def update(self):
        self.itr += 1
        dl = DataLoader(self.expert_dataset, batch_size=1, shuffle=True)
        grads = []
        efc = []
        lfc = []

        for i, data in enumerate(dl):
            if i >= 64:
                return 
            print(i, end='\r')
            map_features = data['map_features'][0]
            map_metadata = {k:v[0] for k,v in data['metadata'].items()}
            expert_traj = data['traj'][0]

            #compute costmap
#            costmap = (map_features * self.weights.view(-1, 1, 1)).sum(dim=0)
            costmap = self.costmapper.get_costmap(data)[0]

            #initialize solver
            initial_state = expert_traj[0]
            HACK = {"state":initial_state, "steer_angle":torch.zeros(1)}
            x = self.mppi.model.get_observations(HACK)
            self.mppi.cost_fn.update_costmap(costmap)

            #solve for traj
            for ii in range(self.mppi_itrs):
                self.mppi.get_control(x, step=False)

            #regular version
            traj = self.mppi.last_states

            #weighting version
            trajs = self.mppi.noisy_states.clone()
            weights = self.mppi.last_weights.clone()

            self.mppi.reset()

            #get learner feature counts

            #regular
#            learner_feature_counts = self.expert_dataset.get_feature_counts(traj, map_features, map_metadata)

            #MPPI weight
            learner_feature_counts = get_feature_counts(trajs, map_features, map_metadata)
            learner_feature_counts = (weights.view(1, -1) * learner_feature_counts).sum(dim=-1)

            expert_feature_counts = get_feature_counts(expert_traj, map_features, map_metadata)

            lfc.append(learner_feature_counts)
            efc.append(expert_feature_counts)
            grads.append(expert_feature_counts - learner_feature_counts)

            """
            #Viz debug
            if (i%10) == 0:
                xmin = map_metadata['origin'][0]
                xmax = xmin + map_metadata['width']
                ymin = map_metadata['origin'][1]
                ymax = ymin + map_metadata['height']
                plt.imshow(costmap.detach(), origin='lower', extent=(xmin, xmax, ymin, ymax))

                plt.plot(traj[:, 0], traj[:, 1], c='r', label='learner')

                plt.plot(expert_traj[:, 0], expert_traj[:, 1], c='b', label='expert')
                plt.title('Itr {}'.format(self.itr))
                plt.legend()

                plt.show()
            """ 

        grads = torch.stack(grads, dim=0)
        lfc = torch.stack(lfc, dim=0)
        efc = torch.stack(efc, dim=0)

        grad = grads.mean(dim=0)

        self.opt.zero_grad()
        self.costmapper.weights -= 0.1 * grad
        self.opt.step()

        print('__________ITR {}__________'.format(self.itr))
        print('WEIGHTS:\n', np.stack([np.array(self.expert_dataset.feature_keys), self.costmapper.weights.detach().numpy()], axis=-1))
        print('LEARNER FC: ', lfc.mean(dim=0))
        print('EXPERT FC:  ', efc.mean(dim=0))

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    horizon = 70
    batch_size = 100

    bag_fp = '/home/yamaha/Desktop/datasets/yamaha_maxent_irl/rosbags/'
    pp_fp = '/home/yamaha/Desktop/datasets/yamaha_maxent_irl/torch_small/'

    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=pp_fp)

    kbm = SteerSetpointKBM(L=3.0, v_target_lim=[3.0, 8.0], steer_lim=[-0.3, 0.3], steer_rate_lim=0.2)

    parameters = {
        'log_K_delta':torch.tensor(10.0)
    }
    kbm.update_parameters(parameters)
    cfn = WaypointCostMapCostFunction(unknown_cost=10., goal_cost=1000., map_params=dataset.metadata)
    mppi = MPPI(model=kbm, cost_fn=cfn, num_samples=2048, num_timesteps=horizon, control_params={'sys_noise':torch.tensor([2.0, 0.5]), 'temperature':0.05})

    weights = torch.zeros(len(dataset.feature_keys), requires_grad=False)
    opt = torch.optim.SGD([weights], lr=0.01)
    costmapper = LinearCostMapper(weights)

    mppi_irl = MPPIIRL(dataset, mppi, costmapper, opt)

    for i in range(100):
        mppi_irl.update()
        torch.save({'weights':costmapper.weights.detach(), 'keys':mppi_irl.expert_dataset.feature_keys}, 'learner_linear/weights_itr_{}.pt'.format(i + 1))
