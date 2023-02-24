"""
Run some experiments to more accurately match expert trajectories with MPPI

Key questions:
    1. For a given costmap, how repeatable are my optimal MPPI params?
    2. How consistent are the optimal MPPI params across a dataset?
    3. For a single map and set of params, how consistent is the MPPI solution?

First set of params to mess with:
    1. goal weight
    2. d_ou_noise

(For now I'll probably do a grid search)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.dataset.global_state_visitation_buffer import GlobalStateVisitationBuffer
from maxent_irl_costmaps.networks.baseline_lethal_height import LethalHeightCostmap
from maxent_irl_costmaps.utils import get_state_visitations, quat_to_yaw

def run(experiment, frame_skip=1, viz=True):
    """
    Wrapper method that generates metrics for an experiment
    Args:
        experiment: the experiment to compute metrics for
    """
#    plt.show(block=False)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    with torch.no_grad():
        for i in range(0, len(experiment.expert_dataset), frame_skip):
            print('{}/{}'.format(i+1, len(experiment.expert_dataset)), end='\r')

            data = experiment.expert_dataset[i]

            #hack back to single dim
            map_features = torch.stack([data['map_features']] * experiment.mppi.B, dim=0)
            metadata = data['metadata']
            xmin = metadata['origin'][0].cpu()
            ymin = metadata['origin'][1].cpu()
            xmax = xmin + metadata['width'].cpu()
            ymax = ymin + metadata['height'].cpu()
            expert_traj = data['traj']

            #compute costmap

            #ensemble
            if hasattr(experiment.network, 'ensemble_forward'):
                #save GPU space in batch (they're copied anyways)
                res = experiment.network.ensemble_forward(map_features[[0]])
                costmap = res['costmap'].mean(dim=1)[0]
                costmap = torch.cat([costmap] * experiment.mppi.B, dim=0)

            #no ensemble
            else:
                res = experiment.network.forward(map_features)
                costmap = res['costmap'][:, 0]

            #initialize solver
            initial_state = expert_traj[0]
            x0 = {"state":initial_state, "steer_angle":data["steer"][[0]] if "steer" in data.keys() else torch.zeros(1, device=initial_state.device)}
            x = torch.stack([experiment.mppi.model.get_observations(x0)] * experiment.mppi.B, dim=0)

            map_params = {
                'resolution': torch.tensor([metadata['resolution']] * experiment.mppi.B, device=experiment.mppi.device),
                'height': torch.tensor([metadata['height']] * experiment.mppi.B, device=experiment.mppi.device),
                'width': torch.tensor([metadata['width']] * experiment.mppi.B, device=experiment.mppi.device),
                'origin': torch.stack([metadata['origin']] * experiment.mppi.B, dim=0)
            }

            goals = [expert_traj[[-1], :2]] * experiment.mppi.B

            experiment.mppi.reset()
            experiment.mppi.cost_fn.data['goals'] = goals
            experiment.mppi.cost_fn.data['costmap'] = costmap
            experiment.mppi.cost_fn.data['costmap_metadata'] = map_params

            d_ou_values = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            goal_weight_values = [10.0, 100.0]

            costs = torch.zeros(len(d_ou_values), len(goal_weight_values))
            trajs = torch.zeros(len(d_ou_values), len(goal_weight_values), experiment.mppi.T, experiment.mppi.n)

            for di, d_ou_value in enumerate(d_ou_values):
                for gi, gw_value in enumerate(goal_weight_values):
                    print('d_ou = {}, gw = {}'.format(d_ou_value, gw_value), end='\r')
                    #set params
                    experiment.mppi.d_ou_scale = d_ou_value
                    experiment.mppi.cost_fn.cost_weights[1] = gw_value
                    experiment.mppi.reset()

                    #solve for traj
                    for ii in range(experiment.mppi_itrs):
                        experiment.mppi.get_control(x, step=False)

                    tidx = experiment.mppi.last_cost.argmin()
                    traj = experiment.mppi.last_states[tidx].clone()

                    all_trajs = experiment.mppi.last_states.clone()

                    traj_cost = (expert_traj[:, :2].unsqueeze(0) - all_trajs[..., :2]).pow(2).sum(dim=-1).sum(dim=-1).mean()

                    costs[di, gi] = traj_cost
                    trajs[di, gi] = traj

            print(costs.T)

            xmin = metadata['origin'][0].cpu()
            ymin = metadata['origin'][1].cpu()
            xmax = xmin + metadata['width'].cpu()
            ymax = ymin + metadata['height'].cpu()

            for ax in axs:
                ax.cla()

            axs[0].imshow(data['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            axs[0].set_title('FPV')

            axs[1].imshow(costmap[0].cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax), vmin=-10., vmax=10.)
            axs[1].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y', label='expert')

            for di, d_ou_value in enumerate(d_ou_values):
                for gi, gw_value in enumerate(goal_weight_values):
                    traj = trajs[di, gi]
#                    axs[1].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), label="dou={}, gw={}".format(d_ou_value, gw_value))
                    axs[1].plot(traj[:, 0].cpu(), traj[:, 1].cpu())
                    axs[1].set_title('costmap')
                    axs[1].legend()

            min_di, min_gi = torch.argwhere(costs == costs.min()).squeeze()

            axs[2].imshow(costs.T)
            axs[2].set_xlabel('d_ou')
            axs[2].set_ylabel('gw')

            axs[4].imshow(costmap[0].cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax), vmin=-10., vmax=10.)
            axs[4].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y', label='expert')
            axs[4].plot(trajs[min_di, min_gi, :, 0].cpu(), trajs[min_di, min_gi, :, 1].cpu())

            fig.suptitle('best d_ou = {}, best_gw = {}'.format(d_ou_values[min_di], goal_weight_values[min_gi]))

            if viz:
                plt.show(block=False)
                plt.pause(1e-2)

            #idk why I have to do this
            if i == (len(experiment.expert_dataset)-1):
                break

    plt.close()
    return {k:torch.tensor(v) for k,v in metrics_res.items()}

if __name__ == '__main__':
    experiment_fp = '/home/atv/Desktop/experiments/yamaha_maxent_irl/2022-11-14-17-53-51_1layer_resnet_adamw_001_nopos/itr_50.pt'
    experiment = torch.load(experiment_fp, map_location='cpu').to('cuda')
    experiment.network.eval()

    bag_fp = ''
    preprocess_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75'

    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=preprocess_fp, horizon=experiment.expert_dataset.horizon, feature_keys=experiment.expert_dataset.feature_keys).to('cuda')
    experiment.expert_dataset = dataset

    res = run(experiment, frame_skip=10)
