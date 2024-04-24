"""
Collection of metrics for evaluating performance of mexent IRL
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.global_state_visitation_buffer import GlobalStateVisitationBuffer
from maxent_irl_costmaps.networks.baseline_lethal_height import LethalHeightCostmap
from maxent_irl_costmaps.utils import get_state_visitations, quat_to_yaw
from maxent_irl_costmaps.geometry_utils import apply_footprint

def get_metrics_planner(experiment, metric_fns = {}, frame_skip=1, viz=True, save_fp=""):
    """
    Temp hack for getting metrics for state_lattice_planner
    Args:
        experiment: the experiment to compute metrics for
        metric_fns: A dict of {label:function} (the ones defined in this file) to use to compute metrics
    """
#    plt.show(block=False)
#    baseline = LethalHeightCostmap(experiment.expert_dataset).to(experiment.device)

    metrics_res = {k:[] for k in metric_fns.keys()}

    with torch.no_grad():
        for i in range(0, len(experiment.expert_dataset), frame_skip):
            print('{}/{}'.format(i+1, len(experiment.expert_dataset)), end='\r')

            data = experiment.expert_dataset[i]

            #hack back to single dim
            map_features = data['map_features'].unsqueeze(0)
            metadata = data['metadata']
            xmin = metadata['origin'][0].cpu()
            ymin = metadata['origin'][1].cpu()
            xmax = xmin + metadata['width'].cpu()
            ymax = ymin + metadata['height'].cpu()
            expert_traj = data['traj']

            #compute costmap
            #for metrics (namely mhd), clip etraj to map bounds
            emask = (expert_traj[:, 0] > xmin) & (expert_traj[:, 0] < xmax) & (expert_traj[:, 1] > ymin) & (expert_traj[:, 1] < ymax)
            expert_traj_clip = expert_traj[emask]

            #ensemble
            if hasattr(experiment.network, 'ensemble_forward'):
                #save GPU space in batch (they're copied anyways)
                res = experiment.network.ensemble_forward(map_features[[0]])
                costmap = res['costmap'].mean(dim=1)[0]

            #no ensemble
            else:
                res = experiment.network.forward(map_features)
                costmap = res['costmap'][:, 0]

            #initialize solver
            #initialize goals for cost function
            expert_kbm_traj = torch.stack([
                expert_traj[:, 0],
                expert_traj[:, 1],
                experiment.quat_to_yaw(expert_traj[:, 3:7]) % (2*np.pi)
            ], dim=-1)

            initial_pos = expert_kbm_traj[0]
            goal_pos = experiment.clip_to_map_bounds(expert_kbm_traj, metadata)

            # setup start state
            angles = torch.linspace(0., 2*np.pi, experiment.planner.primitives['angnum']+1, device=experiment.planner.device)

            sgx = ((initial_pos[0] - metadata['origin'][0]) / experiment.planner.primitives['lindisc']).round().clip(0, experiment.planner.states.shape[0]-1)
            sgy = ((initial_pos[1] - metadata['origin'][1]) / experiment.planner.primitives['lindisc']).round().clip(0, experiment.planner.states.shape[1]-1)
            sga = (initial_pos[2] - angles).abs().argmin(dim=-1) % experiment.planner.primitives['angnum']
            start_state = torch.stack([sgx, sgy, sga], dim=-1).long()

            # setup goal state
            ggx = ((goal_pos[0] - metadata['origin'][0]) / experiment.planner.primitives['lindisc']).round().clip(0, experiment.planner.states.shape[0]-1)
            ggy = ((goal_pos[1] - metadata['origin'][1]) / experiment.planner.primitives['lindisc']).round().clip(0, experiment.planner.states.shape[1]-1)
            gga = (goal_pos[2] - angles).abs().argmin(dim=-1) % experiment.planner.primitives['angnum']

            goal_state = torch.stack([ggx, ggy, gga], dim=-1).long()

            experiment.planner.load_costmap(costmap[0].T, length_weight=0.1)
            solution = experiment.planner.solve(goal_states = goal_state.unsqueeze(0), max_itrs=1000)
            traj = experiment.planner.extract_solution_parallel(solution, start_state.unsqueeze(0))[0].float()
            traj[:, 0] += initial_pos[0]
            traj[:, 1] += initial_pos[1]

            learner_state_visitations = get_state_visitations(traj.unsqueeze(0), metadata)
            expert_state_visitations = get_state_visitations(expert_kbm_traj.unsqueeze(0), metadata)

            for k, fn in metric_fns.items():
                metrics_res[k].append(fn(costmap, expert_traj_clip, traj, expert_state_visitations, learner_state_visitations).cpu().item())

            fig, axs = experiment.visualize(i)

            #debug
#            for ax in axs[1:]:
#                ax.plot(expert_traj_clip[:, 0].cpu(), expert_traj_clip[:, 1].cpu(), c='r')
#                ax.plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='m')

            title = ''
            for k,v in metrics_res.items():
                title += '{}:{:.4f}    '.format(k, v[-1])
            plt.suptitle(title)

            if viz:
                plt.savefig(os.path.join(save_fp, 'figs', '{:06d}.png'.format(i)))

            plt.close()

            #idk why I have to do this
            if i == (len(experiment.expert_dataset)-1):
                break

    return {k:torch.tensor(v) for k,v in metrics_res.items()}

def get_metrics(experiment, metric_fns = {}, frame_skip=1, viz=True, save_fp=""):
    """
    Wrapper method that generates metrics for an experiment
    Args:
        experiment: the experiment to compute metrics for
        metric_fns: A dict of {label:function} (the ones defined in this file) to use to compute metrics
    """
#    plt.show(block=False)
#    baseline = LethalHeightCostmap(experiment.expert_dataset).to(experiment.device)

    metrics_res = {k:[] for k in metric_fns.keys()}

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

            #for metrics (namely mhd), clip etraj to map bounds
            emask = (expert_traj[:, 0] > xmin) & (expert_traj[:, 0] < xmax) & (expert_traj[:, 1] > ymin) & (expert_traj[:, 1] < ymax)
            expert_traj = expert_traj[emask]

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

            #TEMP HACK - Sam TODO: clean up once data is on cluster
            map_params['length_x'] = map_params['width']
            map_params['length_y'] = map_params['height']

            goals = [expert_traj[[-1], :2]] * experiment.mppi.B

            #also get KBM states for expert
            X_expert = {
                'state': expert_traj,
                'steer_angle': data['steer'][:expert_traj.shape[0]].unsqueeze(-1) if 'steer' in data.keys() else torch.zeros(experiment.mppi.B, expert_traj.shape[0], 1, device=initial_states.device)
            }
            expert_kbm_traj = experiment.mppi.model.get_observations(X_expert)

            goals = [experiment.clip_to_map_bounds(expert_traj[:, :2], metadata).view(1, 2)] * experiment.mppi.B

            experiment.mppi.reset()
            experiment.mppi.cost_fn.data['goals'] = goals
            experiment.mppi.cost_fn.data['costmap'] = {
                'data': costmap,
                'metadata': map_params
            }

            #solve for traj
            for ii in range(experiment.mppi_itrs):
                experiment.mppi.get_control(x, step=False)

            tidx = experiment.mppi.last_cost.argmin()
            traj = experiment.mppi.last_states[tidx].clone()

            trajs = experiment.mppi.noisy_states[tidx].clone()
            weights = experiment.mppi.last_weights[tidx].clone()

            footprint_learner_traj = apply_footprint(trajs, experiment.footprint).view(experiment.mppi.K, -1, 2)
            footprint_expert_traj = apply_footprint(expert_kbm_traj.unsqueeze(0), experiment.footprint).view(1, -1, 2)

            learner_state_visitations = get_state_visitations(footprint_learner_traj, metadata, weights)
            expert_state_visitations = get_state_visitations(footprint_expert_traj, metadata)

#            learner_state_visitations = get_state_visitations(trajs, metadata, weights)
#            expert_state_visitations = get_state_visitations(expert_traj.unsqueeze(0), metadata)

            for k, fn in metric_fns.items():
                metrics_res[k].append(fn(costmap, expert_traj, traj, expert_state_visitations, learner_state_visitations).cpu().item())

            fig, axs = experiment.visualize(i)

            #debug
#            for ax in axs[1:]:
#                ax.plot(expert_traj_clip[:, 0].cpu(), expert_traj_clip[:, 1].cpu(), c='r')
#                ax.plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='m')

            title = ''
            for k,v in metrics_res.items():
                title += '{}:{:.4f}    '.format(k, v[-1])
            plt.suptitle(title)

            if viz:
                plt.savefig(os.path.join(save_fp, 'figs', '{:06d}.png'.format(i)))

            plt.close()

            #idk why I have to do this
            if i == (len(experiment.expert_dataset)-1):
                break

    return {k:torch.tensor(v) for k,v in metrics_res.items()}

def expert_cost(
                costmap,
                expert_traj,
                learner_traj,
                expert_state_visitations,
                learner_state_visitations
                ):
    return (costmap * expert_state_visitations).sum()

def learner_cost(
                costmap,
                expert_traj,
                learner_traj,
                expert_state_visitations,
                learner_state_visitations
                ):
    return (costmap * learner_state_visitations).sum()

def position_distance(
                costmap,
                expert_traj,
                learner_traj,
                expert_state_visitations,
                learner_state_visitations
                ):
    return torch.linalg.norm(expert_traj[:, :2] - learner_traj[:, :2], dim=-1).sum()

def kl_divergence(
                costmap,
                expert_traj,
                learner_traj,
                expert_state_visitations,
                learner_state_visitations
                ):
    #We want learner onto expert
    #KL(p||q) = sum_p[p(x) * log(p(x)/q(x))]
    return (learner_state_visitations * torch.log((learner_state_visitations + 1e-6) / (expert_state_visitations + 1e-6))).sum()

def modified_hausdorff_distance(
                costmap,
                expert_traj,
                learner_traj,
                expert_state_visitations,
                learner_state_visitations
                ):
    ap = expert_traj[:, :2]
    bp = learner_traj[:, :2]
    dist_mat = torch.linalg.norm(ap.unsqueeze(0) - bp.unsqueeze(1), dim=-1)
    mhd1 = dist_mat.min(dim=0)[0].mean()
    mhd2 = dist_mat.min(dim=1)[0].mean()
    return max(mhd1, mhd2)
