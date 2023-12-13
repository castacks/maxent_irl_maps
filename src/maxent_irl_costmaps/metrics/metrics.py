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

            #also get KBM states for expert
            X_expert = {
                'state': expert_traj,
                'steer_angle': data['steer'].unsqueeze(-1) if 'steer' in data.keys() else torch.zeros(experiment.mppi.B, experiment.mppi.T, 1, device=initial_states.device)
            }
            expert_kbm_traj = experiment.mppi.model.get_observations(X_expert)

            experiment.mppi.reset()
            experiment.mppi.cost_fn.data['goals'] = goals
            experiment.mppi.cost_fn.data['costmap'] = costmap
            experiment.mppi.cost_fn.data['costmap_metadata'] = map_params

            #TEMP HACK - Sam TODO: clean up once data is on cluster
            experiment.mppi.cost_fn.data['costmap_metadata']['length_x'] = experiment.mppi.cost_fn.data['costmap_metadata']['width']
            experiment.mppi.cost_fn.data['costmap_metadata']['length_y'] = experiment.mppi.cost_fn.data['costmap_metadata']['height']

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

            xmin = metadata['origin'][0].cpu()
            ymin = metadata['origin'][1].cpu()
            xmax = xmin + metadata['width'].cpu()
            ymax = ymin + metadata['height'].cpu()

            fig, axs = plt.subplots(2, 2, figsize=(18, 18))
            axs = axs.flatten()

            axs[0].imshow(data['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            axs[0].set_title('FPV')

            m1 = axs[1].imshow(costmap[0].cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax))
            axs[1].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y', label='expert')
            axs[1].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g', label='learner')
            axs[1].set_title('costmap')
            axs[1].legend()

            axs[2].imshow(learner_state_visitations.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
            axs[2].set_title('learner SV')

            axs[3].imshow(expert_state_visitations.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
            axs[3].set_title('expert SV')

            plt.colorbar(m1, ax=axs[1])

            title = ''
            for k,v in metrics_res.items():
                title += '{}:{:.4f}    '.format(k, v[-1])
            plt.suptitle(title)

            if viz:
#                plt.show(block=False)
#                plt.pause(1e-2)

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
