"""
Collection of metrics for evaluating performance of mexent IRL
"""

import torch
import matplotlib.pyplot as plt

from maxent_irl_costmaps.utils import get_state_visitations

def get_metrics(experiment, metric_fns = {}):
    """
    Wrapper method that generates metrics for an experiment
    Args:
        experiment: the experiment to compute metrics for
        metric_fns: A dict of {label:function} (the ones defined in this file) to use to compute metrics
    """
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    plt.show(block=False)

    res = {k:[] for k in metric_fns.keys()}

    with torch.no_grad():
        for i, batch in enumerate(experiment.expert_dataset):
            print('{}/{}'.format(i+1, len(experiment.expert_dataset)), end='\r')

            map_features = batch['map_features']
            map_metadata = batch['metadata']
            expert_traj = batch['traj']

            #resnet cnn (and actual net interface in general)
            costmap = experiment.network.forward(map_features.view(1, *map_features.shape))[0, 0]

            #initialize solver
            initial_state = expert_traj[0]
            x0 = {"state":initial_state, "steer_angle":batch["steer"][[0]] if "steer" in batch.keys() else torch.zeros(1, device=initial_state.device)}
            x = experiment.mppi.model.get_observations(x0)

            map_params = {
                'resolution': map_metadata['resolution'],
                'height': map_metadata['height'],
                'width': map_metadata['width'],
                'origin': map_metadata['origin']
            }
            experiment.mppi.reset()
            experiment.mppi.cost_fn.update_map_params(map_params)
            experiment.mppi.cost_fn.update_costmap(costmap)
            experiment.mppi.cost_fn.update_goal(expert_traj[-1, :2])

            #solve for traj
            for ii in range(experiment.mppi_itrs):
                experiment.mppi.get_control(x, step=False)

            #regular version
            traj = experiment.mppi.last_states

            #weighting version
            trajs = experiment.mppi.noisy_states.clone()
            weights = experiment.mppi.last_weights.clone()

            experiment.mppi.reset()

            learner_state_visitations = get_state_visitations(trajs, map_metadata, weights)
            expert_state_visitations = get_state_visitations(expert_traj.unsqueeze(0), map_metadata)

            for k, fn in metric_fns.items():
                res[k].append(fn(costmap, expert_traj, traj, expert_state_visitations, learner_state_visitations))

            xmin = map_metadata['origin'][0].cpu()
            ymin = map_metadata['origin'][1].cpu()
            xmax = xmin + map_metadata['width']
            ymax = ymin + map_metadata['height']

            for ax in axs:
                ax.cla()

            axs[0].imshow(batch['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            axs[0].set_title('FPV')

            axs[1].imshow(costmap.cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax))
            axs[1].plot(expert_traj[:, 0], expert_traj[:, 1], c='y', label='expert')
            axs[1].plot(traj[:, 0], traj[:, 1], c='g', label='learner')
            axs[1].set_title('costmap')
            axs[1].legend()

            axs[2].imshow(expert_state_visitations, origin='lower', extent=(xmin, xmax, ymin, ymax))
            axs[2].set_title('expert SV')

            axs[3].imshow(learner_state_visitations, origin='lower', extent=(xmin, xmax, ymin, ymax))
            axs[3].set_title('learner SV')

            title = ''
            for k,v in res.items():
                title += '{}:{:.4f}    '.format(k, v[-1])
            plt.suptitle(title)
            plt.pause(1e-2)
            #idk why I have to do this
            if i == (len(experiment.expert_dataset)-1):
                break

    return {k:torch.tensor(v) for k,v in res.items()}

def expert_cost(costmap, expert_traj, learner_traj, expert_state_visitations, learner_state_visitations):
    return (costmap * expert_state_visitations).sum()

def learner_cost(costmap, expert_traj, learner_traj, expert_state_visitations, learner_state_visitations):
    return (costmap * learner_state_visitations).sum()

def position_distance(costmap, expert_traj, learner_traj, expert_state_visitations, learner_state_visitations):
    return torch.linalg.norm(expert_traj[:, :2] - learner_traj[:, :2], dim=-1).sum()

def kl_divergence(costmap, expert_traj, learner_traj, expert_state_visitations, learner_state_visitations):
    #We want learner onto expert
    #KL(p||q) = sum_p[p(x) * log(p(x)/q(x))]
    return (learner_state_visitations * torch.log((learner_state_visitations + 1e-6) / (expert_state_visitations + 1e-6))).sum()

if __name__ == '__main__':
    experiment_fp = '/home/striest/Desktop/experiments/yamaha_maxent_irl/2022-06-29-11-21-25_trail_driving_cnn_deeper_bnorm_exp/itr_50.pt'
#    experiment_fp = '/home/striest/Desktop/experiments/yamaha_maxent_irl/2022-06-28-11-55-46_trail_driving_cnn_deeper_bnorm/itr_50.pt'
    experiment = torch.load(experiment_fp)

    metrics = {
        'expert_cost':expert_cost,
        'learner_cost':learner_cost,
        'traj':position_distance,
        'kl':kl_divergence,
    }

    res = get_metrics(experiment, metrics)
    torch.save(res, 'res.pt')
