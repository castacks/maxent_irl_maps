import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.global_state_visitation_buffer import GlobalStateVisitationBuffer
from maxent_irl_costmaps.networks.baseline_lethal_height import LethalHeightCostmap
from maxent_irl_costmaps.utils import get_state_visitations, quat_to_yaw
from maxent_irl_costmaps.geometry_utils import apply_footprint
from maxent_irl_costmaps.utils import get_state_visitations, get_speedmap

def get_speedmap_metrics(experiment, frame_skip=1, viz=True, save_fp=""):
    """
    just do this separately
    """
    metrics_res = {k:[] for k in ['speedmap_mle', 'speedmap_mae']}

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
                speedmap_mean = res['speedmap'].loc.mean(dim=1)[0]
                speedmap_std = res['speedmap'].scale.mean(dim=1)[0]

            #no ensemble
            else:
                res = experiment.network.forward(map_features)
                speedmap_mean = res['speedmap'].loc[:, 0]
                speedmap_std = res['speedmap'].scale[:, 0]

            speedmap = torch.distributions.Normal(loc=speedmap_mean, scale=speedmap_std)
            esm = get_speedmap(expert_traj.unsqueeze(0), metadata)
            mask = esm > 0.1

            log_prob = speedmap.log_prob(esm)
            log_prob = log_prob[mask].mean()

            mae = (esm - speedmap.loc).abs()
            mae = mae[mask].mean()

            metrics_res['speedmap_mle'].append(log_prob.mean())
            metrics_res['speedmap_mae'].append(mae.mean())

    return {k:torch.tensor(v) for k,v in metrics_res.items()}
