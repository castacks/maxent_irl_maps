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
    metrics_res = {k:[] for k in ['speedmap_prob', 'speedmap_l1_err']}

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

                _speeds = experiment.network.speed_bins[1:].to(experiment.device).view(-1, 1, 1)

                speedmap_dist = res['speedmap'][0].mean(dim=0).softmax(dim=0)
                speedmap_val = (_speeds * speedmap_dist).sum(dim=0)

            else:
                res = experiment.network.forward(map_features)
                speedmap_mean = res['speedmap'].loc[:, 0]
                speedmap_std = res['speedmap'].scale[:, 0]

            epos = expert_traj[:, :2].unsqueeze(0)
            espeeds = torch.linalg.norm(expert_traj[:, 7:10], dim=-1).unsqueeze(0)
            esm = get_speedmap(epos, espeeds, metadata)

            mask = esm > 0.1

            # compute correct bins for expert speeds
            _sbins = experiment.network.speed_bins[:-1].to(experiment.device).view(-1, 1, 1)
            sdiffs = esm.unsqueeze(0) - _sbins
            sdiffs[sdiffs < 0] = 1e10
            expert_speed_idxs = sdiffs.argmin(dim=0)

            sdist_flat = speedmap_dist.view(speedmap_dist.shape[0], -1).T
            eidxs_flat = expert_speed_idxs.flatten()
            mask_flat = mask.flatten()
            
            speedmap_prob = sdist_flat[torch.arange(len(eidxs_flat)), eidxs_flat][mask_flat]
            speedmap_err = (speedmap_val - esm).abs()[mask]

            metrics_res['speedmap_prob'].append(speedmap_prob.mean())
            metrics_res['speedmap_l1_err'].append(speedmap_err.mean())

    return {k:torch.tensor(v) for k,v in metrics_res.items()}
