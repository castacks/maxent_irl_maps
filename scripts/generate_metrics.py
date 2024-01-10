import rosbag
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import scipy.spatial
import scipy.interpolate

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.os_utils import maybe_mkdir
from maxent_irl_costmaps.metrics.metrics import *

from maxent_irl_costmaps.dataset.preprocess_pointpillars_dataset import PreprocessPointpillarsDataset

from maxent_irl_costmaps.networks.baseline_lethal_height import LethalHeightCostmap

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_fp', type=str, required=True, help='path to save figs to')
    parser.add_argument('--model_fp', type=str, required=True, help='Costmap weights file')
    parser.add_argument('--preprocess_fp', type=str, required=True, help='dir to save preprocessed data to')
    parser.add_argument('--viz', action='store_true', required=False, help='set this flag to visualize output')
    parser.add_argument('--use_planner', action='store_true', required=False, help='set this if optimizer is planner')
    parser.add_argument('--device', type=str, required=False, default='cpu', help='device to run script on')
    args = parser.parse_args()

    model = torch.load(args.model_fp, map_location='cpu')
    model.network.eval()

    dataset = PreprocessPointpillarsDataset(preprocess_fp=args.preprocess_fp, gridmap_type=model.expert_dataset.gridmap_type, feature_mean=model.expert_dataset.feature_mean, feature_std=model.expert_dataset.feature_std)
    model.expert_dataset = dataset

#    if args.baseline:
#        model.network = LethalHeightCostmap(dataset).to(args.device)

    model = model.to(args.device)

    maybe_mkdir(os.path.join(args.save_fp, 'figs'), force=False)
    metrics = {
        'expert_cost':expert_cost,
        'learner_cost':learner_cost,
#        'traj':position_distance,
        'kl':kl_divergence,
#        'kl_global':kl_divergence_global,
        'mhd': modified_hausdorff_distance
    }

#    for i in range(100):
#        dataset.visualize()
#        plt.show()

    if args.use_planner:
        res = get_metrics_planner(model, metrics, frame_skip=1, viz=args.viz, save_fp=args.save_fp)
    else:
        res = get_metrics(model, metrics, frame_skip=1, viz=args.viz, save_fp=args.save_fp)

    torch.save(res, os.path.join(args.save_fp, 'metrics.pt'))
