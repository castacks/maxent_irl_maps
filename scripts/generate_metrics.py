import rosbag
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import scipy.spatial
import scipy.interpolate

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
from torch_mpc.algos.batch_mppi import BatchMPPI
from torch_mpc.cost_functions.waypoint_costmap import WaypointCostMapCostFunction

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.dataset.global_state_visitation_buffer import GlobalStateVisitationBuffer
from maxent_irl_costmaps.os_utils import maybe_mkdir
from maxent_irl_costmaps.metrics.metrics import *

from maxent_irl_costmaps.networks.baseline_lethal_height import LethalHeightCostmap

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_fp', type=str, required=True, help='path to save figs to')
    parser.add_argument('--model_fp', type=str, required=True, help='Costmap weights file')
    parser.add_argument('--bag_fp', type=str, required=True, help='dir for rosbags to train from')
    parser.add_argument('--preprocess_fp', type=str, required=True, help='dir to save preprocessed data to')
    parser.add_argument('--gsv_buffer_fp', type=str, required=True, help='path to the global state visitation buffer')
    parser.add_argument('--map_topic', type=str, required=False, default='/local_gridmap', help='topic to extract map features from')
    parser.add_argument('--odom_topic', type=str, required=False, default='/integrated_to_init', help='topic to extract odom from')
    parser.add_argument('--image_topic', type=str, required=False, default='/multisense/left/image_rect_color', help='topic to extract images from')
    parser.add_argument('--baseline', action='store_true', required=False, help='set this flag to run baseline map')
    args = parser.parse_args()

    model = torch.load(args.model_fp, map_location='cpu')
    model.network.eval()

    dataset = MaxEntIRLDataset(bag_fp=args.bag_fp, preprocess_fp=args.preprocess_fp, map_features_topic=args.map_topic, odom_topic=args.odom_topic, image_topic=args.image_topic, horizon=model.expert_dataset.horizon)

    model.expert_dataset = dataset

    #HACK
    model.mppi.K1 = 100
    model.mppi.K2 = 2048
    model.mppi.K = model.mppi.K1 + model.mppi.K2

    if args.baseline:
        model.network = LethalHeightCostmap(dataset)

    gsv = torch.load(args.gsv_buffer_fp, map_location='cpu')

    maybe_mkdir(args.save_fp, force=False)
    metrics = {
        'expert_cost':expert_cost,
        'learner_cost':learner_cost,
        'traj':position_distance,
        'kl':kl_divergence,
        'kl_global':kl_divergence_global,
        'mhd': modified_hausdorff_distance
    }

    res = get_metrics(model, gsv, metrics)
    torch.save(res, os.path.join(args.save_fp, 'metrics.pt'))
