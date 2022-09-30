"""
Script for updating a global state visitation buffer w. GUI
"""
import os
import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.dataset.global_costmap import GlobalCostmap
from maxent_irl_costmaps.os_utils import walk_bags
from maxent_irl_costmaps.preprocess import load_traj

#local scripts
from plot_global_costmap import plot_gsv, plot_utm_traj
from google_static_maps_api import GoogleStaticMapsAPI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--costmap_fp', type=str, required=True, help='location to get costmap')
    parser.add_argument('--utm_zone', type=int, required=False, default=17, help='UTM zone that the data is in (probably 17 if CMU data)')
    parser.add_argument('--bag_fp', type=str, required=True, help='dir of bags to add')
    parser.add_argument('--preprocess_fp', type=str, required=True, help='dir of bags to add')
    parser.add_argument('--map_topic', type=str, required=False, default='/local_gridmap', help='topic to extract map features from')
    parser.add_argument('--odom_topic', type=str, required=False, default='/integrated_to_init', help='topic to extract odom from')
    parser.add_argument('--image_topic', type=str, required=False, default='/multisense/left/image_rect_color', help='topic to extract images from')
    parser.add_argument('--device', type=str, required=False, default='cpu', help='device to run costmap inference on')
    args = parser.parse_args()

    buf = torch.load(args.costmap_fp)

    dataset = MaxEntIRLDataset(bag_fp=args.bag_fp, preprocess_fp=args.preprocess_fp, map_features_topic=args.map_topic, odom_topic=args.odom_topic, image_topic=args.image_topic, horizon=buf.model.expert_dataset.horizon).to(args.device)

    buf.model.expert_dataset = dataset
    buf.model.network.eval()
    buf.model.network = buf.model.network.to(args.device)

    buf.add_dataset(dataset)
    torch.save(buf, args.costmap_fp)

    bag_fp = np.random.choice(walk_bags(args.bag_fp))
    traj = load_traj(bag_fp, '/odometry/filtered_odom', 0.05)
    buf.create_anim(traj, 'video.mp4')
