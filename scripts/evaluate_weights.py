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

"""
Compare IRL costmaps to RACER baseline
"""
def load_maps(bag_fp, costmap_topic='/local_cost_map_final_occupancy_grid', skip_window=7.0):
    #just assume that the timestamps are close enough
    costmap_list = []
    costmap_out = []
    timestamps = []

    bag = rosbag.Bag(bag_fp, 'r')
    for topic, msg, t in bag.read_messages(topics=[costmap_topic]):
        costmap_list.append(msg)
        timestamps.append(t.to_sec())

    for i, (costmap, t) in enumerate(zip(costmap_list, timestamps)):
        if t > timestamps[-1] - skip_window:
            continue

        costmap_data = np.asarray(costmap.data).astype(np.float32).reshape(costmap.info.width, costmap.info.height)
        costmap_out.append(costmap_data)

    return costmap_out

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_bag', type=str, required=True, help='Bag of expert data to train from')
    parser.add_argument('--weights', type=str, required=True, help='Costmap weights file')
    args = parser.parse_args()

    dataset = MaxEntIRLDataset(bag_fp=args.test_bag, config_fp='AAA')
    costmaps = load_maps(args.test_bag, skip_window=dataset.dt*dataset.horizon)
    weights = torch.load(args.weights)
    weight_keys = weights['keys']
    weights = weights['weights']

    for k1, k2 in zip(dataset.feature_keys, weight_keys):
        assert k1 == k2, 'dataset and weight keys dont match'

    k = min(len(dataset), len(costmaps))
    for i in range(0, k, 10):
        cmap = costmaps[i]
        feats = dataset.dataset['map_features'][i]
        cmap_pred = (feats * weights.view(-1, 1, 1)).sum(dim=0)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        axs[0].imshow(feats[1])
        m1 = axs[1].imshow(cmap, cmap='coolwarm')
        m2 = axs[2].imshow(cmap_pred, cmap='coolwarm')
        axs[0].set_title('Heightmap High')
        axs[1].set_title('RACER Cost')
        axs[2].set_title('IRL Cost')
        plt.colorbar(m1, ax=axs[1])
        plt.colorbar(m2, ax=axs[2])
        plt.show()
