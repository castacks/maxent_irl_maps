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
    parser.add_argument('--weights', type=str, required=True, help='Costmap weights file')
    args = parser.parse_args()

    bag_fp = '/home/yamaha/Desktop/datasets/yamaha_maxent_irl/rosbags/'
    pp_fp = '/home/yamaha/Desktop/datasets/yamaha_maxent_irl/torch/'

    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=pp_fp)
    predictor = torch.load(args.weights)

    if 'net' in predictor.keys():
        network = predictor['net']
        print(network)
        is_deep = True
    else:
        weights = predictor['weights']
        is_deep = False

    for i in range(0, len(dataset), 30):
        data = dataset[i]
        traj = data['traj']
        feats = data['map_features']
        metadata = data['metadata']

        xmin = metadata['origin'][0]
        ymin = metadata['origin'][1]
        xmax = xmin + metadata['width']
        ymax = ymin + metadata['height']

        print('ACTUAL FEATURES STDDEV:', feats.std())
        #deep
        if is_deep:
            with torch.no_grad():
                cmap_pred = torch.moveaxis(network.forward(torch.moveaxis(feats, 0, -1)), -1, 0)[0]
        #linear
        else:
            cmap_pred = (feats * weights.view(-1, 1, 1)).sum(dim=0)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        axs[0].imshow(feats[1], origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))
        m1 = axs[1].imshow(cmap_pred, origin='lower', cmap='coolwarm', extent=(xmin, xmax, ymin, ymax))
        axs[0].plot(traj[:, 0], traj[:, 1], c='y')
        axs[1].plot(traj[:, 0], traj[:, 1], c='y')

        axs[0].set_title('Heightmap High')
        axs[1].set_title('IRL Cost')
        plt.colorbar(m1, ax=axs[1])
        plt.show()
