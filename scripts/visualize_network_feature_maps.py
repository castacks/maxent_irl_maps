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
from maxent_irl_costmaps.os_utils import maybe_mkdir

def visualize_feature_maps(model, idx):
    """
    Look at network feature maps to see what the activations are doing
    """
    with torch.no_grad():
        if idx == -1:
            idx = np.random.randint(len(model.expert_dataset))

        data = model.expert_dataset[idx]

        map_features = data['map_features']
        metadata = data['metadata']
        xmin = metadata['origin'][0].cpu()
        ymin = metadata['origin'][1].cpu()
        xmax = xmin + metadata['width']
        ymax = ymin + metadata['height']
        expert_traj = data['traj']

        #compute costmap
        #resnet cnn
        n = 6

        res = model.network.forward(map_features.view(1, *map_features.shape))
        costmap = res['costmap'][0, 0]
        deep_features = res['features'][0]
        feature_weights = model.network.cost_head.weight[0].squeeze()
        deep_feat_idxs = torch.argsort(feature_weights.abs(), descending=True)

        idx = model.expert_dataset.feature_keys.index('height_high')
        fig1, axs1 = plt.subplots(1, 4, figsize=(24, 6))
        axs1[0].imshow(data['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
        axs1[1].imshow(map_features[idx].cpu(), origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))
        axs1[2].imshow(costmap.cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax), vmin=0., vmax=30.)

        fw2 = feature_weights.abs()[deep_feat_idxs]
        fw2 = torch.cumsum(fw2, dim=0) / fw2.sum()
        axs1[3].plot(fw2, marker='.')
        axs1[3].axvline(n*(n+1), c='r')

        deep_features = deep_features[deep_feat_idxs]
        feature_weights = feature_weights[deep_feat_idxs]
        fig2, axs2 = plt.subplots(n, n+1, figsize=(20, 20))
        axs2 = axs2.flatten()
        for i in range(len(axs2)):
            df = deep_features[i]
            fw = feature_weights[i]
            axs2[i].imshow(df.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
            axs2[i].set_title('{:.4f}'.format(fw))
            axs2[i].get_xaxis().set_visible(False)
            axs2[i].get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_fp', type=str, required=True, help='path to save figs to')
    parser.add_argument('--model_fp', type=str, required=True, help='Costmap weights file')
    parser.add_argument('--bag_fp', type=str, required=True, help='dir for rosbags to train from')
    parser.add_argument('--preprocess_fp', type=str, required=True, help='dir to save preprocessed data to')
    parser.add_argument('--map_topic', type=str, required=False, default='/local_gridmap', help='topic to extract map features from')
    parser.add_argument('--odom_topic', type=str, required=False, default='/integrated_to_init', help='topic to extract odom from')
    parser.add_argument('--image_topic', type=str, required=False, default='/multisense/left/image_rect_color', help='topic to extract images from')
    parser.add_argument('--viz', action='store_true', help='set this flag if you want the pyplot viz. Default is to save to folder')
    args = parser.parse_args()

    model = torch.load(args.model_fp, map_location='cpu')

    dataset = MaxEntIRLDataset(bag_fp=args.bag_fp, preprocess_fp=args.preprocess_fp, map_features_topic=args.map_topic, odom_topic=args.odom_topic, image_topic=args.image_topic, horizon=model.expert_dataset.horizon)

    model.expert_dataset = dataset
    model.network.eval()

    maybe_mkdir(args.save_fp, force=False)

    for i in range(len(dataset)):
        print('{}/{}'.format(i+1, len(dataset)), end='\r')
        fig_fp = os.path.join(args.save_fp, '{:05d}.png'.format(i+1))
        if args.viz:
            visualize_feature_maps(model, idx=-1)
            plt.show()
        else:
            model.visualize(idx = i)
            plt.savefig(fig_fp)
            plt.close()
