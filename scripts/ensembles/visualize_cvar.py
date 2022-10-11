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

def visualize_cvar(model, idx):
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
        mosaic = """
        ACDEFG
        BHIJKL
        """

        fig = plt.figure(tight_layout=True, figsize=(18, 6))
        ax_dict = fig.subplot_mosaic(mosaic)
        all_axs = [ax_dict[k] for k in sorted(ax_dict.keys())]
        axs1 = all_axs[:2]
        axs2 = all_axs[2:]

        #plot image, mean costmap, std costmap, and a few samples
        res = model.network.ensemble_forward(map_features.view(1, *map_features.shape))
        costmaps = res['costmap'][0, :, 0]

        costmap_mean = costmaps.mean(dim=0)
        costmap_std = costmaps.std(dim=0)

        #compute cvar
        qs = torch.linspace(-0.9, 0.9, 10)
        costmap_cvars = []
        for q in qs:
            if q < 0.:
                costmap_q = torch.quantile(costmaps, (1.+q).item(), dim=0)
                mask = costmaps <= costmap_q.unsqueeze(0)
            else:
                costmap_q = torch.quantile(costmaps, q.item(), dim=0)
                mask = costmaps >= costmap_q.unsqueeze(0)

            costmap_cvar = (costmaps * mask).sum(dim=0) / mask.sum(dim=0)
            costmap_cvars.append(costmap_cvar)

            #quick test to quantify the ratio of grass cost to obstacle cost (idx=1132)
#            print('CVAR = {:.2f}, LOW = {:.2f}, HIGH = {:.2f}, RATIO = {:.2f}'.format(q, costmap_cvar[125, 115], costmap_cvar[135, 110], costmap_cvar[125, 115] / costmap_cvar[135, 110]))

        idx = model.expert_dataset.feature_keys.index('height_high')
        axs1[0].imshow(data['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
        axs1[1].imshow(map_features[idx].cpu(), origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))
        yaw = model.mppi.model.get_observations({'state':expert_traj, 'steer_angle':torch.zeros(expert_traj.shape[0], 1)})[0, 2]
        axs1[1].arrow(expert_traj[0, 0], expert_traj[0, 1], 8.*yaw.cos(), 8.*yaw.sin(), color='r', head_width=2.)
#        axs1[1].plot(expert_traj[:, 0], expert_traj[:, 1], c='y', label='expert')

        axs1[0].set_title('FPV')
        axs1[1].set_title('Height High')
#        axs1[1].legend()

#        vmin = torch.quantile(cm, 0.1)
#        vmax = torch.quantile(cm, 0.9)

        vmin = torch.quantile(torch.stack(costmap_cvars), 0.1)
        vmax = torch.quantile(torch.stack(costmap_cvars), 0.9)

        for i in range(len(axs2)):
            cm = costmap_cvars[i]
            q = qs[i]
            r = axs2[i].imshow(cm.cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax)
#            axs2[i].plot(expert_traj[:, 0], expert_traj[:, 1], c='y', label='expert')
            axs2[i].get_xaxis().set_visible(False)
            axs2[i].get_yaxis().set_visible(False)
            axs2[i].set_title('Cvar {:.2f}'.format(q))

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

    dataset = MaxEntIRLDataset(bag_fp=args.bag_fp, preprocess_fp=args.preprocess_fp, map_features_topic=args.map_topic, odom_topic=args.odom_topic, image_topic=args.image_topic, horizon=model.expert_dataset.horizon, feature_keys=model.expert_dataset.feature_keys)

    model.expert_dataset = dataset
    model.network.eval()

    maybe_mkdir(args.save_fp, force=False)

    for i in range(len(dataset)):
        print('{}/{}'.format(i+1, len(dataset)), end='\r')
        fig_fp = os.path.join(args.save_fp, '{:05d}.png'.format(i+1))
        if args.viz:
            visualize_cvar(model, idx=-1)
            plt.show()
        else:
            visualize_cvar(model, idx=i)
            plt.savefig(fig_fp)
            plt.close()
