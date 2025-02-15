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

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_maps.networks.baseline_lethal_height import LethalHeightCostmap
from maxent_irl_maps.os_utils import maybe_mkdir


def visualize_cvar(model, idx):
    """
    Look at network feature maps to see what the activations are doing
    """
    baseline = LethalHeightCostmap(
        model.expert_dataset, lethal_height=0.5, blur_sigma=3.0, clip_low=0.5
    ).to(model.device)
    with torch.no_grad():
        if idx == -1:
            idx = np.random.randint(len(model.expert_dataset))

        data = model.expert_dataset[idx]

        map_features = data["map_features"]
        metadata = data["metadata"]
        xmin = metadata["origin"][0].cpu()
        ymin = metadata["origin"][1].cpu()
        xmax = xmin + metadata["width"]
        ymax = ymin + metadata["height"]
        expert_traj = data["traj"]

        res = [plt.subplots(1, 1, figsize=(12, 12)) for _ in range(6)]

        all_figs = [x[0] for x in res]
        all_axs = [x[1] for x in res]
        axs1 = all_axs[:2]
        axs2 = all_axs[2:]

        # plot image, mean costmap, std costmap, and a few samples
        res = model.network.ensemble_forward(map_features.view(1, *map_features.shape))
        costmaps = res["costmap"][0, :, 0]

        costmap_mean = costmaps.mean(dim=0)
        costmap_std = costmaps.std(dim=0)

        # compute cvar
        qs = torch.tensor([-0.9, 0.0, 0.9])
        costmap_cvars = []
        for q in qs:
            if q < 0.0:
                costmap_q = torch.quantile(costmaps, (1.0 + q).item(), dim=0)
                mask = costmaps <= costmap_q.unsqueeze(0)
            else:
                costmap_q = torch.quantile(costmaps, q.item(), dim=0)
                mask = costmaps >= costmap_q.unsqueeze(0)

            costmap_cvar = (costmaps * mask).sum(dim=0) / mask.sum(dim=0)
            costmap_cvars.append(costmap_cvar)

            # quick test to quantify the ratio of grass cost to obstacle cost (idx=1132)
        #            print('CVAR = {:.2f}, LOW = {:.2f}, HIGH = {:.2f}, RATIO = {:.2f}'.format(q, costmap_cvar[125, 115], costmap_cvar[135, 110], costmap_cvar[125, 115] / costmap_cvar[135, 110]))

        idx = model.expert_dataset.feature_keys.index("height_high")
        axs1[0].imshow(data["image"].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
        axs1[1].imshow(
            map_features[idx].cpu(),
            origin="lower",
            cmap="gray",
            extent=(xmin, xmax, ymin, ymax),
        )
        yaw = model.mppi.model.get_observations(
            {"state": expert_traj, "steer_angle": torch.zeros(expert_traj.shape[0], 1)}
        )[0, 2]
        axs1[1].arrow(
            expert_traj[0, 0],
            expert_traj[0, 1],
            8.0 * yaw.cos(),
            8.0 * yaw.sin(),
            color="r",
            head_width=2.0,
        )
        #        axs1[1].plot(expert_traj[:, 0], expert_traj[:, 1], c='y', label='expert')

        #        axs1[1].legend()

        #        vmin = torch.quantile(cm, 0.1)
        #        vmax = torch.quantile(cm, 0.9)

        vmin = torch.quantile(torch.stack(costmap_cvars), 0.1)
        vmax = torch.quantile(torch.stack(costmap_cvars), 0.9)

        for i in range(len(axs2) - 1):
            cm = costmap_cvars[i]
            q = qs[i]
            r = axs2[i].imshow(
                cm.cpu(),
                origin="lower",
                cmap="plasma",
                extent=(xmin, xmax, ymin, ymax),
                vmin=vmin,
                vmax=vmax,
            )
            #            axs2[i].plot(expert_traj[:, 0], expert_traj[:, 1], c='y', label='expert')
            axs2[i].arrow(
                expert_traj[0, 0],
                expert_traj[0, 1],
                8.0 * yaw.cos(),
                8.0 * yaw.sin(),
                color="r",
                head_width=2.0,
            )
            axs2[i].get_xaxis().set_visible(False)
            axs2[i].get_yaxis().set_visible(False)

        baseline_res = baseline.forward(map_features.view(1, *map_features.shape))
        baseline_costmap = baseline_res["costmap"][0, 0]

        axs2[-1].imshow(baseline_costmap.cpu(), origin="lower", cmap="plasma")
        axs2[-1].get_xaxis().set_visible(False)
        axs2[-1].get_yaxis().set_visible(False)

        return all_figs, all_axs


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_fp", type=str, required=True, help="path to save figs to"
    )
    parser.add_argument(
        "--model_fp", type=str, required=True, help="Costmap weights file"
    )
    parser.add_argument(
        "--bag_fp", type=str, required=True, help="dir for rosbags to train from"
    )
    parser.add_argument(
        "--preprocess_fp",
        type=str,
        required=True,
        help="dir to save preprocessed data to",
    )
    parser.add_argument(
        "--map_topic",
        type=str,
        required=False,
        default="/local_gridmap",
        help="topic to extract map features from",
    )
    parser.add_argument(
        "--odom_topic",
        type=str,
        required=False,
        default="/integrated_to_init",
        help="topic to extract odom from",
    )
    parser.add_argument(
        "--image_topic",
        type=str,
        required=False,
        default="/multisense/left/image_rect_color",
        help="topic to extract images from",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="set this flag if you want the pyplot viz. Default is to save to folder",
    )
    args = parser.parse_args()

    model = torch.load(args.model_fp, map_location="cpu")

    dataset = MaxEntIRLDataset(
        bag_fp=args.bag_fp,
        preprocess_fp=args.preprocess_fp,
        map_features_topic=args.map_topic,
        odom_topic=args.odom_topic,
        image_topic=args.image_topic,
        horizon=model.expert_dataset.horizon,
        feature_keys=model.expert_dataset.feature_keys,
    )

    model.expert_dataset = dataset
    model.network.eval()

    maybe_mkdir(args.save_fp, force=False)
    subdirs = ["FPV", "map", "cvar-0.9", "cvar0.0", "cvar0.9", "baseline"]
    for subdir in subdirs:
        maybe_mkdir(os.path.join(args.save_fp, subdir), force=True)

    for i in range(len(dataset)):
        print("{}/{}".format(i + 1, len(dataset)), end="\r")
        if args.viz:
            figs, axs = visualize_cvar(model, idx=-1)
            plt.show()
        else:
            figs, axs = visualize_cvar(model, idx=i)

            for fig, ax, subdir in zip(figs, axs, subdirs):
                fig_fp = os.path.join(args.save_fp, subdir, "{:05d}.png".format(i + 1))
                fig.savefig(fig_fp, bbox_inches="tight")

            plt.close("all")
