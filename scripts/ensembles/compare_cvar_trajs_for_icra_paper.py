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
from torch_mpc.cost_functions.batch_multi_waypoint_costmap import (
    BatchMultiWaypointCostMapCostFunction,
)

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_maps.networks.baseline_lethal_height import LethalHeightCostmap
from maxent_irl_maps.os_utils import maybe_mkdir


def visualize_cvar(model, idx):
    """
    Much like before, visualize the CVaR maps, but also run a batch MPPI to get behavior differences
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

        # compute costmap
        # resnet cnn
        mosaic = """
        ACD
        BHI
        """

        fig = plt.figure(tight_layout=True, figsize=(18, 12))
        ax_dict = fig.subplot_mosaic(mosaic)
        all_axs = [ax_dict[k] for k in sorted(ax_dict.keys())]
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

        costmaps = torch.stack(costmap_cvars, dim=0)

        baseline_res = baseline.forward(map_features.view(1, *map_features.shape))
        baseline_costmap = baseline_res["costmap"][0]
        costmaps = torch.cat([costmaps, baseline_costmap], dim=0)

        # run MPPI opt
        # initialize solver
        initial_state = expert_traj[0]
        x0 = {
            "state": initial_state,
            "steer_angle": data["steer"][[0]]
            if "steer" in data.keys()
            else torch.zeros(1, device=initial_state.device),
        }
        x = torch.stack([model.mppi.model.get_observations(x0)] * 4, dim=0)

        map_params = {
            "resolution": metadata["resolution"],
            "height": metadata["height"],
            "width": metadata["width"],
            "origin": torch.stack([metadata["origin"]] * 4, dim=0),
        }

        model.mppi.reset()
        model.mppi.cost_fn.update_map_params(map_params)
        model.mppi.cost_fn.update_costmap(costmaps)
        model.mppi.cost_fn.update_goals([expert_traj[[-1], :2]] * 4)

        # solve for traj
        for ii in range(model.mppi_itrs):
            model.mppi.get_control(x, step=False, match_noise=True)

        # regular version
        trajs = model.mppi.last_states
        costs = model.mppi.last_cost

        labels = ["CVaR -0.9", "CVaR 0.0", "CVaR 0.9", "Occupancy"]
        colors = "bgyr"

        idx = model.expert_dataset.feature_keys.index("height_high")
        axs1[0].imshow(data["image"].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
        axs1[1].imshow(
            map_features[idx].cpu(),
            origin="lower",
            cmap="gray",
            extent=(xmin, xmax, ymin, ymax),
        )

        for st, label, color in zip(trajs, labels, colors):
            axs1[1].plot(st[:, 0].cpu(), st[:, 1].cpu(), c=color, label=label)

        axs1[0].set_title("FPV")
        axs1[1].set_title("Height High")
        axs1[1].legend()

        vmin = torch.quantile(torch.stack(costmap_cvars), 0.1)
        vmax = torch.quantile(torch.stack(costmap_cvars), 0.9)

        for ax, cm, traj, cost, label, color in zip(
            axs2, costmaps, trajs, costs, labels, colors
        ):
            if label == "Occupancy":
                r = ax.imshow(
                    cm.cpu(),
                    origin="lower",
                    cmap="plasma",
                    extent=(xmin, xmax, ymin, ymax),
                )
            else:
                r = ax.imshow(
                    cm.cpu(),
                    origin="lower",
                    cmap="plasma",
                    extent=(xmin, xmax, ymin, ymax),
                    vmin=vmin,
                    vmax=vmax,
                )
            ax.plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c=color)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title("{} Cost = {:.2f}".format(label, cost))

        model.mppi.reset()


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
        "--skip", type=int, required=False, default=1, help="process very k-th frame"
    )
    parser.add_argument(
        "--device", type=str, required=False, default="cpu", help="device to run on "
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="set this flag if you want the pyplot viz. Default is to save to folder",
    )
    args = parser.parse_args()

    model = torch.load(args.model_fp, map_location=args.device)

    dataset = MaxEntIRLDataset(
        bag_fp=args.bag_fp,
        preprocess_fp=args.preprocess_fp,
        map_features_topic=args.map_topic,
        odom_topic=args.odom_topic,
        image_topic=args.image_topic,
        horizon=model.expert_dataset.horizon,
    ).to(args.device)

    model.expert_dataset = dataset
    model.network.eval()

    # set up batch MPPI
    cfn = BatchMultiWaypointCostMapCostFunction(
        unknown_cost=model.mppi.cost_fn.unknown_cost,
        goal_cost=model.mppi.cost_fn.goal_cost,
    )

    # probably don't want to hard-code these params...
    mppi_params = {
        "sys_noise": torch.tensor([1.0, 0.1]),
        "temperature": 0.1,
        "use_ou": True,
        "ou_alpha": 0.9,
        "ou_scale": 10.0,
        "d_ou_scale": 5.0,
    }

    batch_mppi = BatchMPPI(
        batch_size=4,
        model=model.mppi.model,
        cost_fn=cfn,
        #        num_samples = model.mppi.K1,
        #        num_uniform_samples = model.mppi.K2,
        num_samples=2048,
        num_uniform_samples=100,
        num_timesteps=model.mppi.T,
        control_params=mppi_params,
    )

    model.mppi = batch_mppi.to(args.device)
    model.mppi_itrs = 20

    # re-adjust the speeds to reflect the actual test speed lim
    model.mppi.model.u_lb[0] = 1.5
    model.mppi.model.u_ub[0] = 3.5

    maybe_mkdir(args.save_fp, force=False)

    for i in range(0, len(dataset), args.skip):
        print("{}/{}".format(i + 1, len(dataset)), end="\r")
        fig_fp = os.path.join(args.save_fp, "{:05d}.png".format(i + 1))
        if args.viz:
            visualize_cvar(model, idx=-1)
            plt.show()
        else:
            visualize_cvar(model, idx=i)
            plt.savefig(fig_fp)
            plt.close()
