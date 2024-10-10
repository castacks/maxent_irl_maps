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
from maxent_irl_costmaps.dataset.global_state_visitation_buffer import (
    GlobalStateVisitationBuffer,
)
from maxent_irl_costmaps.os_utils import maybe_mkdir
from maxent_irl_costmaps.metrics.metrics import *

from maxent_irl_costmaps.networks.baseline_lethal_height import LethalHeightCostmap

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_fp", type=str, required=True, help="path to save figs to"
    )
    parser.add_argument(
        "--experiment_fp",
        type=str,
        required=True,
        help="Dir pointing to experiment itrs",
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
        "--gsv_buffer_fp",
        type=str,
        required=True,
        help="path to the global state visitation buffer",
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
        "--frame_skip",
        type=int,
        required=False,
        default=10,
        help="number of frames to skip in eval",
    )
    parser.add_argument(
        "--itr_skip", type=int, required=False, default=5, help="number of itrs to skip"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        required=False,
        help="set this flag to run baseline map",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="device to run script on",
    )
    args = parser.parse_args()

    # assume the standard "itr_x.pt" format for experiment dir
    model_fps = [x for x in os.listdir(args.experiment_fp) if "itr" in x]
    # start with latest model to always run it
    model_fps = sorted(model_fps, key=lambda x: -int(x[4:-3]))
    model_fps = list(reversed(model_fps[:: args.itr_skip]))

    model = torch.load(
        os.path.join(args.experiment_fp, model_fps[0]), map_location="cpu"
    )
    dataset = MaxEntIRLDataset(
        bag_fp=args.bag_fp,
        preprocess_fp=args.preprocess_fp,
        map_features_topic=args.map_topic,
        odom_topic=args.odom_topic,
        image_topic=args.image_topic,
        horizon=model.expert_dataset.horizon,
    ).to(args.device)

    save_fp = os.path.join(args.save_fp, "experiment_metrics.pt")
    maybe_mkdir(args.save_fp, force=False)
    if os.path.exists(save_fp):
        inp = input("{} exists. Overwrite? [Y/n]")
        if inp == "n":
            exit(0)

    experiments_res = {}
    experiments_res["args"] = vars(args)
    experiments_res["data"] = {}

    for model_fp in model_fps:
        model = torch.load(
            os.path.join(args.experiment_fp, model_fp), map_location="cpu"
        ).to(args.device)
        model.network.eval()
        model.expert_dataset = dataset

        if args.baseline:
            model.network = LethalHeightCostmap(dataset).to(args.device)

        gsv = torch.load(args.gsv_buffer_fp, map_location="cpu").to(args.device)

        metrics = {
            "expert_cost": expert_cost,
            "learner_cost": learner_cost,
            "traj": position_distance,
            "kl": kl_divergence,
            "kl_global": kl_divergence_global,
            "mhd": modified_hausdorff_distance,
        }

        res = get_metrics(model, gsv, metrics, frame_skip=args.frame_skip)
        experiments_res["data"][model_fp] = res
        torch.save(
            experiments_res, save_fp
        )  # deliberate decision to save after every model eval

    print("done")
