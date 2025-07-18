import os
import yaml
import tqdm
import copy
import argparse

import cv2
import torch
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

from maxent_irl_maps.geometry_utils import TrajectoryInterpolator

"""
Scripts that read rosbag_to_dataset output and make IRL datasets from it
Steps:
    1. For each traj folder
        a. Run pp net on colorized pcl and update kf
        b. Add next pose to traj buf
        c. Save:
            i. traj snippet + future traj
            ii. cmds + state
            iii. gridmaps
            iv. kf pp feats
            v. non kf pp feats
            vi. img
"""


def transform_pcl(pcl, odom):
    """
    transform a pointcloud into odom frame (assuming pcl in a local frame at odom)
    """
    htm = pose_to_htm(odom)
    pcl_pos = np.copy(pcl[:, :3])
    pcl_pos = np.concatenate(
        [pcl_pos, np.ones([pcl_pos.shape[0], 1])], axis=-1
    ).reshape(-1, 4, 1)
    pcl_t_pos = (htm @ pcl_pos).reshape(-1, 4)[:, :3]
    pcl_out = np.concatenate([pcl_t_pos, pcl[:, 3:]], axis=-1)
    return pcl_out


def pose_to_htm(pose):
    R = Rotation.from_quat(pose[3:7]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, -1] = pose[:3]
    return T


def run_preproc(config):
    ## start with file management ##
    trajdirs = os.listdir(config["src_dir"])
    if os.path.exists(config["dst_dir"]):
        x = input("{} exists. Overwrite? [Y/n]".format(config["dst_dir"]))
        if x == "n":
            exit(0)

    train_dir = os.path.join(config["dst_dir"], "train")
    test_dir = os.path.join(config["dst_dir"], "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for td in trajdirs:
        src_fp = os.path.join(config["src_dir"], td)
        dst_fp = train_dir if "train" in td else test_dir
        run_preproc_traj(src_fp, dst_fp, config)


def run_preproc_traj(traj_fp, dst_fp, config):
    print("{} -> {}".format(traj_fp, dst_fp))
    res_fp = os.path.join(dst_fp, os.path.basename(traj_fp))
    os.makedirs(res_fp, exist_ok=True)

    ## first load all the trajectories / cmds
    odom = np.loadtxt(os.path.join(traj_fp, config["odom"], "data.txt"))
    odom_ts = np.loadtxt(os.path.join(traj_fp, config["odom"], "timestamps.txt"))
    odom_dts = odom_ts[1:] - odom_ts[:-1]
    odom_mask = (odom_dts > 1e-4) & (odom_dts < 0.2)
    odom = odom[1:][odom_mask]
    odom_ts = odom_ts[1:][odom_mask]
    # note that there can still be positive timejumps in this interpolator
    odom_interp = TrajectoryInterpolator(times=odom_ts, traj=odom)

    steer_angle = np.loadtxt(os.path.join(traj_fp, config["steer_angle"], "data.txt"))
    steer_angle_ts = np.loadtxt(
        os.path.join(traj_fp, config["steer_angle"], "timestamps.txt")
    )
    steer_angle_dts = steer_angle_ts[1:] - steer_angle_ts[:-1]
    steer_angle_mask = (steer_angle_dts > 1e-4) & (steer_angle_dts < 0.2)
    steer_angle = steer_angle[1:][steer_angle_mask]
    steer_angle_ts = steer_angle_ts[1:][steer_angle_mask]
    # note that there can still be positive timejumps in this interpolator
    steer_angle_interp = scipy.interpolate.interp1d(
        steer_angle_ts, steer_angle, fill_value="extrapolate"
    )

    ## now proc gridmaps (this is the master time)
    gridmap_ts = np.loadtxt(
        os.path.join(traj_fp, config["local_gridmap"], "timestamps.txt")
    )
    img_ts = np.loadtxt(os.path.join(traj_fp, config["img"], "timestamps.txt"))
    pcl_ts = np.loadtxt(os.path.join(traj_fp, config["pcl"], "timestamps.txt"))

    valid_cnt = 0
    valid_ts = []

    for i in tqdm.tqdm(range(len(gridmap_ts))):
        gt = gridmap_ts[i]
        target_times = gt + np.arange(config["H"]) * config["dt"]

        if not all(
            [
                validate_sample_times(target_times, src)
                for src in [odom_ts, steer_angle_ts, img_ts, pcl_ts]
            ]
        ):
            # print("skipping sample {}...".format(i))
            pass
        else:
            sub_traj = odom_interp(target_times)
            sub_steer = steer_angle_interp(target_times)

            sub_traj_speed = np.linalg.norm(sub_traj[:, 7:10], axis=-1)

            if sub_traj_speed.mean() < config['min_avg_speed']:
                # print('skipping sample {}'.format(i))
                continue

            img_idx = np.argmin(np.abs(gt - img_ts))
            pcl_idx = np.argmin(np.abs(gt - pcl_ts))
            pcl_t = pcl_ts[pcl_idx]

            # bgr -> rgb
            img = cv2.imread(
                os.path.join(traj_fp, config["img"], "{:08d}.png".format(img_idx)),
                cv2.IMREAD_UNCHANGED,
            )[:, :, [2, 1, 0]]

            pcl = np.load(
                os.path.join(traj_fp, config["pcl"], "{:08d}.npy".format(pcl_idx))
            )
            # move pcl into odom frame (sampling from interp more accurate)
            pcl_tf = transform_pcl(pcl, odom_interp(pcl_t))

            gridmap_data = np.load(
                os.path.join(
                    traj_fp, config["local_gridmap"], "{:08d}_data.npy".format(i)
                )
            )

            gridmap_data[~np.isfinite(gridmap_data)] = 0.0
            gridmap_metadata = yaml.safe_load(
                open(
                    os.path.join(
                        traj_fp,
                        config["local_gridmap"],
                        "{:08d}_metadata.yaml".format(i),
                    ),
                    "r",
                )
            )

            gridmap_feature_keys = gridmap_metadata["feature_keys"]
            gridmap_metadata = {
                k: torch.tensor(v).float()
                for k, v in gridmap_metadata.items()
                if k != "feature_keys"
            }

            #features that are relative to the odom frame should be made ego-centric here
            curr_height = sub_traj[0, 3]
            for i, fk in enumerate(gridmap_feature_keys):
                if fk in [
                    'min_elevation',
                    'mean_elevation',
                    'max_elevation',
                    'terrain',
                ]:
                    gridmap_data[:, :, i] -= curr_height

            start_pos = torch.tensor(sub_traj[0, :2])
            valid = all(start_pos > gridmap_metadata['origin']) and all(start_pos < (gridmap_metadata['origin'] + gridmap_metadata['length']))
            if not valid:
                print('bad sample!')

            res = {
                "traj": torch.tensor(sub_traj).float(),
                "steer": torch.tensor(sub_steer).float(),
                "pointcloud": torch.tensor(pcl_tf).float(),
                "image": torch.tensor(img).float().permute(2, 0, 1)[[2, 1, 0]] / 255.0,
                "gridmap_data": torch.tensor(gridmap_data).float().permute(2,0,1),
                "gridmap_metadata": gridmap_metadata,
                "gridmap_feature_keys": gridmap_feature_keys
            }

            if valid_cnt % config["save_every"] == 0:
                torch.save(
                    res,
                    os.path.join(
                        res_fp, "{:08d}.pt".format(valid_cnt // config["save_every"])
                    ),
                )

            valid_cnt += 1
            valid_ts.append(gt)

    print("{}/{} samples valid".format(valid_cnt, gridmap_ts.shape[0]))


def validate_sample_times(target_times, src_times, tol=0.5):
    """
    Check if all times in target times are within tol of src_times
    """
    dists = np.abs(target_times.reshape(1, -1) - src_times.reshape(-1, 1))
    mindists = np.min(dists, axis=0)
    return (
        np.all(mindists < tol)
        and (target_times.min() > src_times.min())
        and (target_times.max() < src_times.max())
    )


def viz_sample(
    traj,
    steer,
    img,
    pcl,
    gridmap_data,
    gridmap_metadata,
    learned_gridmap_data,
    learned_gridmap_metadata,
):
    """
    viz sample for debug
    """
    extent = (
        gridmap_metadata["origin"][0],
        gridmap_metadata["origin"][0] + gridmap_metadata["width"],
        gridmap_metadata["origin"][1],
        gridmap_metadata["origin"][1] + gridmap_metadata["height"],
    )

    fig, axs = plt.subplots(2, 3, figsize=(27, 18))

    # img
    axs[0, 0].imshow(img)

    # steer
    axs[1, 0].plot(steer)
    axs[1, 0].set_ylim(-415.0, 415.0)

    # gridmap
    baseline_geom = gridmap_data[6]  # step
    axs[0, 1].imshow(baseline_geom, extent=extent, origin="lower", cmap="coolwarm")
    axs[0, 1].plot(traj[:, 0], traj[:, 1], c="y", linewidth=2)

    # pcl
    axs[1, 1].scatter(pcl[:, 0], pcl[:, 1], c=pcl[:, 3:], s=1.0)
    axs[1, 1].plot(traj[:, 0], traj[:, 1], c="y", linewidth=2)
    axs[1, 1].set_aspect(1.0)
    axs[1, 1].set_xlim(extent[0], extent[1])
    axs[1, 1].set_ylim(extent[2], extent[3])

    # learned geometry
    geom = learned_gridmap_data[0]  # local rough
    axs[0, 2].imshow(geom, extent=extent, origin="lower", cmap="coolwarm")
    axs[0, 2].plot(traj[:, 0], traj[:, 1], c="y", linewidth=2)

    # learned rgb
    rgb = learned_gridmap_data[-4:-1]  # rgb
    axs[1, 2].imshow(rgb, extent=extent, origin="lower")
    axs[1, 2].plot(traj[:, 0], traj[:, 1], c="y", linewidth=2)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_fp", type=str, required=True, help="path to config file"
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_fp, "r"))
    run_preproc(config)
