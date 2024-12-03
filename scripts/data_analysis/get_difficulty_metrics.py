"""
Compute a few difficulty metrics from a bag

Currently looking at:
    1. avg speed
    2. avg yawrate
    3. total change in z

avg over a set of 1s windows, I think
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

from maxent_irl_maps.os_utils import walk_bags
from maxent_irl_maps.preprocess import load_traj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag_dir", type=str, required=True, help="root dir of bags to analyze"
    )
    parser.add_argument(
        "--odom_topic",
        type=str,
        required=False,
        default="/integrated_to_init",
        help="topic to get odom from",
    )
    parser.add_argument(
        "--window",
        type=float,
        required=False,
        default=1.0,
        help="time(s) to take avg over",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=False,
        default=10,
        help="number of timesteps per window",
    )
    args = parser.parse_args()

    speeds = []
    yawrates = []
    dzs = []
    dt = args.window / args.n

    cnt = 0
    bag_fps = walk_bags(args.bag_dir)
    for bfp in bag_fps:
        print(cnt, end="\r")
        try:
            traj = load_traj(bfp, args.odom_topic, dt)
            cnt += 1
            for t in range(traj.shape[0] - args.n):
                seg = traj[t : t + args.n]
                sp = np.mean(np.linalg.norm(seg[:, 7:10], axis=-1))
                yr = np.mean(np.abs(seg[:, 12]))
                dz = np.abs(seg[-1, 2] - seg[0, 2])
                if sp < 100:
                    speeds.append(sp)
                if yr < 1.0:
                    yawrates.append(yr)
                if dz < 5.0:
                    dzs.append(dz)
        except:
            print("bad bag {}".format(bfp))

    speeds = np.array(speeds)
    yawrates = np.array(yawrates)
    dzs = np.array(dzs)

    matplotlib.rcParams.update({"font.size": 18})
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].set_title("Speed")
    axs[0].hist(speeds, bins=np.linspace(0, 14, 50), density=True)
    axs[0].axvline(speeds.mean(), c="r")
    axs[0].set_xlabel("Speed (m/s)")
    axs[0].set_ylabel("Density")
    axs[0].set_ylim(0.0, 0.35)

    axs[1].set_title("Yawrates")
    axs[1].hist(yawrates, bins=np.linspace(0, 1, 50), density=True)
    axs[1].axvline(yawrates.mean(), c="r")
    axs[1].set_xlabel("Yawrate (rad/s)")
    axs[1].set_ylabel("Density")
    axs[1].set_ylim(0.0, 10)

    axs[2].set_title("Dzs")
    axs[2].hist(dzs, bins=np.linspace(0, 2, 50), density=True)
    axs[2].axvline(dzs.mean(), c="r")
    axs[2].set_xlabel("Dz (m/s)")
    axs[2].set_ylabel("Density")
    axs[2].set_ylim(0.0, 10)

    #    fig.suptitle(args.bag_dir)
    plt.savefig("wenshan.png")

#    plt.show()
