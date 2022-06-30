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
import argparse

from maxent_irl_costmaps.os_utils import walk_bags
from maxent_irl_costmaps.preprocess import load_traj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_dir', type=str, required=True, help='root dir of bags to analyze')
    parser.add_argument('--odom_topic', type=str, required=False, default='/integrated_to_init', help='topic to get odom from')
    parser.add_argument('--window', type=float, required=False, default=1.0, help='time(s) to take avg over')
    parser.add_argument('--n', type=int, required=False, default=10, help='number of timesteps per window')
    args = parser.parse_args()

    speeds = []
    yawrates = []
    dzs = []
    dt = args.window / args.n

    bag_fps = walk_bags(args.bag_dir)
    for bfp in bag_fps: 
        traj = load_traj(bfp, args.odom_topic, dt)
        for t in range(traj.shape[0] - args.n):
            seg = traj[t:t+args.n]
            sp = np.mean(np.linalg.norm(seg[:, 7:10], axis=-1))
            yr = np.mean(np.abs(seg[:, 12]))
            dz = np.abs(seg[-1, 2] - seg[0, 2])
            speeds.append(sp)
            yawrates.append(yr)
            dzs.append(dz)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].set_title('Speed')
    axs[0].hist(speeds, bins=50)

    axs[1].set_title('Yawrates')
    axs[1].hist(yawrates, bins=50)

    axs[2].set_title('Dzs')
    axs[2].hist(dzs, bins=50)

    fig.suptitle(args.bag_dir)

    plt.show()
