"""
Dataset for registering a lot of runs onto a global map
We accomplish this by using GPS as a common reference frame and storing an
occupancy grid of all states

Note that this class is not a dataset and is just the occupancy grid and its
relevant methods
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_costmaps.preprocess import load_traj
from maxent_irl_costmaps.os_utils import walk_bags

class GlobalStateVisitationBuffer:
    def __init__(self, fp, gps_topic='/odometry/filtered_odom', resolution=0.5, dt=0.05):
        """
        Args:
            fp: The root dir to get trajectories from
            resolution: The discretization of the map
            dt: The dt for the trajectory
        """
        self.base_fp = fp
        self.gps_topic = gps_topic
        self.resolution = resolution
        self.dt = dt

        self.traj_fps = walk_bags(self.base_fp)
        for i, tfp in enumerate(self.traj_fps):
            print('{}/{} ({})'.format(i+1, len(self.traj_fps), os.path.basename(tfp)), end='\r')
            traj = load_traj(tfp, self.gps_topic, self.dt)

if __name__ == '__main__':
    fp = '/home/striest/Desktop/datasets/debug_walk'
    buf = GlobalStateVisitationBuffer(fp)
