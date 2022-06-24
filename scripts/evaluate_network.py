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
if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fp', type=str, required=True, help='Costmap weights file')
    args = parser.parse_args()

    model = torch.load(args.model_fp, map_location='cpu')

    #temp line bc Sam forgot to include model in MPPI's to func
    model.mppi.model = model.mppi.model.to('cpu')
    model.mppi.cost_fn = model.mppi.cost_fn.to('cpu')

    model.visualize()
