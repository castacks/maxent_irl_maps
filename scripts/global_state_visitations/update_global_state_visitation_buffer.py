"""
Script for updating a global state visitation buffer w. GUI
"""
import os
import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd

from maxent_irl_costmaps.dataset.global_state_visitation_buffer import GlobalStateVisitationBuffer
from maxent_irl_costmaps.os_utils import walk_bags

#local scripts
from plot_global_state_visitations import plot_gsv, plot_utm_traj
from google_static_maps_api import GoogleStaticMapsAPI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gsv_fp', type=str, required=True, help='location to get global state_vistations')
    parser.add_argument('--utm_zone', type=int, required=False, default=17, help='UTM zone that the data is in (probably 17 if CMU data)')
    parser.add_argument('--bag_dir', type=str, required=True, help='dir of bags to add')
    args = parser.parse_args()

    buf = torch.load(args.gsv_fp)

    bag_fps = walk_bags(args.bag_dir)

    print('Processing the following bags:')
    for bfp in bag_fps:
        print('\t{}'.format(bfp))

    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    axs = [axs]
    plt.show(block=False)

    for i, bfp in enumerate(bag_fps):
        print('{} ({}/{})'.format(bfp, i, len(bag_fps)))
        axs[0].cla()

        #plot the current buffer
        fig, axs = plot_gsv(buf, args.utm_zone, fig, axs)

        #show the new traj
        res, traj = buf.check_trajectory(bfp)
        if res:
            fig, axs = plot_utm_traj(buf, args.utm_zone, traj, fig, axs, {'c':'y'})
            plt.pause(1e-2)

            res2 = input('Add trajectory (in yellow) to buf? [Y/n]:')
            if res2 != 'n':
                buf.insert_traj(traj)

            torch.save(buf, args.gsv_fp)

    axs[0].cla()
    fig, axs = plot_gsv(buf, args.utm_zone, fig, axs)
    plt.pause(1e-2)
    input('done (enter to exit)')

