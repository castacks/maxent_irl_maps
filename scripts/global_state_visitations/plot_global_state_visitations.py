"""
Viz script for visualizing a global state visitation buffer on top of satellite imagery
"""

import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse

from pyproj import Proj
from google_static_maps_api import GoogleStaticMapsAPI
from google_static_maps_api import MAPTYPE
from google_static_maps_api import MAX_SIZE
from google_static_maps_api import SCALE

from maxent_irl_costmaps.dataset.global_state_visitation_buffer import GlobalStateVisitationBuffer

def get_lat_long_grid(gsv, utm_zone):
    """
    convert all the cells in the gsv buffer into lat/long for plotting
    """
    ox = gsv.metadata['origin'][0].item()
    oy = gsv.metadata['origin'][1].item()
    dx = gsv.metadata['length_x']
    dy = gsv.metadata['length_y']
    res = gsv.metadata['resolution']

    proj = Proj("+proj=utm +zone={} +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(utm_zone))
    grid = np.stack(np.meshgrid(
        np.arange(ox, ox + dx, res),
        np.arange(oy, oy + dy, res),
        indexing='ij'
    ), axis=-1)

    #for some reason, the coordinates got messed up. Need to 
    lon, lat = proj(-grid[:, :, 1], grid[:, :, 0], inverse=True)
    return np.stack([lat, lon], axis=-1)

def plot_gsv(gsv, utm_zone, fig=None, axs=None, plot_speed=False):
    """
    use google maps API to get satellite imagery and heatmap the global state visitations on top of it
    """
    if fig is None or axs is None:
        if plot_speed:
            fig, axs = plt.subplots(1, 2, figsize=(24, 12))
        else:
            fig, axs = plt.subplots(1, 1, figsize=(12, 12))
            axs = [axs]

    ox = gsv.metadata['origin'][0].item()
    oy = gsv.metadata['origin'][1].item()
    dx = gsv.metadata['length_x']
    dy = gsv.metadata['length_y']
    res = gsv.metadata['resolution']

    grid = get_lat_long_grid(gsv, str(utm_zone))
    xmid = int(grid.shape[0]/2)
    ymid = int(grid.shape[1]/2)
    gsv_center = (grid[xmid, ymid, 0], grid[xmid, ymid, 1])

    GoogleStaticMapsAPI.register_api_key('AIzaSyCnxI2B01c7kxX-t8yxyJfunjfxaFfdspk')
    img = GoogleStaticMapsAPI.map(
        center=gsv_center,
        zoom=15,
        maptype='satellite',
    )
    grid_px_df = GoogleStaticMapsAPI.to_tile_coordinates(
        pd.Series(grid[:, :, 0].flatten()),
        pd.Series(grid[:, :, 1].flatten()),
        gsv_center[0],
        gsv_center[1],
        15,
        MAX_SIZE,
        2
    )

    grid_px = np.stack([
        grid_px_df['y_pixel'].to_numpy(),
        grid_px_df['x_pixel'].to_numpy()
    ], axis=-1)
    grid_px = grid_px.reshape(grid.shape)
#    speed_data = gsv.get_mean_speed_map()

    mask = (gsv.data > 0).numpy()

    mask_px = grid_px[mask]
    mask_val = gsv.data[mask].numpy()
#    mask_speed_val = speed_data[mask].numpy()

    cmap = matplotlib.cm.get_cmap('coolwarm')

    if mask.sum() > 0:
        r1 = axs[0].scatter(mask_px[:, 1], mask_px[:, 0], s=1., c=np.log(mask_val), cmap=cmap)
#        fig.colorbar(r1, ax=axs[0])

#        if plot_speed:
#            r2 = axs[1].scatter(mask_px[:, 1], mask_px[:, 0], s=1., c=mask_speed_val, cmap=cmap)
#            fig.colorbar(r2, ax=axs[1])

    axs[0].imshow(img)
    axs[0].set_title('Log State Visitations')

#    if plot_speed:
#        axs[1].imshow(img)
#        axs[1].set_title('Speeds')
    return fig, axs

def plot_utm_traj(gsv, utm_zone, traj, fig=None, axs=None, plt_kwargs=None):
    """
    Plot a traj in utm on a gsv
    """
    if fig is None or axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(24, 12))

    ox = gsv.metadata['origin'][0].item()
    oy = gsv.metadata['origin'][1].item()
    dx = gsv.metadata['length_x']
    dy = gsv.metadata['length_y']
    res = gsv.metadata['resolution']

    proj = Proj("+proj=utm +zone={} +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(utm_zone))

    #for some reason, the coordinates got messed up. Need to 
    traj_lon, traj_lat = proj(-traj[:, 1], traj[:, 0], inverse=True)

    grid = get_lat_long_grid(gsv, str(utm_zone))
    xmid = int(grid.shape[0]/2)
    ymid = int(grid.shape[1]/2)
    gsv_center = (grid[xmid, ymid, 0], grid[xmid, ymid, 1])

    grid_px_df = GoogleStaticMapsAPI.to_tile_coordinates(
        pd.Series(traj_lat),
        pd.Series(traj_lon),
        gsv_center[0],
        gsv_center[1],
        15,
        MAX_SIZE,
        2
    )

    grid_px = np.stack([
        grid_px_df['y_pixel'].to_numpy(),
        grid_px_df['x_pixel'].to_numpy()
    ], axis=-1)

    axs[0].plot(grid_px[:, 1], grid_px[:, 0], **plt_kwargs)

    return fig, axs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buf_fp', type=str, required=True, help='path to gsv buffer')
    parser.add_argument('--utm_zone', type=int, required=False, default=17, help='UTM zone for data')
    args = parser.parse_args()

    buf = torch.load(args.buf_fp)
    plot_gsv(buf, args.utm_zone)
    plt.show()
