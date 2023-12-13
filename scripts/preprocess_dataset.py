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

from pointpillars_inference.kalman_filter import update_kf, shift_kf
from network_common.pointpillars.pointpillars_utils import bin_points, decorate_points

from maxent_irl_costmaps.geometry_utils import TrajectoryInterpolator

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
def run_preproc(config):
    ## start with file management ##
    trajdirs = os.listdir(config['src_dir'])
    if os.path.exists(config['dst_dir']):
        x = input('{} exists. Overwrite? [Y/n]'.format(config['dst_dir']))
        if x == 'n':
            exit(0)

    train_dir = os.path.join(config['dst_dir'], 'train')
    test_dir = os.path.join(config['dst_dir'], 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    pointpillars = torch.load(config['pointpillars'], map_location='cuda')
    pointpillars.eval()

    for td in trajdirs:
        src_fp = os.path.join(config['src_dir'], td)
        dst_fp = train_dir if 'train' in td else test_dir
        run_preproc_traj(src_fp, dst_fp, config, pointpillars)

def run_preproc_traj(traj_fp, dst_fp, config, pointpillars):
    print('{} -> {}'.format(traj_fp, dst_fp))
    res_fp = os.path.join(dst_fp, os.path.basename(traj_fp))
    os.makedirs(res_fp, exist_ok=True)

    ## first load all the trajectories / cmds
    odom = np.load(os.path.join(traj_fp, config['odom'], 'odometry.npy'))
    odom_ts = np.loadtxt(os.path.join(traj_fp, config['odom'], 'timestamps.txt'))
    odom_dts = (odom_ts[1:] - odom_ts[:-1])
    odom_mask = (odom_dts > 1e-4) & (odom_dts < 0.2)
    odom = odom[1:][odom_mask]
    odom_ts = odom_ts[1:][odom_mask]
    #note that there can still be positive timejumps in this interpolator
    odom_interp = TrajectoryInterpolator(times=odom_ts, traj=odom)

    steer_angle = np.load(os.path.join(traj_fp, config['steer_angle'], 'float.npy'))
    steer_angle_ts = np.loadtxt(os.path.join(traj_fp, config['steer_angle'], 'timestamps.txt'))
    steer_angle_dts = (steer_angle_ts[1:] - steer_angle_ts[:-1])
    steer_angle_mask = (steer_angle_dts > 1e-4) & (steer_angle_dts < 0.2)
    steer_angle = steer_angle[1:][steer_angle_mask]
    steer_angle_ts = steer_angle_ts[1:][steer_angle_mask]
    #note that there can still be positive timejumps in this interpolator
    steer_angle_interp = scipy.interpolate.interp1d(steer_angle_ts, steer_angle, fill_value='extrapolate')

    ## now proc gridmaps (this is the master time)
    gridmap_ts = np.loadtxt(os.path.join(traj_fp, config['local_gridmap'], 'timestamps.txt'))
    img_ts = np.loadtxt(os.path.join(traj_fp, config['img'], 'timestamps.txt'))
    pcl_ts = np.loadtxt(os.path.join(traj_fp, config['pcl'], 'timestamps.txt'))

    valid_cnt = 0
    valid_ts = []

    kf = {
            'geometry': None,
            'semantics': None,
            'rgb': None,
            'unc': None,
            'pose': None,
            'cnt': None,
            'resolution': None
        }

    for i in tqdm.tqdm(range(len(gridmap_ts))):
        gt = gridmap_ts[i]
        target_times = gt + np.arange(config['H']) * config['dt']

        if not all([validate_sample_times(target_times, src) for src in [odom_ts, steer_angle_ts, img_ts, pcl_ts]]):
            print('skipping sample {}...'.format(i))
            ## reset the kalman filter if pose jump
            kf = {
                    'geometry': None,
                    'semantics': None,
                    'rgb': None,
                    'unc': None,
                    'pose': None,
                    'cnt': None,
                    'resolution': None
                }
            continue
        else:
            sub_traj = odom_interp(target_times)
            sub_steer = steer_angle_interp(target_times)

            img_idx = np.argmin(np.abs(gt - img_ts))
            pcl_idx = np.argmin(np.abs(gt - pcl_ts))
            pcl_t = pcl_ts[pcl_idx]

            # bgr -> rgb
            img = cv2.imread(os.path.join(traj_fp, config['img'], '{:06d}.png'.format(img_idx)), cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]

            pcl = np.load(os.path.join(traj_fp, config['pcl'], '{:06d}.npy'.format(pcl_idx)))
            # move pcl into odom frame (sampling from interp more accurate)
            pcl_tf = transform_pcl(pcl, odom_interp(pcl_t))

            gridmap_data = np.load(os.path.join(traj_fp, config['local_gridmap'], '{:06d}_data.npy'.format(i)))

            #temp hack
            gridmap_data[~np.isfinite(gridmap_data)] = 0.
            gridmap_data = np.transpose(gridmap_data, (0,2,1))
            gridmap_metadata = yaml.safe_load(open(os.path.join(traj_fp, config['local_gridmap'], '{:06d}_metadata.yaml'.format(i)), 'r'))

            gridmap_metadata['length_x'] = gridmap_metadata['width']
            gridmap_metadata['length_y'] = gridmap_metadata['height']
            gridmap_feature_keys = gridmap_metadata['feature_keys']
            gridmap_metadata = {k:torch.tensor(v).float() for k,v in gridmap_metadata.items() if k != 'feature_keys'}


            # run pointpillars network
            pcl_in = np.copy(pcl_tf)
            pcl_in[:, 2] -= sub_traj[0, 2] #need zs to be local (x, y handled by metadata)
            pointpillars_out = run_network(pointpillars, pcl_in, gridmap_metadata)

            geometry = pointpillars_out['features'].loc[0].cpu().numpy()

            geom_unc = pointpillars_out['features'].scale[0].cpu().numpy()
            rgb_unc = pointpillars_out['rgb'].scale[0].cpu().numpy()
            rgb_pred = pointpillars_out['rgb'].loc[0].cpu().numpy()

            cnt = np.ones([1, geom_unc.shape[1], geom_unc.shape[2]])

            semantic_probs = pointpillars_out['semantics'][0].softmax(dim=0).cpu().numpy()

            kf_update = {
                'geometry': geometry,
                'geometry_unc': geom_unc,
                'semantics': semantic_probs,
                'rgb': rgb_pred,
                'rgb_unc': rgb_unc,
                'cnt': cnt,
                'pose': sub_traj[0, :2],
                'resolution': gridmap_metadata['resolution'].item()
            }

            kf = update_kf(kf, kf_update, decay=1.0)

            geometry_kf = kf['geometry']
            semantics_kf = kf['semantics']
            rgb_kf = 1. / (1 + np.exp(-kf['rgb']))

            geometry_unc = kf['geometry_unc'].sum(axis=0, keepdims=True)
            semantics_unc = (kf['semantics'] * -np.log(kf['semantics'])).sum(axis=0, keepdims=True)
            rgb_unc = kf['rgb_unc'].sum(axis=0, keepdims=True)

            learned_gridmap_data = np.concatenate([
                geometry_kf,
                geometry_unc,
                semantics_kf,
                semantics_unc,
                rgb_kf,
                rgb_unc
            ], axis=0)

            learned_gridmap_metadata = copy.deepcopy(gridmap_metadata)
            learned_gridmap_feature_keys = [
                'step',
                'local_slope',
                'local_rough',
                'robot_slope',
                'robot_rough',
                'geom_unc',
                'p_void',
                'p_sky',
                'p_dirt',
                'p_grass',
                'p_gravel',
                'p_road',
                'p_floor',
                'p_water',
                'p_bush',
                'p_tree',
                'p_wall',
                'p_rock',
                'p_object',
                'p_actor',
                'p_vehicle',
                'p_pole',
                'semantic_entropy',
                'r',
                'g',
                'b',
                'rgb_unc'
            ]

            res = {
                'traj': torch.tensor(sub_traj).float(),
                'steer': torch.tensor(sub_steer).float(),
                'pointcloud': torch.tensor(pcl_tf).float(),
                'image': torch.tensor(img).float().permute(2, 0, 1)[[2, 1, 0]] / 255.,
                'gridmap_data': torch.tensor(gridmap_data).float().permute(0, 2, 1),
                'gridmap_metadata': gridmap_metadata,
                'gridmap_feature_keys': gridmap_feature_keys,
                'learned_gridmap_data': torch.tensor(learned_gridmap_data).float().permute(0, 2, 1),
                'learned_gridmap_metadata': learned_gridmap_metadata,
                'learned_gridmap_feature_keys': learned_gridmap_feature_keys
            }

            if valid_cnt % config['save_every'] == 0:
                torch.save(res, os.path.join(res_fp, '{:06d}.pt'.format(valid_cnt//config['save_every'])))

            # note that this viz may be off but the maxent viz is ok
#            if valid_cnt % 400 == 0:
#                viz_sample(sub_traj, sub_steer, img, pcl_tf, gridmap_data, gridmap_metadata, learned_gridmap_data, learned_gridmap_metadata)

            valid_cnt += 1
            valid_ts.append(gt)

    print('{}/{} samples valid'.format(valid_cnt, gridmap_ts.shape[0]))

    ## debug plot ##
#    fig, axs = plt.subplots(2, 2)
#    axs = axs.flatten()
#    for ax in axs:
#        ax.plot(valid_ts, valid_ts, c='r', marker='x', label='samples')
#    for ax, data in zip(axs, [odom_ts, steer_angle_ts, img_ts, pcl_ts]):
#        ax.plot(odom_ts, odom_ts, c='g', marker='.', label='odom')
#    for ax, label in zip(axs, ['odom', 'steer', 'img', 'pcl']):
#        ax.set_title(label)
#    plt.show()

def run_network(net, pcl, metadata):
    acc, cnt, mask = bin_points(pcl, metadata, max_pts=64, device=net.device)
    acc = torch.tensor(acc, device=net.device).float()
    cnt = torch.tensor(cnt, device=net.device).float()
    mask = torch.tensor(mask, device=net.device).float()

    acc = decorate_points(acc, cnt, mask, metadata)

    res = acc.unsqueeze(0)

    ## run the network ##
    with torch.no_grad():
        res = net.forward(res)

    return res

def transform_pcl(pcl, odom):
    """
    transform a pointcloud into odom frame (assuming pcl in a local frame at odom)
    """
    htm = pose_to_htm(odom)
    pcl_pos = np.copy(pcl[:, :3])
    pcl_pos = np.concatenate([pcl_pos, np.ones([pcl_pos.shape[0], 1])], axis=-1).reshape(-1, 4, 1)
    pcl_t_pos = (htm @ pcl_pos).reshape(-1, 4)[:, :3]
    pcl_out = np.concatenate([pcl_t_pos, pcl[:, 3:]], axis=-1)
    return pcl_out

def pose_to_htm(pose):
    R = Rotation.from_quat(pose[3:7]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, -1] = pose[:3]
    return T

def validate_sample_times(target_times, src_times, tol=0.5):
    """
    Check if all times in target times are within tol of src_times
    """
    dists = np.abs(target_times.reshape(1, -1) - src_times.reshape(-1, 1))
    mindists = np.min(dists, axis=0)
    return np.all(mindists < tol) and (target_times.min() > src_times.min()) and (target_times.max() < src_times.max())

def viz_sample(traj, steer, img, pcl, gridmap_data, gridmap_metadata, learned_gridmap_data, learned_gridmap_metadata):
    """
    viz sample for debug
    """
    extent = (
        gridmap_metadata['origin'][0], 
        gridmap_metadata['origin'][0] + gridmap_metadata['width'], 
        gridmap_metadata['origin'][1], 
        gridmap_metadata['origin'][1] + gridmap_metadata['height']
    )

    fig, axs = plt.subplots(2, 3, figsize=(27, 18))

    # img
    axs[0, 0].imshow(img)

    # steer
    axs[1, 0].plot(steer)
    axs[1, 0].set_ylim(-415., 415.)

    # gridmap
    baseline_geom = gridmap_data[6] #step
    axs[0, 1].imshow(baseline_geom, extent=extent, origin='lower', cmap='coolwarm')
    axs[0, 1].plot(traj[:, 0], traj[:, 1], c='y', linewidth=2)

    # pcl
    axs[1, 1].scatter(pcl[:, 0], pcl[:, 1], c=pcl[:, 3:], s=1.)
    axs[1, 1].plot(traj[:, 0], traj[:, 1], c='y', linewidth=2)
    axs[1, 1].set_aspect(1.)
    axs[1, 1].set_xlim(extent[0], extent[1])
    axs[1, 1].set_ylim(extent[2], extent[3])

    # learned geometry
    geom = learned_gridmap_data[0] #local rough
    axs[0, 2].imshow(geom, extent=extent, origin='lower', cmap='coolwarm')
    axs[0, 2].plot(traj[:, 0], traj[:, 1], c='y', linewidth=2)

    # learned rgb
    rgb = learned_gridmap_data[-4:-1] #rgb
    axs[1, 2].imshow(rgb, extent=extent, origin='lower')
    axs[1, 2].plot(traj[:, 0], traj[:, 1], c='y', linewidth=2)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True, help='path to config file')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_fp, 'r'))
    run_preproc(config)
