import os

import torch
import numpy as np

from bev_prediction.utils import BEV_ELEVATION_KEYS

from tartandriver_perception_infra.dataset.dataset import PerceptionDataset

from tartandriver_utils.geometry_utils import TrajectoryInterpolator

class MaxEntIRLDataset(PerceptionDataset):
    """
    Wrapper around base perception dataset that:
        1. Adds a min speed filter
        2. TODO computes feature normalizations
    """
    def __init__(self, config):
        super(MaxEntIRLDataset, self).__init__(config)

        ## only assert these two because the perception inputs can change
        assert 'odometry' in self.dataloaders.keys(), "Expect 'odometry to be a data key'"
        assert 'steer_angle' in self.dataloaders.keys(), "Expect 'steer_angle to be a data key'"
        assert 'tsample' in self.dataloaders['odometry'].keys(), "Need to forward sample odometry for IRL"
        assert 'tsample' in self.dataloaders['steer_angle'].keys(), "Need to forward sample steer angle for IRL"
        assert (self.dataloaders['odometry']['tsample'] == self.dataloaders['steer_angle']['tsample']).all()

        self.min_speed = config['irl']['min_speed']
        self.max_ds = config['irl']['max_ds']
        self.sample_every = config['irl']['sample_every']

        self.use_distance_based_sampling = 'distance_to_sample' in config['irl'].keys()

        print(f"Found {len(self.rdirs)} run dirs:")
        for rdir in self.rdirs:
            print('\t' + rdir)

        self.filter_speed_idxs()

        if self.use_distance_based_sampling:
            self.sample_traj_distance = config['irl']['distance_to_sample']
            self.precompute_distances()

    def filter_speed_idxs(self):
        odom_dl = self.dataloaders['odometry']
        idx_hash_new = []

        ## its faster to load the data directly
        for i, rdir in enumerate(self.rdirs):
            odom_data = torch.tensor(np.loadtxt(os.path.join(rdir, odom_dl['dir'], 'data.txt')))
            idxs = self.idx_hash[self.idx_hash[:, 0] == i]

            ## dataset may contain zero samples from a run after filtering
            if len(idxs) == 0:
                continue

            assert odom_data.shape[0] > (idxs[:, 1].max() + odom_dl['tsample'].shape[0])

            H = odom_dl['tsample'].shape[0]
            N = idxs.shape[0]

            speeds = torch.linalg.norm(odom_data[:, 7:10], axis=-1)
            speed_seg_idxs = idxs[:, 1].reshape(N,1) + odom_dl['tsample'].reshape(1,H)
            speed_seg = speeds[speed_seg_idxs]

            avg_speed = speed_seg.mean(dim=-1)

            ## also filter pose jumps
            xypos = odom_data[:, :2]
            pos_seg = xypos[speed_seg_idxs]
            seg_ds = torch.linalg.norm(pos_seg[:, 1:] - pos_seg[:, :-1], dim=-1)
            max_disp = seg_ds.max(dim=-1)[0]

            valid_mask = (avg_speed > self.min_speed) & (max_disp < self.max_ds)

            valid_idxs = torch.argwhere(valid_mask).squeeze()

            valid_idxs = valid_idxs[::self.sample_every]

            _ihn = torch.stack([
                torch.zeros(valid_idxs.shape[0], dtype=torch.long) + i,
                valid_idxs
            ], dim=-1)

            idx_hash_new.append(_ihn)

        idx_hash_new = torch.cat(idx_hash_new, dim=0)
        print(f"subsampled {self.idx_hash.shape[0]}->{idx_hash_new.shape[0]} dpts")
        self.idx_hash = idx_hash_new

    def precompute_distances(self):
        """
        precompute sample windows for distance-based sampling
        e.g. for every datapoint in the dataset, find the idx where the distance window has been traversed
        """
        odom_dl = self.dataloaders['odometry']

        ## first step is to compute cumulative distance for every run dir
        ## also just cache the trajdata bc we need to re-use it
        self.rdir_cdist = {}
        self.rdir_traj = {}
        for rdir in self.rdirs:
            #easier to just load the npy directly
            all_odom = torch.tensor(np.loadtxt(os.path.join(rdir, odom_dl['dir'], 'data.txt')))
            odom_xy = all_odom[:, :2]
            odom_ds = torch.linalg.norm(odom_xy[1:] - odom_xy[:-1], axis=-1)
            #handle pose jumps
            odom_ds[odom_ds > self.max_ds] = 0.
            odom_cdist = torch.cumsum(odom_ds, dim=0)
            self.rdir_cdist[rdir] = odom_cdist
            self.rdir_traj[rdir] = all_odom.to(self.device)

        ## now for all dpts, find the idx where cdist[idx1] + dist = cdist[idx2]
        self.dist_stop_idxs = {rdir:{} for rdir in self.rdirs}
        for ii, (rdir_idx, subidx) in enumerate(self.idx_hash):
            rdir = self.rdirs[rdir_idx]
            traj_cdist = self.rdir_cdist[rdir]
            curr_cdist = traj_cdist[subidx]

            dist_to_target = (traj_cdist - (curr_cdist + self.sample_traj_distance)).abs()
            end_ii = torch.argmin(dist_to_target)
            self.dist_stop_idxs[rdir][subidx.item()] = end_ii.item()

    def __getitem__(self, idx):
        dpt = super().__getitem__(idx)
        if 'voxel_input' in dpt.keys():
            dpt['voxel_input'] = self.preproc_voxel(dpt)

        if 'bev_input' in dpt.keys():
            dpt['bev_input'] = self.preproc_bev(dpt, dpt['bev_input'])
        
        if self.use_distance_based_sampling:
            self.resample_distances(dpt)

        return dpt

    def getitem_batch(self, idxs):
        dpt = super().getitem_batch(idxs)
        if 'voxel_input' in dpt.keys():
            dpt['voxel_input'] = self.preproc_voxel(dpt)

        if 'bev_input' in dpt.keys():
            dpt['bev_input'] = self.preproc_bev(dpt, dpt['bev_input'])

        if self.use_distance_based_sampling:
            self.resample_distances_batch(dpt)

        return dpt
    
    def resample_distances_batch(self, dpt):
        B = dpt['odometry']['data'].shape[0]
        odom_resample = {
            'data': [],
            'stamp': []
        }
        for bi in range(B):
            subdpt = {
                'rdir': dpt['rdir'][bi],
                'subidx': dpt['subidx'][bi],
                'odometry': {
                    'data': dpt['odometry']['data'][bi],
                    'stamp': dpt['odometry']['stamp'][bi],
                }
            }
            self.resample_distances(subdpt)
            odom_resample['data'].append(subdpt['odometry']['data'])
            odom_resample['stamp'].append(subdpt['odometry']['stamp'])

        odom_resample = {k:torch.stack(v, dim=0) for k,v in odom_resample.items()}

        dpt['odometry_old'] = dpt['odometry']
        dpt['odometry'] = odom_resample

    def resample_distances(self, dpt):
        H = dpt['odometry']['data'].shape[0]

        rdir = dpt['rdir']
        subidx = dpt['subidx'].item()
        stop_idx = self.dist_stop_idxs[rdir][subidx]

        traj = self.rdir_traj[rdir]
        cdist = self.rdir_cdist[rdir]

        traj_seg = traj[subidx:stop_idx]
        cdist_seg = cdist[subidx:stop_idx]
        z = (cdist_seg - cdist_seg[0]) / (cdist_seg.max() - cdist_seg.min())

        z_target = torch.linspace(0., 1., H)

        traj_interpolator = TrajectoryInterpolator(z, traj_seg)
        traj_resample = torch.stack([traj_interpolator(_z) for _z in z_target], dim=0)
        traj_resample = traj_resample.float().to(self.device)

        dpt['odometry_old'] = dpt['odometry']
        dpt['odometry'] = {
            'data': traj_resample,
            'stamp': dpt['odometry_old']['stamp'].clone()
        }

        # import matplotlib.pyplot as plt
        # plt.plot(traj_seg[:, 0].cpu().numpy(), traj_seg[:, 1].cpu().numpy(), c='b', marker='.', label='traj orig')
        # plt.plot(traj_resample[:, 0].cpu().numpy(), traj_resample[:, 1].cpu().numpy(), c='r', marker='x', label='traj_resample')
        # plt.legend()
        # plt.gca().set_aspect(1.)
        # plt.show()

    def preproc_voxel(self, dpt):
        curr_heights = dpt['odometry']['data'][..., 0, 2]
        voxel_data = dpt['voxel_input']

        voxel_data['metadata'].origin[..., 2] -= curr_heights

        #blarg
        if curr_heights.ndim == 0:
            curr_heights = curr_heights.unsqueeze(0)

        bidxs = voxel_data['data'].indices[:, 0]
        features = voxel_data['data'].features
        fks = voxel_data['feature_keys']

        idxs_to_update = [i for i in range(len(fks)) if fks.label[i] in ['zmin', 'zmax']]

        features[:, idxs_to_update] -= curr_heights[bidxs].unsqueeze(-1)

        voxel_data['data'] = voxel_data['data'].replace_feature(features)

        return voxel_data
    
    def preproc_bev(self, dpt, bev_dpt):
        """
        Return two bev maps because we dont want to normalize the mask/d2traj
        """
        curr_heights = dpt['odometry']['data'][..., 0, 2]

        #blarg
        if curr_heights.ndim == 0:
            curr_heights = curr_heights.unsqueeze(0)

        bev_fks = bev_dpt['feature_keys']
        bev_data = bev_dpt['data']

        terrain_idxs_to_local = [i for i,k in enumerate(bev_fks.label) if k in BEV_ELEVATION_KEYS]

        ndims = len(bev_data.shape) - 1
        bshape = [-1] + [1] * ndims

        terrain_feats_to_update = bev_data[..., terrain_idxs_to_local, :, :]
        is_placeholder = terrain_feats_to_update.abs() < 1e-8

        terrain_feats_local = terrain_feats_to_update - curr_heights.view(*bshape)

        #keep the placeholder = 0 for non-valid cells
        terrain_feats_to_update = torch.where(is_placeholder, 0., terrain_feats_local)

        bev_data[..., terrain_idxs_to_local, :, :] = terrain_feats_to_update

        bev_feats_dpt = {
            'metadata': bev_dpt['metadata'],
            'feature_keys': bev_dpt['feature_keys'],
            'data': bev_data,
            'stamp': bev_dpt['stamp']
        }

        return bev_feats_dpt