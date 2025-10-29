import os

import torch
import numpy as np

from tartandriver_perception_infra.dataset.dataset import PerceptionDataset

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
        self.sample_every = config['irl']['sample_every']

        print(f"Found {len(self.rdirs)} run dirs:")
        for rdir in self.rdirs:
            print('\t' + rdir)

        self.filter_speed_idxs()

    def filter_speed_idxs(self):
        odom_dl = self.dataloaders['odometry']
        idx_hash_new = []

        ## its faster to load the data directly
        for i, rdir in enumerate(self.rdirs):
            odom_data = torch.tensor(np.loadtxt(os.path.join(rdir, odom_dl['dir'], 'data.txt')))
            idxs = self.idx_hash[self.idx_hash[:, 0] == i]

            assert odom_data.shape[0] == (idxs.shape[0] + odom_dl['tsample'].shape[0])

            H = odom_dl['tsample'].shape[0]
            N = idxs.shape[0]

            speeds = torch.linalg.norm(odom_data[:, 7:10], axis=-1)
            speed_seg_idxs = torch.arange(N).reshape(N,1) + odom_dl['tsample'].reshape(1,H)
            speed_seg = speeds[speed_seg_idxs]

            avg_speed = speed_seg.mean(dim=-1)
            valid_idxs = torch.argwhere(avg_speed > self.min_speed).squeeze()

            valid_idxs = valid_idxs[::self.sample_every]

            _ihn = torch.stack([
                torch.zeros(valid_idxs.shape[0], dtype=torch.long) + i,
                valid_idxs
            ], dim=-1)

            idx_hash_new.append(_ihn)

        idx_hash_new = torch.cat(idx_hash_new, dim=0)
        print(f"subsampled {self.idx_hash.shape[0]}->{idx_hash_new.shape[0]} dpts")
        self.idx_hash = idx_hash_new

    def __getitem__(self, idx):
        dpt = super().__getitem__(idx)
        if 'voxel_input' in dpt.keys():
            dpt['voxel_input'] = self.preproc_voxel(dpt)
        return dpt

    def getitem_batch(self, idxs):
        dpt = super().getitem_batch(idxs)
        if 'voxel_input' in dpt.keys():
            dpt['voxel_input'] = self.preproc_voxel(dpt)

        return dpt

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
