import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

from maxent_irl_costmaps.os_utils import walk_bags

class PreprocessPointpillarsDataset(Dataset):
    """
    Dataset for pointpillars + maxent IRL
    """
    def __init__(self, preprocess_fp, gridmap_type, feature_mean=None, feature_std=None):
        """
        Args:
        """
        self.preprocess_fp = preprocess_fp
        self.gridmap_type = gridmap_type
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.device = 'cpu'

        self.initialize_dataset()

    def initialize_dataset(self):
        self.fps = walk_bags(self.preprocess_fp, extension='.pt')
        print('found {} datapoints'.format(len(self)))

        preprocess = self.feature_mean is None or self.feature_std is None

        if preprocess:
            nfeats = len(torch.load(self.fps[0])['{}_feature_keys'.format(self.gridmap_type)])
            self.feature_mean = torch.zeros(nfeats)
            self.feature_var = torch.zeros(nfeats)
            self.feature_std = torch.zeros(nfeats)

        dpt = self[0]
        self.feature_keys = dpt['feature_keys']
        self.metadata = dpt['metadata']
        self.horizon = dpt['traj'].shape[0]
        print('feature_keys: {}'.format(dpt['feature_keys']))

        if preprocess:
            K = 1
            for i, fp in enumerate(self.fps):
                print('{}/{} ({})'.format(i+1, len(self), fp), end='\r')
                traj = torch.load(self.fps[i])
                mapfeats = traj['{}_data'.format(self.gridmap_type)]

                if torch.any(mapfeats.abs() > 1e6):
                    continue

                if not torch.isfinite(mapfeats).all():
                    print('nan in map features!')
                    continue


                mapfeats = mapfeats[:, 10:-10, 10:-10].reshape(nfeats, -1)
                k = mapfeats.shape[-1]

                mean_new = (self.feature_mean + (mapfeats.sum(dim=-1) / K)) * (K / (K+k))
                x_ssd = (mapfeats - mean_new.view(-1, 1)).pow(2).sum(dim=-1)
                mean_diff = self.feature_mean - mean_new
                x_correction = (mapfeats - mean_new.view(-1, 1)).sum(dim=-1)
                d_ssd = x_ssd + mean_diff * x_correction
                var_new = (self.feature_var + (d_ssd / K)) * (K / (K+k))

                self.feature_mean = mean_new
                self.feature_var = var_new
                K += k

            self.feature_std = self.feature_var.sqrt() + 1e-6
            self.feature_std[~torch.isfinite(self.feature_std)] = 1e-6

        print('feat mean = {}'.format(self.feature_mean))
        print('feat std  = {}'.format(self.feature_std))

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        dpt = torch.load(self.fps[idx], map_location=self.device)

        data = {
            'map_features': dpt['{}_data'.format(self.gridmap_type)],
            'metadata': {k:v for k,v in dpt['{}_metadata'.format(self.gridmap_type)].items() if k != 'feature_keys'},
            'steer': dpt['steer'],
            'traj': dpt['traj'],
            'image': dpt['image'],
            'feature_keys': dpt['{}_feature_keys'.format(self.gridmap_type)]
        }

        data['map_features'] = ((data['map_features'] - self.feature_mean.view(-1, 1, 1)) / self.feature_std.view(-1, 1, 1)).clip(-10, 10)
        return data

    def to(self, device):
        self.device = device
        self.feature_mean = self.feature_mean.to(self.device)
        self.feature_std = self.feature_std.to(self.device)
        return self
