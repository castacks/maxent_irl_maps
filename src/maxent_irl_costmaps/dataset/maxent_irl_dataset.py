import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import os
import tqdm
import matplotlib.pyplot as plt

from maxent_irl_costmaps.os_utils import walk_bags

class MaxEntIRLDataset(Dataset):
    """
    """
    def __init__(self, root_fp, feature_keys=None, device='cpu'):
        self.root_fp = root_fp
        self.dpt_fps = walk_bags(self.root_fp, extension='.pt')

        if self.should_preprocess():
            print('preprocessing...')
            self.preprocess()

        ## load feature normalizations
        normalizations = yaml.safe_load(open(os.path.join(self.root_fp, 'normalizations.yaml'), 'r'))
        self.feature_mean = torch.tensor(normalizations['feature_mean'], device=device).float()
        self.feature_std = torch.tensor(normalizations['feature_std'], device=device).float()

        if feature_keys is None:
            self.feature_keys = normalizations['feature_keys']
        else:
            self.feature_keys = feature_keys

        self.fidxs = self.compute_feature_idxs(self.feature_keys, normalizations['feature_keys'])

        self.device = device

    def compute_feature_idxs(self, target_keys, src_keys):
        res = []
        for ti, tk in enumerate(target_keys):
            try:
                si = src_keys.index(tk)
                res.append(si)
            except:
                print('couldnt find key {} in gridmap. Check normalizations.yaml'.format(tk))
                exit(1)

        return res
        
    def should_preprocess(self):
        return not os.path.exists(os.path.join(self.root_fp, 'normalizations.yaml'))

    def preprocess(self):
        sample_dpt = torch.load(os.path.join(self.root_fp, self.dpt_fps[0]))

        fks = sample_dpt['gridmap_feature_keys']
        fbuf = torch.zeros(0, len(fks))

        for i in tqdm.tqdm(range(min(1000, len(self)))):
            idx = np.random.randint(len(self.dpt_fps))
            dfp = self.dpt_fps[idx]
            fp = os.path.join(self.root_fp, dfp)
            dpt = torch.load(fp)
            mfeats = dpt['gridmap_data']
            mask = torch.all(torch.isfinite(mfeats), axis=0)
            mfeats = mfeats.permute(1,2,0)[mask]

            sidxs = torch.randperm(len(mfeats))[:1000]

            fbuf = torch.cat([fbuf, mfeats], axis=0)

        feature_mean = fbuf.mean(dim=0)
        feature_std = fbuf.std(dim=0)

        res = {
            'feature_keys': fks,
            'feature_mean': feature_mean.numpy().tolist(),
            'feature_std': feature_std.numpy().tolist(),
        }

        with open(os.path.join(self.root_fp, 'normalizations.yaml'), 'w') as fh:
            yaml.dump(res, fh)

    def __len__(self):
        return len(self.dpt_fps)

    def __getitem__(self, idx):
        dpt = torch.load(os.path.join(self.root_fp, self.dpt_fps[idx]), map_location=self.device)

        map_data = dpt['gridmap_data'][self.fidxs]
        _mean = self.feature_mean[self.fidxs].view(-1, 1, 1)
        _std = self.feature_std[self.fidxs].view(-1, 1, 1)
        map_data_norm = (map_data - _mean) / _std
        map_data_clip = map_data_norm.clip(-10., 10.)
        fmask = ~torch.isfinite(map_data_clip)
        map_data_clip[fmask] = 10.

        if fmask.sum() > 0:
            print('found NaN in dataset!')

        return {
            'traj': dpt['traj'],
            'steer': dpt['steer'],
            'image': dpt['image'],
            'map_features': map_data_clip,
            'metadata': dpt['gridmap_metadata']
        }

    def to(self, device):
        self.device = device
        self.feature_mean = self.feature_mean.to(device)
        self.feature_std = self.feature_std.to(device)
        return self
