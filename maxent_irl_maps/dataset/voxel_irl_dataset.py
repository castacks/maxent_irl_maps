import os
import yaml

import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from ros_torch_converter.datatypes.voxel_grid import VoxelGridTorch
from ros_torch_converter.datatypes.image import ImageTorch

from tartandriver_utils.os_utils import is_kitti_dir
from tartandriver_utils.o3d_viz_utils import traj_to_o3d

class VoxelIRLDataset(Dataset):
    """
    Dataset for directly training 3d->2d with IRL

    Note that spconv tensors dont work with the default dataloader lol
    """
    def __init__(
            self,
            root_fp,
            H=75,
            sample_every=20,
            min_avg_speed=1.,
            voxel_dir='voxel_map',
            odom_dir='odom',
            steer_angle_dir='steer_angle',
            image_dir='image_left_color',
            device="cpu"
        ):

        self.H = H
        self.sample_every = sample_every
        self.min_avg_speed = min_avg_speed

        self.root_fp = root_fp
        self.voxel_dir = voxel_dir
        self.odom_dir = odom_dir
        self.steer_angle_dir = steer_angle_dir
        self.image_dir = image_dir

        self.voxel_fps = self.get_voxel_fps()

        self.device = device

        self.feature_keys = ["vfm_{}".format(i) for i in range(self[0]["voxel_grid"].features.shape[-1])]

    def get_voxel_fps(self):
        voxel_fps = []

        for rdir in os.listdir(self.root_fp):
            run_dir = os.path.join(self.root_fp, rdir)
            subdirs = os.listdir(run_dir)
            is_valid_run_dir = is_kitti_dir(run_dir) and all([x in subdirs for x in (self.voxel_dir, self.odom_dir, self.steer_angle_dir, self.image_dir)])
            if is_valid_run_dir:
                print('found valid run dir {}'.format(run_dir))
                voxel_dir = os.path.join(run_dir, self.voxel_dir)
                odom_dir = os.path.join(run_dir, self.odom_dir)
                steer_angle_dir = os.path.join(run_dir, self.steer_angle_dir)

                odom = np.loadtxt(os.path.join(run_dir, self.odom_dir, 'data.txt'))

                valid_voxel_fps = sorted([x for x in os.listdir(voxel_dir) if '.npz' in x])
                valid_voxel_fps = valid_voxel_fps[:-self.H:self.sample_every]
                valid_voxel_idxs = np.array([int(fp[:8]) for fp in valid_voxel_fps])

                odom_idxs = valid_voxel_idxs.reshape(-1, 1) + np.arange(self.H).reshape(1, -1)
                vels = np.linalg.norm(odom[odom_idxs][..., 7:10], axis=-1)

                valid_voxel_idxs = valid_voxel_idxs[vels.mean(axis=-1) > self.min_avg_speed]

                voxel_fps.extend([(rdir, i) for i in valid_voxel_idxs])

        return voxel_fps

    def __len__(self):
        return len(self.voxel_fps)

    def __getitem__(self, idx):
        run_dir, i = self.voxel_fps[idx]

        voxel_map = VoxelGridTorch.from_kitti(os.path.join(self.root_fp, run_dir, self.voxel_dir), i, self.device)
        image = ImageTorch.from_kitti(os.path.join(self.root_fp, run_dir, self.image_dir), i, self.device)

        odom = np.loadtxt(os.path.join(self.root_fp, run_dir, self.odom_dir, 'data.txt'))
        steer = np.loadtxt(os.path.join(self.root_fp, run_dir, self.steer_angle_dir, 'data.txt'))

        sub_odom = odom[i:i+self.H]
        sub_steer = steer[i:i+self.H]

        return {
            'traj': torch.tensor(sub_odom).float().to(self.device),
            'steer': torch.tensor(sub_steer).float().to(self.device),
            'image': image.image,
            'voxel_grid': voxel_map.voxel_grid
        }

    def to(self, device):
        self.device = device
        return self

    def viz_dpt(self, dpt):
        vg_viz = dpt['voxel_grid'].visualize()
        traj_viz = traj_to_o3d(dpt['traj'], color=[1., 0., 0.])

        o3d.visualization.draw_geometries([vg_viz, traj_viz])

if __name__ == '__main__':
    root_fp = '/media/striest/offroad/datasets/irl_kitti/'

    dataset = VoxelIRLDataset(root_fp)
    
    for _ in range(10):
        idx = np.random.randint(len(dataset))

        dataset.viz_dpt(dataset[idx])