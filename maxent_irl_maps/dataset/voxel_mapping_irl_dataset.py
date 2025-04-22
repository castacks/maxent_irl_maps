import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import os
import tqdm
import matplotlib.pyplot as plt

from ros_torch_converter.datatypes.pointcloud import PointCloudTorch
from ros_torch_converter.datatypes.image import ImageTorch
from ros_torch_converter.datatypes.rb_state import OdomRBStateTorch
from ros_torch_converter.datatypes.float import Float32Torch

class VoxelMappingIRLDataset:
    """
    Unlike the MaxEntIRLDataset, this one will give raw perception as inputs

    Idk if this is the right call, but collating this stuff is weird and kinda pointless,
        since we will proc one at a time anyways. Thus I am choosing to not make this a 
        torch.utils.Dataset (meaning we'll have to batch/sample ourselves).
    """

    def __init__(self, root_fp, H=75, sensor_frames=np.arange(-150, 0, 5), device="cpu"):
        """
        Args:
            root_fp: The root dir of the kitti-formatted dataset. Assumed to be shallow
            H: number of future timesteps of traj to use
            sensor_frames: The relative timestamps of the sensor frames to use
        """
        self.root_fp = root_fp
        self.run_dirs = os.listdir(self.root_fp)
        self.sensor_frames = sensor_frames
        self.H = H
        self.device = device

        self.setup_idxlist()

    def __len__(self):
        return self.idxlist.shape[0]

    def __getitem__(self, i):
        rdir = self.run_dirs[self.idxlist[i, 0]]
        idx = self.idxlist[i, 1]
        rfp = os.path.join(self.root_fp, rdir)

        sensor_idxs = idx + self.sensor_frames
        state_idxs = idx + np.arange(self.H + 1)

        res = {
            'perception': {
                'pointcloud': [],
                'image': [],
                'state': [],
            },
            'supervision': {
                'state': [],
                'steer': []
            }
        }

        for sidx in sensor_idxs:
            pc = PointCloudTorch.from_kitti(os.path.join(rfp, 'super_odometry_pc'), sidx, device=self.device)
            img = ImageTorch.from_kitti(os.path.join(rfp, 'image_left_color'), sidx, device=self.device)
            state = OdomRBStateTorch.from_kitti(os.path.join(rfp, 'odom'), sidx, device=self.device)

            res['perception']['pointcloud'].append(pc)
            res['perception']['image'].append(img)
            res['perception']['state'].append(state)

        for sidx in state_idxs:
            state = OdomRBStateTorch.from_kitti(os.path.join(rfp, 'odom'), sidx, device=self.device)
            steer = Float32Torch.from_kitti(os.path.join(rfp, 'steer_angle'), sidx, device=self.device)

            res['supervision']['state'].append(state)
            res['supervision']['steer'].append(steer)

        return res

    def setup_idxlist(self):
        self.N = 0
        self.run_dir_idxs = np.arange(len(self.run_dirs))
        self.idxlist = []
        for i, rdir in zip(self.run_dir_idxs, self.run_dirs):
            timestamp_fp = os.path.join(self.root_fp, rdir, 'target_timestamps.txt')
            timestamps = np.loadtxt(timestamp_fp)
            idxs = np.arange(timestamps.shape[0])
            idxs_min = idxs + self.sensor_frames[0]
            idxs_max = idxs + self.H

            valid_mask = (idxs_min >= 0) & (idxs_max < timestamps.shape[0])

            valid_idxs = idxs[valid_mask]
            ridx = np.ones_like(valid_idxs) * i

            subidxlist = np.stack([ridx, valid_idxs], axis=-1)
            self.idxlist.append(subidxlist)

        self.idxlist = np.concatenate(self.idxlist, axis=0)

    def to(self, device):
        self.device = device
        return self

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    args = {
        'root_fp': '/media/striest/aaa/2025_irl_kitti',
        'H': 75,
        'sensor_frames': np.arange(-100, 100, 10),
        'device': 'cuda'
    }
    dataset = VoxelMappingIRLDataset(**args)

    print(dataset[50])