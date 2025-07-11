import os
import yaml
import time
import argparse

import torch
import torch_scatter

import numpy as np
import open3d as o3d
import spconv.pytorch as spconv
import matplotlib.pyplot as plt

from ros_torch_converter.datatypes.voxel_grid import VoxelGridTorch

from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.localmapping.bev.bev_localmapper import BEVGrid
from physics_atv_visual_mapping.utils import normalize_dino


from maxent_irl_maps.networks.resnet import ResnetCostmapBlock

def voxel_grid_to_spconv(voxel_grid):
    indices = voxel_grid.raster_indices_to_grid_indices(voxel_grid.raster_indices).int()
    indices = torch.cat([torch.zeros_like(indices[:, [0]]), indices], dim=-1)

    features = torch.zeros(indices.shape[0], voxel_grid.features.shape[-1], device=voxel_grid.device)
    features[voxel_grid.feature_mask] = voxel_grid.features

    spatial_shape = voxel_grid.metadata.N.tolist()
    batch_size = 1

    return spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)

def spconv_to_voxel_grid(spconv_tensor, metadata):
    vg_out = VoxelGrid(metadata=metadata, n_features=spconv_tensor.features.shape[-1], device=metadata.device)
    vg_out.raster_indices = vg_out.grid_indices_to_raster_indices(spconv_tensor.indices[:, 1:].long())
    vg_out.features = spconv_tensor.features
    vg_out.feature_mask = torch.ones(vg_out.raster_indices.shape[0], dtype=torch.bool, device=metadata.device)

    return vg_out

class VoxelToCostNet(torch.nn.Module):
    """
    voxel grid -> bev -> cost
    """
    def __init__(self, voxel_metadata, n_feats, device='cuda'):
        super(VoxelToCostNet, self).__init__()

        self.voxel_metadata = voxel_metadata
        self.n_feats = n_feats
        self.device = device

        # self.voxel_conv_type = spconv.SparseConv3d
        self.voxel_conv_type = spconv.SubMConv3d
        self.voxel_net = self.setup_voxel_net()

        self.cnn = self.setup_cnn()

    def setup_voxel_net(self):
        nn = spconv.SparseSequential(
            self.voxel_conv_type(self.n_feats, 64, 3),
            torch.nn.ReLU(),
            self.voxel_conv_type(64, 64, 3),
            torch.nn.ReLU(),
            self.voxel_conv_type(64, 64, 3),
        ).to(self.device)

        return nn

    def setup_cnn(self):
        cnn = torch.nn.Sequential(
            ResnetCostmapBlock(
                in_channels=64,
                out_channels=64,
                activation=torch.nn.ReLU,
                kernel_size=5,
            ),
            ResnetCostmapBlock(
                in_channels=64,
                out_channels=3,
                activation=torch.nn.ReLU,
                kernel_size=5,
            ),
        ).to(self.device)

        return cnn

    def predict(self, voxel_grid):
        """
        Does all the ros_torch_converter wrapper stuff
        """
        voxel_metadata = voxel_grid.metadata
        voxel_spconv = voxel_grid_to_spconv(voxel_grid)

        res = self.forward(voxel_spconv)

        bev_metadata = LocalMapperMetadata(
            origin=voxel_metadata.origin[:2],
            length=voxel_metadata.length[:2],
            resolution=voxel_metadata.resolution[:2],
        )

        bev_grid = BEVGrid(metadata=bev_metadata, n_features=res.shape[-1], feature_keys=['feat_{}'.format(i) for i in range(res.shape[-1])])
        bev_grid.data = res[0]

        return bev_grid

    # @torch.compile
    def forward(self, voxel_spconv):
        voxel_feats = self.voxel_net.forward(voxel_spconv)

        #store as flatten until after the scatter op
        # bev_feats = torch.zeros(voxel_feats.batch_size*self.voxel_metadata.N[0]*self.voxel_metadata.N[1], voxel_feats.features.shape[-1], device=self.device)
        dimsize = voxel_feats.batch_size*self.voxel_metadata.N[0]*self.voxel_metadata.N[1]

        _n1 = self.voxel_metadata.N[0]*self.voxel_metadata.N[1]
        _n2 = self.voxel_metadata.N[1]

        scatter_idxs = voxel_feats.indices[:, 0] * _n1 + voxel_feats.indices[:, 1] * _n2 + voxel_feats.indices[:, 2]

        bev_feats = torch_scatter.scatter(src=voxel_feats.features, index=scatter_idxs.long(), dim=0, dim_size=dimsize, reduce='max')

        #[B x C x W x H]
        bev_feats = bev_feats.view(voxel_feats.batch_size, self.voxel_metadata.N[0], self.voxel_metadata.N[1], -1).permute(0,3,1,2)

        bev_preds = self.cnn.forward(bev_feats)

        #[B x W x H x C]
        return bev_preds.permute(0,2,3,1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxel_dir', type=str, required=True, help='path to voxel')
    args = parser.parse_args()

    N = int((len(os.listdir(args.voxel_dir)) - 2) / 2)

    for _ in range(10):
        i = np.random.randint(N)

        voxel_map = VoxelGridTorch.from_kitti(args.voxel_dir, i, 'cuda')

        net = VoxelToCostNet(voxel_map.voxel_grid.metadata, voxel_map.voxel_grid.features.shape[-1])

        print(net)
        print("net has {} params".format(sum([x.numel() for x in net.parameters()])))

        res = net.predict(voxel_map.voxel_grid)

        t1 = time.time()
        with torch.no_grad():
            res = net.predict(voxel_map.voxel_grid)
        torch.cuda.synchronize()
        t2 = time.time()

        #viz
        o3d.visualization.draw_geometries([voxel_map.voxel_grid.visualize()])

        plt.title('inference = {:.4f}s'.format(t2-t1))
        plt.imshow(normalize_dino(res.data).cpu().numpy());plt.show()