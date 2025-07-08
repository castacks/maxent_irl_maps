import torch
import numpy as np

import torch_scatter
import spconv.pytorch as spconv

from maxent_irl_maps.networks.resnet import ResnetCategorical

def voxel_grid_to_spconv(voxel_grid):
    indices = voxel_grid.raster_indices_to_grid_indices(voxel_grid.raster_indices).int()
    indices = torch.cat([torch.zeros_like(indices[:, [0]]), indices], dim=-1)

    features = torch.zeros(indices.shape[0], voxel_grid.features.shape[-1], device=voxel_grid.device)
    features[voxel_grid.feature_mask] = voxel_grid.features

    spatial_shape = voxel_grid.metadata.N.tolist()
    batch_size = 1

    return spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)

class VoxelResnetCategorical(torch.nn.Module):
    """
    Same as the Resnet Categorical, except start w/ voxel grid and 3d conv block
    """
    def __init__(self, in_channels, bev_in_channels, voxel_params, resnet_params, device):
        super(VoxelResnetCategorical, self).__init__()
        self.in_channels = in_channels
        self.bev_in_channels = bev_in_channels
        self.device = device

        self.voxel_net = self.setup_voxel_net(**voxel_params)
        self.bev_net = self.setup_bev_net(resnet_params)

    def predict(self, voxel_grid, bev_grid, return_mean_entropy=False):
        """
        Does all the ros_torch_converter wrapper stuff
        """
        voxel_spconv = voxel_grid_to_spconv(voxel_grid)
        bev_data = bev_grid.data.unsqueeze(0).permute(0,3,1,2)

        res = self.forward(voxel_spconv, bev_data, return_mean_entropy)

        return res

    def forward(self, voxel_spconv, bev_data, return_mean_entropy):
        voxel_feats = self.voxel_net.forward(voxel_spconv)

        #store as flatten until after the scatter op
        nb = voxel_spconv.batch_size
        nx = voxel_spconv.spatial_shape[0]
        ny = voxel_spconv.spatial_shape[1]

        dimsize = nb*nx*ny

        _n1 = nx*ny
        _n2 = ny

        scatter_idxs = voxel_feats.indices[:, 0] * _n1 + voxel_feats.indices[:, 1] * _n2 + voxel_feats.indices[:, 2]

        bev_feats = torch_scatter.scatter(src=voxel_feats.features, index=scatter_idxs.long(), dim=0, dim_size=dimsize, reduce='max')

        #[B x C x W x H]
        bev_feats = bev_feats.view(nb, nx, ny, -1).permute(0,3,1,2)

        bev_feats = torch.cat([bev_feats, bev_data], dim=1)

        bev_preds = self.bev_net.forward(bev_feats, return_mean_entropy)

        #[B x W x H x C]
        return bev_preds

    def setup_bev_net(self, resnet_params):
        return ResnetCategorical(**resnet_params, in_channels=self.voxel_net[-1].out_channels + self.bev_in_channels).to(self.device)

    def setup_voxel_net(self, hidden_channels, hidden_activation, hidden_kernel_size):
        if hidden_activation == "tanh":
            hidden_activation = torch.nn.Tanh
        elif hidden_activation == "relu":
            hidden_activation = torch.nn.ReLU

        layers = []
        hc = [self.in_channels] + hidden_channels

        for i in range(len(hc) - 2):
            layers.append(spconv.SubMConv3d(
                hc[i],
                hc[i+1],
                hidden_kernel_size
            ))
            layers.append(hidden_activation())

        layers.append(spconv.SubMConv3d(
            hc[-2],
            hc[-1],
            hidden_kernel_size
        ))

        return spconv.SparseSequential(*layers).to(self.device)