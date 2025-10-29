import torch
from torch import nn

import torch_scatter

# from tartandriver_perception_infra.networks.building_blocks.mlp import MLP
from tartandriver_perception_infra.networks.building_blocks.resnet import ResNet
from tartandriver_perception_infra.networks.building_blocks.lss import LSS
from tartandriver_perception_infra.networks.building_blocks.voxel_recolor import VoxelRecolor
from tartandriver_perception_infra.networks.util import get_rays_from_camera_matrix

from tartandriver_utils.geometry_utils import pose_to_htm

from maxent_irl_maps.utils import compute_map_mean_entropy

class BEVToCostSpeed(nn.Module):
    def __init__(
        self,
        in_channels,
        resnet_params,
        cost_params,
        speed_params,
        bev_normalizations,
        device="cpu",
    ):
        super(BEVToCostSpeed, self).__init__()
        self.device = device
        self.in_channels = in_channels

        # self.bev_normalizations = bev_normalizations
        self.register_buffer('bev_normalizations_mean', bev_normalizations['mean'])
        self.register_buffer('bev_normalizations_std', bev_normalizations['std'])

        self.setup_resnet(resnet_params)
        self.setup_cost_head(cost_params)
        self.setup_speed_head(speed_params)

    def forward(self, x, return_mean_entropy=True):
        bev_features = x["bev_data"]["data"]
        bev_features_norm = self.normalize_bev_features(bev_features)

        features = self.resnet.forward(bev_features_norm)

        res = self.run_cost_speed_heads(features, return_mean_entropy)

        return res

    def run_cost_speed_heads(self, features, return_mean_entropy):
        cost_logits = self.cost_head(features)
        speed_logits = self.speed_head(features)

        res = {}
        if self.cost_type == 'Categorical':
            res['cost_logits'] = cost_logits
            if return_mean_entropy:
                res["costmap"], res["costmap_entropy"] = compute_map_mean_entropy(cost_logits, self.cost_bins)
        elif self.cost_type == 'Continuous':
            if return_mean_entropy:
                res["costmap"] = cost_logits
                res["costmap_entropy"] = torch.zeros_like(res["costmap"])

        if self.speed_type == 'Categorical':
            res['speed_logits'] = speed_logits
            if return_mean_entropy:
                res["speedmap"], res["speedmap_entropy"] = compute_map_mean_entropy(speed_logits, self.speed_bins)
        elif self.speed_type == 'Continuous':
            import pdb;pdb.set_trace()

        return res
    
    def normalize_bev_features(self, x):
        _mean = self.bev_normalizations_mean.view(-1, 1, 1)
        _std = self.bev_normalizations_std.view(-1, 1, 1)

        return ((x - _mean) / _std).clip(-10., 10.)

    def setup_resnet(self, params):
        hidden_channels = params['hidden_channels']
        _params = {k:v for k,v in params.items() if k != 'hidden_channels'}
        self.resnet = ResNet(
            in_channels = self.in_channels,
            out_channels = hidden_channels[-1],
            hidden_channels = hidden_channels[:-1],
            device = self.device,
            **_params
        )

    def setup_cost_head(self, params):
        self.cost_type = params['type']
        net_params = params['net_params']
        output_params = params['output_params']

        if params['type'] == 'Categorical':
            self.cost_nbins = output_params['nbins']
            self.min_cost, self.max_cost = output_params['bounds']
            self.cost_bins = torch.linspace(self.min_cost, self.max_cost, self.cost_nbins+1, device=self.device)
            self.cost_head = ResNet(
                in_channels = self.resnet.channel_sizes[-1],
                out_channels = self.cost_nbins,
                device = self.device,
                **net_params,
            )

        elif params['type'] == 'Continuous':
            self.cost_head = ResNet(
                in_channels = self.resnet.channel_sizes[-1],
                out_channels = 1,
                device = self.device,
                **net_params
            )

    def setup_speed_head(self, params):
        self.speed_type = params['type']
        net_params = params['net_params']
        output_params = params['output_params']

        if params['type'] == 'Categorical':
            self.speed_nbins = output_params['nbins']
            self.min_speed, self.max_speed = output_params['bounds']
            self.speed_bins = torch.linspace(self.min_speed, self.max_speed, self.speed_nbins+1, device=self.device)
            self.speed_head = ResNet(
                in_channels = self.resnet.channel_sizes[-1],
                out_channels = self.speed_nbins,
                device = self.device,
                **net_params,
            )

        elif params['type'] == 'Continuous':
            pass

    def to(self, device):
        self.device = device
        self.resnet = self.resnet.to(self.device)
        self.cost_head = self.cost_head.to(self.device)
        self.speed_head = self.speed_head.to(self.device)
        
        self.bev_normalizations_mean = self.bev_normalizations_mean.to(self.device)
        self.bev_normalizations_std = self.bev_normalizations_std.to(self.device)

        return self

class FromPretrain(nn.Module):
    """
    slap on some prediction heads to a pretrained voxel net lol
    """
    def __init__(self, pretrain_fp, cost_params, speed_params, device='cpu'):
        super(FromPretrain, self).__init__()
        self.device = device
        self.backbone = torch.load(pretrain_fp, weights_only=False, map_location=device).network
    
        self.setup_cost_head(cost_params)
        self.setup_speed_head(speed_params)

    def forward(self, x, return_mean_entropy=True):
        voxel_data = self.preproc_voxel(x)

        with torch.no_grad():
            backbone_res = self.backbone.forward({'voxel_input': voxel_data}, return_features=True)
            bev_features = backbone_res['features']

        res = self.run_cost_speed_heads(bev_features, return_mean_entropy)

        return res

    def run_cost_speed_heads(self, features, return_mean_entropy):
        cost_logits = self.cost_head(features)
        speed_logits = self.speed_head(features)

        res = {}
        if self.cost_type == 'Categorical':
            res['cost_logits'] = cost_logits
            if return_mean_entropy:
                res["costmap"], res["costmap_entropy"] = compute_map_mean_entropy(cost_logits, self.cost_bins)
        elif self.cost_type == 'Continuous':
            if return_mean_entropy:
                res["costmap"] = cost_logits
                res["costmap_entropy"] = torch.zeros_like(res["costmap"])

        if self.speed_type == 'Categorical':
            res['speed_logits'] = speed_logits
            if return_mean_entropy:
                res["speedmap"], res["speedmap_entropy"] = compute_map_mean_entropy(speed_logits, self.speed_bins)
        elif self.speed_type == 'Continuous':
            import pdb;pdb.set_trace()

        return res
    
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

    def setup_cost_head(self, params):
        self.cost_type = params['type']
        net_params = params['net_params']
        output_params = params['output_params']

        if params['type'] == 'Categorical':
            self.cost_nbins = output_params['nbins']
            self.min_cost, self.max_cost = output_params['bounds']
            self.cost_bins = torch.linspace(self.min_cost, self.max_cost, self.cost_nbins+1, device=self.device)
            self.cost_head = ResNet(
                in_channels = self.backbone.unet.outsize[0],
                out_channels = self.cost_nbins,
                device = self.device,
                **net_params,
            )

        elif params['type'] == 'Continuous':
            self.cost_head = ResNet(
                in_channels = self.backbone.unet.outsize[0],
                out_channels = 1,
                device = self.device,
                **net_params
            )

    def setup_speed_head(self, params):
        self.speed_type = params['type']
        net_params = params['net_params']
        output_params = params['output_params']

        if params['type'] == 'Categorical':
            self.speed_nbins = output_params['nbins']
            self.min_speed, self.max_speed = output_params['bounds']
            self.speed_bins = torch.linspace(self.min_speed, self.max_speed, self.speed_nbins+1, device=self.device)
            self.speed_head = ResNet(
                in_channels = self.backbone.unet.outsize[0],
                out_channels = self.speed_nbins,
                device = self.device,
                **net_params,
            )

        elif params['type'] == 'Continuous':
            pass


    def to(self, device):
        self.device = device
        self.backbone = self.backbone.to(device)
        self.cost_head = self.cost_head.to(self.device)
        self.speed_head = self.speed_head.to(self.device)

        return self

class VoxelRecolorBEVToCostSpeed(BEVToCostSpeed):
    """
    Same as the basic BEV outputs, but on the input side, perform a feature extraction
     and recolor on the voxel grid, then terrain-aware BEV-splat it
    """
    def __init__(
        self,
        in_channels,
        image_insize,
        resnet_params,
        cost_params,
        speed_params,
        bev_normalizations,
        recolor_params,
        terrain_layer='terrain',
        bev_layers=['num_voxels', 'slope', 'diff', 'min_elevation_filtered_inflated_mask'],
        overhang=2.0,
        device="cpu",
    ):
        self.image_insize = image_insize
        self.terrain_layer = terrain_layer
        self.bev_layers = bev_layers
        self.overhang = overhang

        super(VoxelRecolorBEVToCostSpeed, self).__init__(
            in_channels,
            resnet_params,
            cost_params,
            speed_params,
            bev_normalizations,
            device
        )

        self.setup_voxel_recolor(recolor_params)

        ## need to re-setup resnet with new input features
        del(self.resnet)
        self.resetup_resnet(resnet_params)
    
    def forward(self, x, return_mean_entropy=True):
        bev_metadata = x['bev_data']['metadata']
        voxel_metadata = x['coord_voxel_data']['metadata']

        assert torch.allclose(bev_metadata.origin, voxel_metadata.origin[:, :2])
        assert torch.allclose(bev_metadata.length, voxel_metadata.length[:, :2])
        assert torch.allclose(bev_metadata.resolution, voxel_metadata.resolution[:, :2])

        bev_features = x["bev_data"]["data"]
        bev_features_norm = self.normalize_bev_features(bev_features)

        ## voxel recolor here
        voxel_recolor_data = self.voxel_recolor.forward(x)

        if voxel_recolor_data is None:
            return None

        ## batch-splat terrain features
        terrain_idx = x['bev_data']['feature_keys'].index(self.terrain_layer)
        terrain = bev_features[:, terrain_idx] #[BxWxH]

        splat_voxel_features = self.terrain_splat_voxel_features(voxel_recolor_data, terrain)

        bev_feat_idxs = [x['bev_data']['feature_keys'].index(k) for k in self.bev_layers]
        bev_inp_features = bev_features_norm[:, bev_feat_idxs]

        inp_features = torch.cat([bev_inp_features, splat_voxel_features], dim=1)

        features = self.resnet.forward(inp_features)

        res = self.run_cost_speed_heads(features, return_mean_entropy)

        return res

    def terrain_splat_voxel_features(self, voxel_data, terrain):
        """
        Args:
            voxel_data: [VxN] SpConv tensor
            terrain: [BxWxH] terrain map (assumed to have same metadata as voxels)
        """
        _voxels = voxel_data['data']

        _voxel_features = _voxels.features[:, :self.voxel_recolor.out_channels]
        _voxel_idxs = _voxels.indices
        _feat_mask = _voxels.features[:, voxel_data['feature_keys'].index('feature_mask')] > 0.5

        _voxel_metadata = voxel_data['metadata']

        _terrain_flat = terrain.flatten()

        _bis, _xis, _yis, _zis = _voxel_idxs.long().T

        B = _voxels.batch_size
        nx, ny, nz = _voxels.spatial_shape
        _os = _voxel_metadata.origin[_bis]
        _rs = _voxel_metadata.resolution[_bis]

        _voxel_elevs = (_zis * _rs[:, 2]) + _os[:, 2]
        _raster_idxs = (_bis*nx*ny) + (_xis*ny) + _yis
        _terrain_cmp_elev = _terrain_flat[_raster_idxs]

        _overhang_mask = (_voxel_elevs - _terrain_cmp_elev) < self.overhang

        full_mask = _feat_mask & _overhang_mask

        feats_sum = torch_scatter.scatter(
            src = _voxel_features[full_mask],
            index = _raster_idxs[full_mask],
            dim_size = B*nx*ny,
            reduce = 'sum',
            dim=0
        )

        feats_cnt = torch_scatter.scatter(
            src = torch.ones(full_mask.sum(), device=self.device),
            index = _raster_idxs[full_mask],
            dim_size = B*nx*ny,
            reduce = 'sum'
        ) + 1e-8

        feats = feats_sum / feats_cnt.unsqueeze(-1)

        bev_feats = feats.reshape(B, nx, ny, -1).permute(0,3,1,2)

        # ##debug viz
        # import open3d as o3d
        # from physics_atv_visual_mapping.utils import normalize_dino
        # from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
        # from tartandriver_utils.o3d_viz_utils import make_bev_mesh
        # for i in range(_voxels.batch_size):
        #     bmask = _bis == i
        #     _metadata = LocalMapperMetadata(
        #         origin = _voxel_metadata.origin[i],
        #         length = _voxel_metadata.length[i],
        #         resolution = _voxel_metadata.resolution[i],
        #     )

        #     bev_metadata = LocalMapperMetadata(
        #         origin = _voxel_metadata.origin[i, :2],
        #         length = _voxel_metadata.length[i, :2],
        #         resolution = _voxel_metadata.resolution[i, :2],
        #     )

        #     mask = bmask & _overhang_mask & _feat_mask
        #     _pts = _voxel_idxs[mask][:, 1:] * _metadata.resolution.view(1, 3) + _metadata.origin.view(1, 3)
        #     _colors = normalize_dino(_voxel_features[mask])

        #     pc = o3d.geometry.PointCloud()
        #     pc.points = o3d.utility.Vector3dVector(_pts.detach().cpu().numpy())
        #     pc.colors = o3d.utility.Vector3dVector(_colors.detach().cpu().numpy())

        #     _bev_feats = normalize_dino(bev_feats[i].permute(1,2,0)).detach()
        #     _bev_cnt = feats_cnt.reshape(B,nx,ny)[i]
        #     _terrain = terrain[i]

        #     bev_mesh = make_bev_mesh(bev_metadata, _terrain, _bev_cnt > 0, _bev_feats)

        #     o3d.visualization.draw_geometries([pc, bev_mesh])

        return bev_feats

    def setup_voxel_recolor(self, params):
        self.voxel_recolor = VoxelRecolor(
            image_insize=self.image_insize,
            device=self.device,
            # return_coord_data=False,
            **params
        )

    def resetup_resnet(self, params):
        """
        ResNet insize changes because it consumes lss and bev
        """
        hidden_channels = params['hidden_channels']
        _params = {k:v for k,v in params.items() if k != 'hidden_channels'}
        self.resnet = ResNet(
            in_channels = len(self.bev_layers) + self.voxel_recolor.out_channels,
            out_channels = hidden_channels[-1],
            hidden_channels = hidden_channels[:-1],
            device = self.device,
            **_params
        )

class BEVLSSToCostSpeed(BEVToCostSpeed):
    """
    Same as the basic BEV network, but also add in a
    Lift Splat Shoot head
    """
    def __init__(
        self,
        in_channels,
        image_insize,
        resnet_params,
        cost_params,
        speed_params,
        lss_params,
        bev_normalizations,
        device="cpu",
    ):    
        self.image_insize = image_insize
        super(BEVLSSToCostSpeed, self).__init__(
            in_channels,
            resnet_params,
            cost_params,
            speed_params,
            bev_normalizations,
            device
        )

        self.setup_lss(lss_params)

        ## need to re-setup resnet with lss input features
        del(self.resnet)
        self.resetup_resnet(resnet_params)

    def resetup_resnet(self, params):
        """
        ResNet insize changes because it consumes lss and bev
        """
        hidden_channels = params['hidden_channels']
        _params = {k:v for k,v in params.items() if k != 'hidden_channels'}
        self.resnet = ResNet(
            in_channels = self.in_channels + self.lss.out_channels,
            out_channels = hidden_channels[-1],
            hidden_channels = hidden_channels[:-1],
            device = self.device,
            **_params
        )

    def forward(self, x, return_mean_entropy=True):
        bev_features = x["bev_data"]["data"]
        bev_metadata = x["bev_data"]["metadata"]
        bev_features_norm = self.normalize_bev_features(bev_features)

        #unsqueeze for now for single-cam (TODO need to use a camlist arg)
        images = x["feature_image"]["data"].unsqueeze(1)

        #assume all intrinsics the same for now
        intrinsics = x["feature_image_intrinsics"]["data"][0]

        pose_H = x["tf_odom_to_cam"]["data"]

        max_depth = torch.linalg.norm(x["bev_data"]["metadata"].length[0]) / 2.
        with torch.no_grad():
            #TODO unhack the unsqueeze when we switch to camlist
            cam_pts = self.sample_camera_frustums(
                pose=pose_H,
                image=images,
                intrinsics=intrinsics,
                n_bins=self.lss.n_depth_bins,
                max_depth=max_depth
            ).unsqueeze(1)

        _metadata = torch.stack([
            bev_metadata.origin,
            bev_metadata.length,
            bev_metadata.resolution
        ], dim=1)

        lss_features = self.lss.forward(images, cam_pts, _metadata).permute(0, 3, 1, 2)

        features = torch.cat([bev_features_norm, lss_features], dim=1)

        #lss viz debug
        # import matplotlib.pyplot as plt
        # from tartandriver_utils.o3d_viz_utils import normalize_dino
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(normalize_dino(lss_features[0, :3].permute(1,2,0)).cpu().numpy())
        # axs[1].imshow(normalize_dino(bev_features_norm[0, :3].permute(1,2,0)).cpu().numpy())
        # plt.show()

        features = self.resnet.forward(features)
        res = self.run_cost_speed_heads(features, return_mean_entropy)

        return res
    
    def sample_camera_frustums(self, pose, image, intrinsics, n_bins, max_depth):
        B, _, _, nx, ny = image.shape

        #TODO batch this
        rays = get_rays_from_camera_matrix(intrinsics, nx, ny)
        rays /= torch.linalg.norm(rays, dim=-1, keepdims=True)

        depth_bins = torch.linspace(0., max_depth, n_bins+1, device=self.device)[1:]

        frustum = rays.reshape(nx, ny, 1, 3) * depth_bins.reshape(1, 1, n_bins, 1)

        #[BxWxHxDx3]
        frustums = frustum.unsqueeze(0).tile(B, 1, 1, 1, 1)

        frustums = self.transform_frustums(frustums, pose)

        return frustums

    def transform_frustums(self, frustums, poses):
        """
        Apply transform E to all points in the frustum
        """
        B = poses.shape[0]

        _pts = torch.cat([
            frustums,
            torch.ones_like(frustums[..., [0]])
        ], dim=-1).unsqueeze(-1) #[BxWxHxDx4x1]

        _tf = poses.view(B,1,1,1,4,4)

        _tf_pts = _tf @ _pts

        return _tf_pts[:, :, :, :, :3, 0]

    def setup_lss(self, params):
        self.lss = LSS(
            image_insize = self.image_insize,
            device = self.device,
            **params
        )

"""
LSS debug viz
"""

# if True:
#     import open3d as o3d
#     import numpy as np
#     for bi in range(pose_H.shape[0]):
#         _pose = pose_to_htm(x["odometry"]["data"][bi, 0])
#         _campose = x["tf_odom_to_cam"]["data"][bi]

#         _pts = cam_pts[bi]
#         img = images[bi][0].permute(1,2,0)

#         ax = o3d.geometry.TriangleMesh.create_coordinate_frame()
#         big = np.eye(4) * 5
#         big[-1, -1] = 1.
#         ax = ax.transform(big)

#         ax2 = o3d.geometry.TriangleMesh.create_coordinate_frame()
#         ax2 = ax2.transform(_pose.cpu().numpy())

#         ax3 = o3d.geometry.TriangleMesh.create_coordinate_frame()
#         ax3 = ax3.transform(_campose.cpu().numpy())

#         pcs = []
#         for di in range(_pts.shape[-2]):
#             subpts = _pts[:, :, di, :]
#             pc = o3d.geometry.PointCloud()
#             pc.points = o3d.utility.Vector3dVector(subpts.reshape(-1, 3).cpu().numpy())
#             pc.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3).cpu().numpy())
#             pcs.append(pc)

#         o3d.visualization.draw_geometries([ax, ax2, ax3] + pcs)

#         import pdb;pdb.set_trace()