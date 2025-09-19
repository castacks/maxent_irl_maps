import torch
from torch import nn

# from tartandriver_perception_infra.networks.building_blocks.mlp import MLP
from tartandriver_perception_infra.networks.building_blocks.resnet import ResNet
from tartandriver_perception_infra.networks.building_blocks.lss import LSS
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
            import pdb;pdb.set_trace()


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
            pass

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

    def setup_resnet(self, params):
        """
        ResNet insize changes because it consumes lss and bev
        """
        hidden_channels = params['hidden_channels']
        _params = {k:v for k,v in params.items() if k != 'hidden_channels'}
        self.resnet = ResNet(
            in_channels = self.in_channels + self.image_insize[0],
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