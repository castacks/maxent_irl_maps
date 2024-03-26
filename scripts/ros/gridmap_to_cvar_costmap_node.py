#! /usr/bin/python3

import os
import rospy
import torch
import numpy as np

from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32
from nav_msgs.msg import OccupancyGrid, Odometry
from grid_map_msgs.msg import GridMap

from rosbag_to_dataset.dtypes.gridmap import GridMapConvert

from maxent_irl_costmaps.experiment_management.parse_configs import setup_experiment

class CvarCostmapperNode:
    """
    Node that listens to gridmaps from perception and uses IRL nets to make them into costmaps
    In addition to the baseline costmappers, take in a CVaR value from -1.0 to 1.0 and use an ensemble of costmaps
    """
    def __init__(self, grid_map_topic, cost_map_topic, speed_map_topic, odom_topic, cvar_topic, speedmap_lcb_topic, dataset, network, obstacle_threshold, categorical_speedmaps, vmin, vmax, publish_gridmap, device):
        """
        Args:
            grid_map_topic: the topic to get map features from
            cost_map_topic: The topic to publish costmaps to
            speed_map_topic: The topic to publish speedmaps to
            odom_topic: The topic to get height from 
            cvar_topic: The topic to listen to CVaR from
            speedmap_lcb_topic: The topic to listen to speedmap lcb from
            dataset: The dataset that the network was trained on. (Need to get feature mean/var)
            obstacle_threshold: value above which cells are treated as infinite-cost obstacles
            categorical_speedmaps: whether speedmaps are given as a categorical
            vmin: For viz, this quantile of the costmap is set as 0 cost
            vmax: For viz, this quantile of the costmaps is set as 1 cost
            publish_gridmap: If true, publish costmap as a GridMap, else OccupancyGrid
            network: the network to produce costmaps.
        """
        self.feature_keys = dataset.feature_keys
        self.feature_mean = dataset.feature_mean[dataset.fidxs].to(device).view(-1, 1, 1)
        self.feature_std = dataset.feature_std[dataset.fidxs].to(device).view(-1, 1, 1)
#        self.map_metadata = dataset.metadata
        self.network = network.to(device)
        self.obstacle_threshold = obstacle_threshold
        self.categorical_speedmaps = categorical_speedmaps

        self.vmin = vmin
        self.vmax = vmax
        self.cvar = 0.
        self.speedmap_lcb = 0.5

        self.publish_gridmap = publish_gridmap

        self.device = device

        self.current_height = 0.
        self.current_speed = 0.

        #note that there are some map features that are state-dependent (like speed)
        #we won't pass them into the gridmap convert, and generate them at the end
        ego_features = ['ego_speed']
        self.gridmap_feature_keys = []
        self.gridmap_feature_idxs = []
        self.ego_feature_keys = []
        self.ego_feature_idxs = []
        for i, fk in enumerate(self.feature_keys):
            if fk in ego_features:
                self.ego_feature_keys.append(fk)
                self.ego_feature_idxs.append(i)
            else:
                self.gridmap_feature_keys.append(fk)
                self.gridmap_feature_idxs.append(i)

        #we will set the output resolution dynamically
        self.grid_map_cvt = GridMapConvert(channels=self.gridmap_feature_keys, size=[1, 1])

        self.grid_map_sub = rospy.Subscriber(grid_map_topic, GridMap, self.handle_grid_map, queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.handle_odom, queue_size=1)
        self.cvar_sub = rospy.Subscriber(cvar_topic, Float32, self.handle_cvar, queue_size=1)
        self.speedmap_lcb_sub = rospy.Subscriber(speedmap_lcb_topic, Float32, self.handle_lcb, queue_size=1)

        self.cost_map_viz_pub = rospy.Publisher(cost_map_topic + '/viz', GridMap, queue_size=1)

        if publish_gridmap:
            self.cost_map_pub = rospy.Publisher(cost_map_topic, GridMap, queue_size=1)
            self.speed_map_pub = rospy.Publisher(speed_map_topic, GridMap, queue_size=1)
        else:
            self.cost_map_pub = rospy.Publisher(cost_map_topic, OccupancyGrid, queue_size=1)
            self.speed_map_pub = rospy.Publisher(speed_map_topic, OccupancyGrid, queue_size=1)

    def handle_odom(self, msg):
        self.current_height = msg.pose.pose.position.z
        self.current_speed = msg.twist.twist.linear.x

    def handle_cvar(self, msg):
        if msg.data > -1. and msg.data < 1.:
            self.cvar = msg.data
        else:
            rospy.logerr("CVaR expected to be (-1, 1) exclusive, got {}".format(msg.data))

    def handle_lcb(self, msg):
        if msg.data > 0. and msg.data < 1.:
            self.speedmap_lcb = msg.data
        else:
            rospy.logerr("Speed quantile expected to be (0, 1), got {}".format(msg.data))

    def create_ego_features(self, keys, nx, ny):
        res = []
        for k in keys:
            if k == 'ego_speed':
                res.append(torch.zeros(nx, ny, device=self.device) * self.current_speed)

        return torch.stack(res, dim=0)

    def compute_map_cvar(self, maps, cvar):
        if self.cvar < 0.:
            map_q = torch.quantile(maps, 1.+cvar, dim=0)
            mask = (maps <= map_q.view(1, *map_q.shape))
        else:
            map_q = torch.quantile(maps, self.cvar, dim=0)
            mask = (maps >= map_q.view(1, *map_q.shape))

        cvar_map = (maps * mask).sum(dim=0) / mask.sum(dim=0)
        return cvar_map

    def handle_grid_map(self, msg):
        t1 = rospy.Time.now().to_sec()
        nx = int(msg.info.length_x / msg.info.resolution)
        ny = int(msg.info.length_y / msg.info.resolution)
        self.grid_map_cvt.size = [nx, ny]
        gridmap = self.grid_map_cvt.ros_to_numpy(msg)

        rospy.loginfo_throttle(1.0, "output shape: {}".format(gridmap['data'].shape))

        ## create geometric features ##
        geometric_map_feats = torch.from_numpy(gridmap['data']).float().to(self.device)
        for k in self.gridmap_feature_keys:
            if k in ['height_low', 'height_mean', 'height_high', 'height_max', 'terrain']:
                idx = self.feature_keys.index(k)
                geometric_map_feats[idx] -= self.current_height

        ## create ego features ##
        map_feats = torch.zeros(len(self.feature_keys), nx, ny, device=self.device)
        map_feats[self.gridmap_feature_idxs] = geometric_map_feats

        if len(self.ego_feature_idxs) > 0:
            ego_map_feats = self.create_ego_features(self.ego_feature_keys, nx, ny)
            map_feats[self.ego_feature_idxs] = ego_map_feats

        map_feats[~torch.isfinite(map_feats)] = 0.
        map_feats[map_feats.abs() > 100.] = 0.

        map_feats_norm = (map_feats - self.feature_mean.view(-1, 1, 1)) / self.feature_std.view(-1, 1, 1)

        #this part is different for CVaR
        with torch.no_grad():
            res = self.network.ensemble_forward(map_feats_norm.view(1, *map_feats_norm.shape))
            costmaps = res['costmap'][0, :, 0]

            if self.categorical_speedmaps:
                speedmap_probs = res['speedmap'][0].mean(dim=0).softmax(dim=0)
                speedmap_cdf = torch.cumsum(speedmap_probs, dim=0)
                speedmap = self.compute_speedmap_quantile(speedmap_cdf, self.network.speed_bins.to(self.device), self.speedmap_lcb)
            else:
                speedmap = (res['speedmap'].loc[0, :] + self.speedmap_lcb * res['speedmap'].scale[0, :]).mean(dim=0)

        cvar_costmap = self.compute_map_cvar(costmaps, self.cvar)

        #experiment w/ normalizing
        rospy.loginfo_throttle(1.0, "cost min = {:.4f}, max = {:.4f}".format(cvar_costmap.min(), cvar_costmap.max()))
        rospy.loginfo_throttle(1.0, "speed min = {:.4f}, max = {:.4f}".format(speedmap.min(), speedmap.max()))

#        vmin_val = torch.quantile(cvar_costmap, self.vmin)
#        vmax_val = torch.quantile(cvar_costmap, self.vmax)
        vmin_val = -2.
        vmax_val = 5.

        #convert to occgrid scaling. Also apply the obstacle threshold
#        mask = (cvar_costmap >= self.obstacle_threshold).cpu().numpy()

        costmap_occgrid = cvar_costmap.cpu().numpy()
        costmap_viz = ((cvar_costmap - vmin_val) / (vmax_val - vmin_val)).clip(0., 1.)

        costmap_color = torch.stack([
            torch.ones_like(costmap_viz),
            1.-costmap_viz,
            torch.zeros_like(costmap_viz)
        ], dim=-1) #[W x H x 3]
        costmap_color[costmap_viz > 0.9] = 0.
        costmap_color[costmap_viz < 0.1] = 0.8
        costmap_color = (255.*costmap_color).cpu().numpy().astype(np.int32)
        costmap_color = np.moveaxis(costmap_color, -1, 0) #[3 x W x H]
        costmap_color = costmap_color[0]*(2**16) + costmap_color[1]*(2**8) + costmap_color[2]
        costmap_color = costmap_color.view(dtype=np.float32)

        costmap_color_msg = self.costmap_to_gridmap(costmap_color, msg, costmap_layer='costmap_color')
        self.cost_map_viz_pub.publish(costmap_color_msg)

        costmap_msg = self.costmap_to_gridmap(costmap_occgrid, msg) if self.publish_gridmap else self.costmap_to_occgrid(costmap_occgrid.astype(np.int32), msg)
        self.cost_map_pub.publish(costmap_msg)

        speedmap_msg = self.costmap_to_gridmap(speedmap.cpu().numpy(), msg, costmap_layer='speedmap')
        self.speed_map_pub.publish(speedmap_msg)

        t2 = rospy.Time.now().to_sec()
        rospy.loginfo_throttle(1.0, "inference time: {:.4f}s".format(t2-t1))

        rospy.loginfo_throttle(1.0, "CVaR = {:.2f}".format(self.cvar))

    def costmap_to_gridmap(self, costmap, msg, costmap_layer='costmap'):
        """
        convert costmap into gridmap msg

        Args:
            costmap: The data to load into the gridmap
            msg: The input msg to extrach metadata from
            costmap: The name of the layer to get costmap from
        """
        costmap_msg = GridMap()
        costmap_msg.info = msg.info
        costmap_msg.layers = [costmap_layer]

        costmap_layer_msg = Float32MultiArray()
        costmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=costmap.shape[0],
                stride=costmap.shape[0]
            )
        )
        costmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=costmap.shape[0],
                stride=costmap.shape[0] * costmap.shape[1]
            )
        )

        costmap_layer_msg.data = costmap[::-1, ::-1].flatten()
        costmap_msg.data.append(costmap_layer_msg)
        return costmap_msg

    def costmap_to_occgrid(self, costmap, msg):
        """
        convert costmap into occupancy grid msg
        
        Args:
            costmap: The data to load into the occgrid
            msg: The input msg to extract metadata from
        """
        costmap_msg = OccupancyGrid()
        costmap_msg.header.stamp = msg.info.header.stamp
        costmap_msg.header.frame_id = msg.info.header.frame_id
        costmap_msg.info.resolution = msg.info.resolution
        costmap_msg.info.width = int(msg.info.length_x / msg.info.resolution)
        costmap_msg.info.height = int(msg.info.length_y / msg.info.resolution)
        costmap_msg.info.origin.position.x = msg.info.pose.position.x - msg.info.length_x/2.
        costmap_msg.info.origin.position.y = msg.info.pose.position.y - msg.info.length_y/2.
        costmap_msg.info.origin.position.z = self.current_height
        costmap_msg.data = costmap.flatten()
        return costmap_msg

    def compute_speedmap_quantile(self, speedmap_cdf, speed_bins, q):
        """
        Given a speedmap (parameterized as cdf + bins) and a quantile,
            compute the speed corresponding to that quantile

        Args:
            speedmap_cdf: BxWxH Tensor of speedmap cdf
            speed_bins: B+1 Tensor of bin edges
            q: float containing the quantile 
        """
        B = len(speed_bins)
        #need to stack zeros to front
        _cdf = torch.cat([
            torch.zeros_like(speedmap_cdf[[0]]),
            speedmap_cdf
        ], dim=0)

        _cdf = _cdf.view(B, -1)

        _cdiffs = q - _cdf
        _mask = _cdiffs <= 0

        _qidx = (_cdiffs + 1e10*_mask).argmin(dim=0)

        _cdf_low = _cdf.T[torch.arange(len(_qidx)), _qidx]
        _cdf_high = _cdf.T[torch.arange(len(_qidx)), _qidx+1]

        _k = (q - _cdf_low) / (_cdf_high - _cdf_low)

        _speedmap_low = speed_bins[_qidx]
        _speedmap_high = speed_bins[_qidx+1]

        _speedmap = (1-_k) * _speedmap_low + _k * _speedmap_high

        return _speedmap.reshape(*speedmap_cdf.shape[1:])

if __name__ == '__main__':
    rospy.init_node('costmapper_node')

    model_fp = rospy.get_param('~model_fp')
    grid_map_topic = rospy.get_param('~gridmap_topic')
    cost_map_topic = rospy.get_param('~costmap_topic')
    speed_map_topic = rospy.get_param('~speedmap_topic')

    odom_topic = rospy.get_param('~odom_topic')
    cvar_topic = rospy.get_param('~cvar_topic')
    speedmap_lcb_topic = rospy.get_param("~speedmap_lcb_topic")

    obstacle_threshold = rospy.get_param('~obstacle_threshold') #values above this are treated as obstacle and assigned a special value (127)
    vmin = rospy.get_param('~viz_vmin', 0.2) #values below this quantile are viz-ed as 0
    vmax = rospy.get_param('~viz_vmax', 0.9) #values above this quantile are viz-ed as 1

    publish_gridmap = rospy.get_param('~publish_gridmap')

    device = rospy.get_param('~device', 'cuda')

    param_fp = os.path.join(os.path.split(model_fp)[0], '_params.yaml')
    mppi_irl = setup_experiment(param_fp)['algo']

    mppi_irl.network.load_state_dict(torch.load(model_fp))
    mppi_irl.network.eval()

    costmapper = CvarCostmapperNode(grid_map_topic, cost_map_topic, speed_map_topic, odom_topic, cvar_topic, speedmap_lcb_topic, mppi_irl.expert_dataset, mppi_irl.network, obstacle_threshold, mppi_irl.categorical_speedmaps, vmin, vmax, publish_gridmap, device)

    rospy.spin()
