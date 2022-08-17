#! /usr/bin/python3

import rospy
import numpy as np
import torch

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from nav_msgs.msg import OccupancyGrid, Odometry
from grid_map_msgs.msg import GridMap

from rosbag_to_dataset.dtypes.gridmap import GridMapConvert

class CostmapperNode:
    """
    Node that listens to gridmaps from perception and uses IRL nets to make them into costmaps
    """
    def __init__(self, grid_map_topic, cost_map_topic, odom_topic, dataset, network, obstacle_threshold, vmin, vmax, publish_gridmap, device):
        """
        Args:
            grid_map_topic: the topic to get map features from
            cost_map_topic: The topic to publish costmaps to
            odom_topic: The topic to get height from 
            dataset: The dataset that the network was trained on. (Need to get feature mean/var)
            obstacle_threshold: value above which cells are treated as infinite-cost obstacles
            vmin: For viz, this quantile of the costmap is set as 0 cost
            vmax: For viz, this quantile of the costmaps is set as 1 cost
            publish_gridmap: If true, publish costmap as a GridMap, else OccupancyGrid
            network: the network to produce costmaps.
        """
        self.feature_keys = dataset.feature_keys
        self.feature_mean = dataset.feature_mean.to(device)
        self.feature_std = dataset.feature_std.to(device)
        self.map_metadata = dataset.metadata
        self.network = network.to(device)
        self.obstacle_threshold = obstacle_threshold

        self.vmin = vmin
        self.vmax = vmax

        self.publish_gridmap = publish_gridmap

        self.device = device

        self.current_height = 0.

        #we will set the output resolution dynamically
        self.grid_map_cvt = GridMapConvert(channels=self.feature_keys, size=[1, 1])

        self.grid_map_sub = rospy.Subscriber(grid_map_topic, GridMap, self.handle_grid_map, queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.handle_odom, queue_size=1)
        self.cost_map_viz_pub = rospy.Publisher(cost_map_topic + '/viz', OccupancyGrid, queue_size=1)

        if publish_gridmap:
            self.cost_map_pub = rospy.Publisher(cost_map_topic, GridMap, queue_size=1)
        else:
            self.cost_map_pub = rospy.Publisher(cost_map_topic, OccupancyGrid, queue_size=1)

    def handle_odom(self, msg):
        self.current_height = msg.pose.pose.position.z

    def handle_grid_map(self, msg):
        t1 = rospy.Time.now().to_sec()
        nx = int(msg.info.length_x / msg.info.resolution)
        ny = int(msg.info.length_y / msg.info.resolution)
        self.grid_map_cvt.size = [nx, ny]
        gridmap = self.grid_map_cvt.ros_to_numpy(msg)

        rospy.loginfo_throttle(1.0, "output shape: {}".format(gridmap['data'].shape))

        map_feats = torch.from_numpy(gridmap['data']).float().to(self.device)
        for k in self.feature_keys:
            if k in ['height_low', 'height_mean', 'height_high', 'height_max', 'terrain']:
                idx = self.feature_keys.index(k)
                map_feats[idx] -= self.current_height

        map_feats[~torch.isfinite(map_feats)] = 0.
        map_feats[map_feats.abs() > 100.] = 0.

        map_feats_norm = (map_feats - self.feature_mean.view(-1, 1, 1)) / self.feature_std.view(-1, 1, 1)
        with torch.no_grad():
            res = self.network.forward(map_feats_norm.view(1, *map_feats_norm.shape))
            costmap = res['costmap'][0, 0]

        #experiment w/ normalizing
        rospy.loginfo_throttle(1.0, "min = {:.4f}, max = {:.4f}".format(costmap.min(), costmap.max()))

        vmin_val = torch.quantile(costmap, self.vmin)
        vmax_val = torch.quantile(costmap, self.vmax)

        #convert to occgrid scaling. Also apply the obstacle threshold
        mask = (costmap >=self.obstacle_threshold).cpu().numpy()
        costmap_viz = (100. * (costmap - vmin_val) / (vmax_val - vmin_val)).clip(0., 100.).long().cpu().numpy()
        costmap_viz[mask] = 100
        costmap_occgrid = (costmap).long().cpu().numpy()
        costmap_occgrid[mask] = 100

        costmap_msg = self.costmap_to_gridmap(costmap_occgrid, msg) if self.publish_gridmap else self.costmap_to_occgrid(costmap_occgrid, msg)
        self.cost_map_pub.publish(costmap_msg)

        costmap_viz_msg = self.costmap_to_occgrid(costmap_viz, msg)
        self.cost_map_viz_pub.publish(costmap_viz_msg)

        t2 = rospy.Time.now().to_sec()
        rospy.loginfo_throttle(1.0, "inference time: {:.4f}s".format(t2-t1))

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

if __name__ == '__main__':
    rospy.init_node('costmapper_node')

    model_fp = rospy.get_param('~model_fp')
    grid_map_topic = rospy.get_param('~gridmap_topic')
    cost_map_topic = rospy.get_param('~costmap_topic')
    odom_topic = rospy.get_param('~odom_topic')

    obstacle_threshold = rospy.get_param('~obstacle_threshold') #values above this are treated as obstacle and assigned a special value (127)
    vmin = rospy.get_param('~viz_vmin', 0.2) #values below this quantile are viz-ed as 0
    vmax = rospy.get_param('~viz_vmax', 0.9) #values above this quantile are viz-ed as 1

    publish_gridmap = rospy.get_param('~publish_gridmap')

    device = rospy.get_param('~device', 'cuda')
    mppi_irl = torch.load(model_fp, map_location=device)
    mppi_irl.network.eval()

    costmapper = CostmapperNode(grid_map_topic, cost_map_topic, odom_topic, mppi_irl.expert_dataset, mppi_irl.network, obstacle_threshold, vmin, vmax, publish_gridmap, device)

    rospy.spin()
