#! /usr/bin/python3

import rospy
import numpy as np
import torch

from nav_msgs.msg import OccupancyGrid, Odometry
from grid_map_msgs.msg import GridMap

from rosbag_to_dataset.dtypes.gridmap import GridMapConvert

class CostmapperNode:
    """
    Node that listens to gridmaps from perception and uses IRL nets to make them into costmaps
    """
    def __init__(self, grid_map_topic, cost_map_topic, odom_topic):
        """
        Args:
            grid_map_topic: the topic to get map features from
            cost_map_topic: The topic to publish costmaps to
            odom_topic: The topic to get height from 
            dataset: The dataset that the network was trained on. (Need to get feature mean/var)
            network: the network to produce costmaps.
        """
        self.current_height = 0.

        #we will set the output resolution dynamically
        self.grid_map_cvt = GridMapConvert(channels=['diff'], size=[1, 1])

        self.grid_map_sub = rospy.Subscriber(grid_map_topic, GridMap, self.handle_grid_map, queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.handle_odom, queue_size=1)
        self.cost_map_pub = rospy.Publisher(cost_map_topic, OccupancyGrid, queue_size=1)

    def handle_odom(self, msg):
        self.current_height = msg.pose.pose.position.z

    def handle_grid_map(self, msg):
        rospy.loginfo('handling gridmap...')
        nx = int(msg.info.length_x / msg.info.resolution)
        ny = int(msg.info.length_y / msg.info.resolution)
        self.grid_map_cvt.size = [nx, ny]
        gridmap = self.grid_map_cvt.ros_to_numpy(msg)

        rospy.loginfo_throttle(1.0, "output shape: {}".format(gridmap['data'].shape))

        map_feats = torch.from_numpy(gridmap['data']).float()

        #geometrically determine cost
        costmap = map_feats[0] > 1.0
        costmap = (costmap * 100.).long().numpy()

        costmap_msg = OccupancyGrid()
        costmap_msg.header.stamp = msg.info.header.stamp
        costmap_msg.header.frame_id = msg.info.header.frame_id
        costmap_msg.info.resolution = msg.info.resolution
        costmap_msg.info.width = int(msg.info.length_x / msg.info.resolution)
        costmap_msg.info.height = int(msg.info.length_y / msg.info.resolution)
        costmap_msg.info.origin.position.x = msg.info.pose.position.x - msg.info.length_x/2.
        costmap_msg.info.origin.position.y = msg.info.pose.position.y - msg.info.length_y/2.

        costmap_msg.data = costmap.flatten()

        self.cost_map_pub.publish(costmap_msg)

if __name__ == '__main__':
    rospy.init_node('costmapper_node')

    grid_map_topic = '/local_gridmap'
    cost_map_topic = '/local_cost_map_final_occupancy_grid'
    odom_topic = '/warty/odom'

    costmapper = CostmapperNode(grid_map_topic, cost_map_topic, odom_topic)

    rospy.spin()
