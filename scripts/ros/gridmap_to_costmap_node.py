#! /usr/bin/python3

import rospy
import numpy as np
import torch

from nav_msgs.msg import OccupancyGrid
from grid_map_msgs.msg import GridMap

from rosbag_to_dataset.dtypes.gridmap import GridMapConvert

class CostmapperNode:
    """
    Node that listens to gridmaps from perception and uses IRL nets to make them into costmaps
    """
    def __init__(self, grid_map_topic, cost_map_topic):
        self.grid_map_sub = rospy.Subscriber(grid_map_topic, GridMap, self.handle_grid_map, queue_size=1)
        self.cost_map_pub = rospy.Publisher(cost_map_topic, OccupancyGrid, queue_size=1)
        self.grid_map_cvt = GridMapConvert(channels=['diff'], output_resolution=[120, 120])

    def handle_grid_map(self, msg):
        rospy.loginfo('handling gridmap...')
        gridmap = self.grid_map_cvt.ros_to_numpy(msg)
        gridmap[~np.isfinite(gridmap)] = 0.
        costmap = (gridmap[0] > 1.3).astype(np.uint8) * 100

        costmap_msg = OccupancyGrid()
        costmap_msg.header.stamp = msg.info.header.stamp
        costmap_msg.header.frame_id = msg.info.header.frame_id
        costmap_msg.info.resolution = msg.info.resolution
        costmap_msg.info.width = int(msg.info.length_x / msg.info.resolution)
        costmap_msg.info.height = int(msg.info.length_y / msg.info.resolution)
        costmap_msg.info.origin.position.x = msg.info.pose.position.x - msg.info.length_x/2.
        costmap_msg.info.origin.position.y = msg.info.pose.position.y - msg.info.length_y/2.

        costmap_msg.data = costmap[::-1, ::-1].flatten()

        self.cost_map_pub.publish(costmap_msg)

if __name__ == '__main__':
    rospy.init_node('costmapper_node')

    grid_map_topic = '/local_gridmap'
    cost_map_topic = '/local_cost_map_final_occupancy_grid'

    costmapper = CostmapperNode(grid_map_topic, cost_map_topic)

    rospy.spin()
