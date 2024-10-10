#! /usr/bin/python3

import os
import time
import torch
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32
from nav_msgs.msg import OccupancyGrid, Odometry
from grid_map_msgs.msg import GridMap

from ros_torch_converter.conversions.gridmap import GridMapToTorchMap

from maxent_irl_maps.experiment_management.parse_configs import setup_experiment
from maxent_irl_maps.networks.baselines import (
    AlterBaseline,
    SemanticBaseline,
    AlterSemanticBaseline,
)


class CvarCostmapperNode(Node):
    """
    Node that listens to gridmaps from perception and uses IRL nets to make them into costmaps
    In addition to the baseline costmappers, take in a CVaR value from -1.0 to 1.0 and use an ensemble of costmaps
    """

    def __init__(self):
        """ """
        super().__init__("cvar_maxent_irl")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("models_dir", ""),
                ("model_fp", ""),
                ("baseline", ""),
                ("gridmap_topic", ""),
                ("output_topic", ""),
                ("odom_topic", ""),
                ("cvar_topic", ""),
                ("speedmap_q_topic", ""),
                ("device", ""),
            ],
        )

        self.get_logger().info(
            "PARAMETERS:\n{}".format(
                {
                    k: v.get_parameter_value().string_value
                    for k, v in self._parameters.items()
                }
            )
        )

        models_dir = self.get_parameter("models_dir").get_parameter_value().string_value
        model_fp = self.get_parameter("model_fp").get_parameter_value().string_value
        model_fp = os.path.join(models_dir, model_fp)

        self.get_logger().info("loading IRL model {}".format(os.path.join(model_fp)))

        model_param_fp = os.path.join(os.path.split(model_fp)[0], "_params.yaml")
        irl = setup_experiment(model_param_fp)["algo"]
        irl.network.load_state_dict(torch.load(model_fp))
        irl.network.eval()

        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.gridmap_topic = (
            self.get_parameter("gridmap_topic").get_parameter_value().string_value
        )
        self.output_map_topic = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )
        self.odom_topic = (
            self.get_parameter("odom_topic").get_parameter_value().string_value
        )
        self.cvar_topic = (
            self.get_parameter("cvar_topic").get_parameter_value().string_value
        )
        self.speedmap_q_topic = (
            self.get_parameter("speedmap_q_topic").get_parameter_value().string_value
        )

        self.feature_keys = irl.expert_dataset.feature_keys
        self.feature_mean = (
            irl.expert_dataset.feature_mean[irl.expert_dataset.fidxs]
            .to(self.device)
            .view(-1, 1, 1)
        )
        self.feature_std = (
            irl.expert_dataset.feature_std[irl.expert_dataset.fidxs]
            .to(self.device)
            .view(-1, 1, 1)
        )

        self.network = irl.network.to(self.device)
        self.categorical_speedmaps = True

        self.cvar = 0.0
        self.speedmap_q = 0.5

        self.current_height = 0.0
        self.current_speed = 0.0

        # we will set the output resolution dynamically
        self.grid_map_cvt = GridMapToTorchMap(feature_keys=self.feature_keys)

        self.grid_map_sub = self.create_subscription(
            GridMap, self.gridmap_topic, self.handle_grid_map, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.handle_odom, 10
        )
        self.cvar_sub = self.create_subscription(
            Float32, self.cvar_topic, self.handle_cvar, 10
        )
        self.speedmap_q_sub = self.create_subscription(
            Float32, self.speedmap_q_topic, self.handle_lcb, 10
        )

        self.output_map_pub = self.create_publisher(GridMap, self.output_map_topic, 10)

    def handle_odom(self, msg):
        self.current_height = msg.pose.pose.position.z
        self.current_speed = msg.twist.twist.linear.x

    def handle_cvar(self, msg):
        if msg.data > -1.0 and msg.data < 1.0:
            self.cvar = msg.data
        else:
            self.get_logger().error(
                "CVaR expected to be (-1, 1) exclusive, got {}".format(msg.data)
            )

    def handle_lcb(self, msg):
        if msg.data > 0.0 and msg.data < 1.0:
            self.speedmap_q = msg.data
        else:
            self.get_logger().error(
                "Speed quantile expected to be (0, 1), got {}".format(msg.data)
            )

    def create_ego_features(self, keys, nx, ny):
        res = []
        for k in keys:
            if k == "ego_speed":
                res.append(torch.zeros(nx, ny, device=self.device) * self.current_speed)

        return torch.stack(res, dim=0)

    def compute_map_cvar(self, maps, cvar):
        if self.cvar < 0.0:
            map_q = torch.quantile(maps, 1.0 + cvar, dim=0)
            mask = maps <= map_q.view(1, *map_q.shape)
        else:
            map_q = torch.quantile(maps, self.cvar, dim=0)
            mask = maps >= map_q.view(1, *map_q.shape)

        cvar_map = (maps * mask).sum(dim=0) / mask.sum(dim=0)
        return cvar_map

    def handle_grid_map(self, msg):
        t1 = time.time()
        nx = int(msg.info.length_x / msg.info.resolution)
        ny = int(msg.info.length_y / msg.info.resolution)
        gridmap = self.grid_map_cvt.cvt(msg)

        ## create geometric features ##
        map_feats = gridmap["data"].float().to(self.device)
        for k in self.feature_keys:
            if k in [
                "height_low",
                "height_mean",
                "height_high",
                "height_max",
                "terrain",
            ]:
                idx = self.feature_keys.index(k)
                map_feats[idx] -= self.current_height

        map_feats[~torch.isfinite(map_feats)] = 0.0
        map_feats[map_feats.abs() > 100.0] = 0.0

        self.get_logger().info(
            "avg feat: {}".format(map_feats.mean(dim=-1).mean(dim=-1))
        )

        map_feats_norm = (
            map_feats - self.feature_mean.view(-1, 1, 1)
        ) / self.feature_std.view(-1, 1, 1)

        # this part is different for CVaR
        with torch.no_grad():
            res = self.network.ensemble_forward(
                map_feats_norm.view(1, *map_feats_norm.shape)
            )
            costmaps = res["costmap"][0, :, 0]

            if self.categorical_speedmaps:
                speedmap_probs = res["speedmap"][0].mean(dim=0).softmax(dim=0)
                speedmap_cdf = torch.cumsum(speedmap_probs, dim=0)
                speedmap = self.compute_speedmap_quantile(
                    speedmap_cdf,
                    self.network.speed_bins.to(self.device),
                    self.speedmap_q,
                )
            else:
                speedmap = res["speedmap"][0, 0]

        cvar_costmap = self.compute_map_cvar(costmaps, self.cvar)

        # experiment w/ normalizing
        self.get_logger().info(
            "cost min = {:.4f}, max = {:.4f}".format(
                cvar_costmap.min(), cvar_costmap.max()
            ),
            throttle_duration_sec=1.0,
        )
        self.get_logger().info(
            "speed min = {:.4f}, max = {:.4f}".format(speedmap.min(), speedmap.max()),
            throttle_duration_sec=1.0,
        )

        t2 = time.time()

        map_data = [cvar_costmap, speedmap]
        map_layers = ["costmap", "speedmap"]
        output_map = self.map_data_to_gridmap(msg, map_data, map_layers)
        self.output_map_pub.publish(output_map)

        t3 = time.time()
        self.get_logger().info(
            "inference time: {:.4f}s, msg time: {:.4f}s".format(t2 - t1, t3 - t2),
            throttle_duration_sec=1.0,
        )
        self.get_logger().info(
            "CVaR = {:.2f}, q = {:.2f}".format(self.cvar, self.speedmap_q),
            throttle_duration_sec=1.0,
        )

    def map_data_to_gridmap(self, msg, data, layers="costmap"):
        """
        convert costmap into gridmap msg

        Args:
            msg: The input msg to extract metadata from
            data: List of Tensors ([WxH]) to convert to gridmap
            layers: List of strings corresponding to data
        """
        msg_out = GridMap()
        msg_out.header = msg.header
        msg_out.info = msg.info
        msg_out.layers = layers

        for _data in data:
            layer_msg = Float32MultiArray()
            layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="column_index", size=_data.shape[0], stride=_data.shape[0]
                )
            )
            layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="row_index",
                    size=_data.shape[0],
                    stride=_data.shape[0] * _data.shape[1],
                )
            )

            layer_msg.data = _data.cpu().numpy()[::-1, ::-1].T.flatten().tolist()
            msg_out.data.append(layer_msg)
        return msg_out

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
        # need to stack zeros to front
        _cdf = torch.cat([torch.zeros_like(speedmap_cdf[[0]]), speedmap_cdf], dim=0)

        _cdf = _cdf.view(B, -1)

        _cdiffs = q - _cdf
        _mask = _cdiffs <= 0

        _qidx = (_cdiffs + 1e10 * _mask).argmin(dim=0)

        _cdf_low = _cdf.T[torch.arange(len(_qidx)), _qidx]
        _cdf_high = _cdf.T[torch.arange(len(_qidx)), _qidx + 1]

        _k = (q - _cdf_low) / (_cdf_high - _cdf_low)

        _speedmap_low = speed_bins[_qidx]
        _speedmap_high = speed_bins[_qidx + 1]

        _speedmap = (1 - _k) * _speedmap_low + _k * _speedmap_high

        return _speedmap.reshape(*speedmap_cdf.shape[1:])


def main(args=None):
    rclpy.init(args=args)

    costmapper_node = CvarCostmapperNode()
    rclpy.spin(costmapper_node)

    costmapper_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
