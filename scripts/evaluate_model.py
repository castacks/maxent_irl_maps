import rosbag
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import scipy.spatial
import scipy.interpolate

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
from torch_mpc.models.skid_steer import SkidSteer

"""
validate model accuracy
"""
if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_fp', type=str, required=True, help='Path to raw data')
    parser.add_argument('--odom_topic', type=str, required=False, default='/warty/odom', help='topic to read odom from')
    parser.add_argument('--cmd_topic', type=str, required=False, default='/warty/rc_teleop/cmd_vel')
    args = parser.parse_args()

    model = SkidSteer()

    traj = []
    cmds = []
    odom_timestamps = []
    cmd_timestamps = []

    bag = rosbag.Bag(args.bag_fp, 'r')
    for topic, msg, t in bag.read_messages(topics=[args.odom_topic, args.cmd_topic]):
        if topic == args.odom_topic:
            pose = msg.pose.pose
            p = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ])

            traj.append(p)
            odom_timestamps.append(msg.header.stamp.to_sec())
        elif topic == args.cmd_topic:
            cmd = np.array([
                msg.linear.x,
                msg.angular.z
            ])

            cmds.append(cmd)
            cmd_timestamps.append(t.to_sec()) #cmd messages not stamped

    traj = np.stack(traj, axis=0)
    odom_timestamps = np.array(odom_timestamps)

    cmds = np.stack(cmds, axis=0)
    cmd_timestamps = np.array(cmd_timestamps)

    #interpolate traj to get accurate timestamps
    interp_x = scipy.interpolate.interp1d(odom_timestamps, traj[:, 0])
    interp_y = scipy.interpolate.interp1d(odom_timestamps, traj[:, 1])
    interp_z = scipy.interpolate.interp1d(odom_timestamps, traj[:, 2])
    
    rots = scipy.spatial.transform.Rotation.from_quat(traj[:, 3:])
    interp_q = scipy.spatial.transform.Slerp(odom_timestamps, rots)
    
    interp_v = scipy.interpolate.interp1d(cmd_timestamps, cmds[:, 0])
    interp_w = scipy.interpolate.interp1d(cmd_timestamps, cmds[:, 1])

    #validate forward rollout
    dt = 0.1
    start_time = max(odom_timestamps[0], cmd_timestamps[0])
    times = start_time + np.arange(50) * dt

    X_gt = np.stack([
        interp_x(times),
        interp_y(times),
        interp_q(times).as_euler('zxy')[:, 0]
    ], axis=-1)

    U = np.stack([
        interp_v(times),
        interp_w(times)
    ], axis=-1)

    #model
    model = SkidSteer()
    X_pred = model.rollout(torch.from_numpy(X_gt[0]), torch.from_numpy(U))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(X_gt[:, 0], X_gt[:, 1], c='b', label='gt')
    axs[0].plot(X_pred[:, 0], X_pred[:, 1], c='r', label='pred')

    for i, label in enumerate(['x', 'y', 'th']):
        axs[1].plot(X_gt[:, i], label='{}_gt'.format(label))
        axs[1].plot(X_pred[:, i], label='{}_pred'.format(label))

    for ax in axs:
        ax.legend()
    plt.show()
