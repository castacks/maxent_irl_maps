#collection of rotation stuff to make life easier for some stuff

import torch
import numpy as np
import scipy.interpolate, scipy.spatial

class TrajectoryInterpolator:
    """
    Helper class for interpolating trajectories
    Expects either a 7-element trajectory as: [T x [x, y, z, qx, qy, qz, qw]]
    or a 13-element trajectory as: [T x [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]]

    funtionally, works identically to the scipy interpolation object
    """
    def __init__(self, times, traj,interp_kwargs={}):
        """
        Args:
            traj: the traj to interpolate (of shape [T x {7,13}])
            times: the times corresponding to the traj
            interp_kwargs: kwargs to pass to the interpolator
        """
        assert len(traj.shape) == 2, 'Expected traj of shape [T x 7/13], got {}'.format(traj.shape)
        assert traj.shape[-1] in [7, 13], 'Expected traj of shape [T x 7/13], got {}'.format(traj.shape)
        assert times.shape[0] == traj.shape[0], 'Got {} times, but {} steps in traj'.format(times.shape[0], traj.shape[0])

        self.is_velocity = (traj.shape[-1] == 13)

        #edge case check
        idxs = np.argsort(times)

        #interpolate traj to get accurate times
        self.interp_x = scipy.interpolate.interp1d(times[idxs], traj[idxs, 0], **interp_kwargs)
        self.interp_y = scipy.interpolate.interp1d(times[idxs], traj[idxs, 1], **interp_kwargs)
        self.interp_z = scipy.interpolate.interp1d(times[idxs], traj[idxs, 2], **interp_kwargs)
        
        rots = scipy.spatial.transform.Rotation.from_quat(traj[:, 3:7])
        self.interp_q = scipy.spatial.transform.Slerp(times[idxs], rots[idxs], **interp_kwargs)

        if self.is_velocity:
            self.interp_vx = scipy.interpolate.interp1d(times[idxs], traj[idxs, 7], **interp_kwargs)
            self.interp_vy = scipy.interpolate.interp1d(times[idxs], traj[idxs, 8], **interp_kwargs)
            self.interp_vz = scipy.interpolate.interp1d(times[idxs], traj[idxs, 9], **interp_kwargs)

            self.interp_wx = scipy.interpolate.interp1d(times[idxs], traj[idxs, 10], **interp_kwargs)
            self.interp_wy = scipy.interpolate.interp1d(times[idxs], traj[idxs, 11], **interp_kwargs)
            self.interp_wz = scipy.interpolate.interp1d(times[idxs], traj[idxs, 12], **interp_kwargs)

    def __call__(self, qtimes):
        """
        Interpolate the traj according to qtimes.
        Args:
            qtimes: the set of times to query
        """
        if self.is_velocity:
            xs = self.interp_x(qtimes)
            ys = self.interp_y(qtimes)
            zs = self.interp_z(qtimes)
            qs = self.interp_q(qtimes).as_quat()

            vxs = self.interp_vx(qtimes)
            vys = self.interp_vy(qtimes)
            vzs = self.interp_vz(qtimes)
            wxs = self.interp_wx(qtimes)
            wys = self.interp_wy(qtimes)
            wzs = self.interp_wz(qtimes)

            traj = np.concatenate([
                np.stack([xs, ys, zs], axis=-1),
                qs,
                np.stack([vxs, vys, vzs, wxs, wys, wzs], axis=-1)
            ], axis=-1)
        else:
            xs = self.interp_x(qtimes)
            ys = self.interp_y(qtimes)
            zs = self.interp_z(qtimes)
            qs = self.interp_q(qtimes).as_quat()

            traj = np.concatenate([
                np.stack([xs, ys, zs], axis=-1),
                qs
            ], axis=-1)

        return traj

if __name__ == '__main__':
    #test trajectory interpolator by reading a rosbag
    import rosbag
    import matplotlib.pyplot as plt

    bag_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train/20220630/2022-06-30-16-13-36_0.bag'
    odom_topic = '/odometry/filtered_odom'

    bag = rosbag.Bag(bag_fp, 'r')
    traj = []
    timestamps = []

    for topic, msg, t in bag.read_messages(topics=[odom_topic]):
        traj.append(np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]))
        timestamps.append(msg.header.stamp.to_sec())

    #zero x, y, z.
    traj = np.stack(traj, axis=0)
    traj[:, :3] -= traj[[0], :3]
    timestamps = np.array(timestamps)

    skip = 20
    subtraj = traj[::skip]
    subtimes = timestamps[::skip]

    tinterp = TrajectoryInterpolator(subtimes, subtraj)
    qtimes = timestamps[:-2*skip]

    itraj = tinterp(qtimes)

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
    axs[0].scatter(subtraj[:, 0], subtraj[:, 1], c='b', marker='.', label='knot pts')
    axs[0].plot(itraj[:, 0], itraj[:, 1], c='r', label='interp')
    axs[0].legend()

    colors = 'rgbcmyk'
    for i, label in enumerate(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']):
        color = colors[i % len(colors)]
        axs[1].scatter(subtimes, subtraj[:, i], c=color, label=label, s=1.)
        axs[1].plot(qtimes, itraj[:, i], c=color, alpha=0.5)
    axs[1].legend()
        
    plt.show()
