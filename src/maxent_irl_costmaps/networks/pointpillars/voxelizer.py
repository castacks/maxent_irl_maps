"""
Take a pointcloud and voxelize it into pillars
"""

import torch
import numpy as np
from numba import jit

#@jit(nopython=True, fastmath=True, parallel=True)
def pillarize(pcl, pos, length_x, length_y, res, max_npoints=100, max_npillars=10000):
    """
    Take a pointcloud as a numpy array and bin it into pillars according to metadata/pos
    Args:
        pcl: the pointcloud to pillarize ([P x 3])
        pos: the position to CENTER the grid around ([3]) (i.e. grid is from [pos-length/2, pos + length/2]). Note that we still need z to normalize the pillars
        metadata: the metadata of the grid (i.e. resolution, length)
        max_npoints: maximum number of points per pillar
        max_npillars: maximum number of pillars

    Returns:
        pillars: [max_npillars x max_npoints x 8] the pillars to process
        pillar_idxs: [max_npillars x 2] the indexes of the pillars in the discretized grid

    Some things to note:
        1. we want to do this in numpy so we can compile it with numba
        2. beacuse I'm numba'ing it, I'm writing it the slow way
    """
    ox = -length_x/2
    oy = -length_y/2
    nx = int(length_x / res)
    ny = int(length_y / res)
    width = max(nx, ny)

    #center the pointcloud at the provided pose
    pcl = pcl - np.expand_dims(pos, axis=0)

    #first filter out points not in bounds
    valid_mask = (pcl[:, 0] >= ox) & (pcl[:, 0] < ox + length_x) & (pcl[:, 1] >= oy) & (pcl[:, 1] < oy + length_y)
    pcl = pcl[valid_mask]

    grid_xs = ((pcl[:, 0] - ox) / res).astype(np.int32)
    grid_ys = ((pcl[:, 1] - oy) / res).astype(np.int32)

    grid_hashes = grid_xs * width + grid_ys
    cnts = np.bincount(grid_hashes)
    topk_grid_hashes = np.argsort(cnts)[-max_npillars:]

    pillars = np.zeros((max_npillars, max_npoints, 8))
    pillar_idxs = -np.ones((max_npillars, 2)).astype(np.int32)
    #yay jit
    i = 0
    for grid_hash in topk_grid_hashes:
        points = pcl[grid_hashes == grid_hash]

        #since we're sorted in descendig order, can break
        if points.shape[0] == 0:
            break

        pillar_xidx = (grid_hash // width)
        pillar_yidx = (grid_hash % width)
        pillar_x = (pillar_xidx * res) + ox + res/2
        pillar_y = (pillar_yidx * res) + oy + res/2

        pillar_centroid = np.zeros((1, 3))
        k = points.shape[0]
        for point in points:
            pillar_centroid[0] += point / k

        centroid_diffs = points - pillar_centroid
        pillar_diffs = np.stack((points[:, 0] - pillar_x, points[:, 1] - pillar_y), axis=-1)
        decorated_points = np.concatenate((points, centroid_diffs, pillar_diffs), axis=-1)

        if decorated_points.shape[0] > max_npoints:
            #shuffle only shuffles the first axis
            np.random.shuffle(decorated_points)
            decorated_points = decorated_points[:max_npoints]

        pillars[i, :decorated_points.shape[0], :] = decorated_points
        pillar_idxs[i, 0] = pillar_xidx
        pillar_idxs[i, 1] = pillar_yidx
        i += 1

    return pillars, pillar_idxs

def plot_pillars(pillars, pillar_idxs, metadata):
    fig = plt.figure(figsize = (18, 9))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2)
    res = metadata['resolution']
    nx = int(metadata['length_x'] / res)
    ny = int(metadata['length_y'] / res)
    pillar_grid = np.zeros([nx, ny])
    pts = []
    for pillar, pidx in zip(pillars, pillar_idxs):
        subpillar = pillar[np.linalg.norm(pillar, axis=-1) > 1e-2]
        pts.append(subpillar)
        pillar_grid[pidx[0], pidx[1]] = subpillar.shape[0]

    allpts = np.concatenate(pts, axis=0)
    ax0.scatter(allpts[:, 0], allpts[:, 1], allpts[:, 2], s=1., alpha=0.2, c=allpts[:, 2]/allpts[:, 2].max(), cmap='rainbow')
    ax1.imshow(pillar_grid.T, origin='lower')
    plt.show()

if __name__ == '__main__':
    import time
    import rosbag
    import ros_numpy
    from sensor_msgs.msg import PointCloud2
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    bag_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps_pointclouds/rosbags_debug/2022-07-10-13-48-09_1.bag'
    bag = rosbag.Bag(bag_fp, 'r')
    pcl_topic = '/merged_pointcloud'
    pose_topic = '/integrated_to_init'
    pose = None
    times = []
    cnt = 0

    metadata = {
        'origin':None,
        'length_x':120.0,
        'length_y':120.0,
        'resolution':0.5
    }

    for topic, msg, t in bag.read_messages(topics=[pcl_topic, pose_topic]):
        if topic == pose_topic:
            pose = msg.pose

        if topic == pcl_topic and pose is not None:
            if (cnt % 100) == 0:
                #dumb rosbag hack
                msg2 = PointCloud2(
                    header = msg.header,
                    height = msg.height,
                    width = msg.width,
                    fields = msg.fields,
                    is_bigendian = msg.is_bigendian,
                    point_step = msg.point_step,
                    row_step = msg.row_step,
                    data = msg.data,
                    is_dense = msg.is_dense
                )
                pcl = ros_numpy.numpify(msg2)
                pcl_tensor = np.stack([pcl[k] for k in 'xyz'], axis=-1)
                pos_tensor = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
                t1 = time.time()
                pillars, pillar_idxs = pillarize(pcl_tensor, pos_tensor, metadata['length_x'], metadata['length_y'], metadata['resolution'], max_npillars=10000)
                t2 = time.time()
                times.append(t2-t1)
                break

                plot_pillars(pillars, pillar_idxs, metadata)
            cnt += 1

    times = np.array(times)
    print('AVG PROCESS = {:.4f}'.format(times.mean()))
    print('')
