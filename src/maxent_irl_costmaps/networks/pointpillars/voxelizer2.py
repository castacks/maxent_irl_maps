"""
Ok, second attempt. To recap, we need to take a pointcloud and
    1. Determine the k most populated pillars
    2. For each point
        a. determine if in pillar
        b. if so, add to pillar centroid
        c. and if space, add to pillar tensor
"""
import torch
import numpy as np
from numba import jit

def pillarize(pcl, pos, metadata, max_npoints=100, max_npillars=10000):
    """
    sanitize inputs and pass to numba
    """
    #first center the pointcloud
    pcl_center = pcl - pos

    #filter extraneous points
    ox = -metadata['length_x']/2.
    oy = -metadata['length_y']/2.
    lx = metadata['length_x']
    ly = metadata['length_y']
    res = metadata['resolution']
    nx = int(lx / res)
    ny = int(ly / res)
    width = max(nx, ny)
    mask = (pcl_center[:, 0] >= ox) & (pcl_center[:, 0] < ox + lx) & (pcl_center[:, 1] >= oy) & (pcl_center[:, 1] < oy + ly)
    pcl_center = pcl_center[mask]

    #bin points into cells
    grid_xs = ((pcl_center[:, 0] - ox) / res).astype(np.int32)
    grid_ys = ((pcl_center[:, 1] - oy) / res).astype(np.int32)
    grid_hashes = grid_xs * width + grid_ys
    cnts = np.bincount(grid_hashes)
    topk_grid_hashes = np.argsort(cnts)[-max_npillars:]

    pillars = np.ones((max_npillars, max_npoints, 3)) * 0.
    pillar_cnts = np.zeros(max_npillars).astype(np.int32)
    pillar_idxs = np.stack([
        topk_grid_hashes // width,
        topk_grid_hashes % width
    ], axis=-1)

    grid_hash_mapping = -np.ones(cnts.shape[0]).astype(np.int32)
    grid_hash_mapping[topk_grid_hashes] = np.arange(topk_grid_hashes.shape[0])

    pillarize_kernel(
        pcl_center,
        pillars,
        pillar_cnts,
        grid_hash_mapping,
        ox,
        oy,
        nx,
        ny,
        res,
        max_npoints
    )

    #SAM TODO: move decoration to the numba part
    pillar_centroids = pillars.sum(axis=1) / np.expand_dims(pillar_cnts, axis=-1)
    centroid_diffs = pillars - np.expand_dims(pillar_centroids, axis=1)

    pillar_centers = np.stack([
        pillar_idxs[:, 0] * res + ox + res/2.,
        pillar_idxs[:, 1] * res + oy + res/2.,
    ], axis=-1)
    pillar_diffs = pillars[:, :, :2] - np.expand_dims(pillar_centers, axis=1)

    decorated_points = np.concatenate([
        pillars,
        centroid_diffs,
        pillar_diffs
    ], axis=-1)

    k = np.arange(max_npoints).reshape(1, -1)
    k2 = np.tile(k, (max_npillars, 1))
    empty_mask = k2 >= np.expand_dims(pillar_cnts, axis=-1)
    decorated_points[empty_mask] = 1e6

    return torch.tensor(decorated_points).float(), torch.tensor(pillar_idxs).long()

@jit(nopython=True, fastmath=True)
def pillarize_kernel(
                        pcl,
                        pillars,
                        pillar_cnt,
                        grid_hash_mapping,
                        ox,
                        oy,
                        nx,
                        ny,
                        res,
                        max_npoints
                    ):
    """
    speed this part up with numba. 
    Args:
        pcl: [N x 3] array of points to pillarize
        pillars: [B x P x 3] accumulator of points
        pillar_cnt: [B] array to keep track of num pts
        grid_hash_mapping:
    """
    width = max(nx, ny)
    for ii, point in enumerate(pcl):
        gx = int((point[0] - ox) / res)
        gy = int((point[1] - oy) / res)
        pt_hash = gx * width + gy
        if grid_hash_mapping[pt_hash] != -1:
            pillar_idx = grid_hash_mapping[pt_hash]
            if pillar_cnt[pillar_idx] < max_npoints:
                pillars[pillar_idx, pillar_cnt[pillar_idx]] = point
                pillar_cnt[pillar_idx] += 1

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
        valid_mask = np.all(np.isfinite(pillar), axis=-1)
        subpillar = pillar[valid_mask]
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
                pillars, pillar_idxs = pillarize(pcl_tensor, pos_tensor, metadata, max_npillars=10000, max_npoints=128)
                t2 = time.time()
                times.append(t2-t1)

                plot_pillars(pillars, pillar_idxs, metadata)
                break

    times = np.array(times)
    print('Avg pillarize time: {:.4f}s'.format(times.mean()))
