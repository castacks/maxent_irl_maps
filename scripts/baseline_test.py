import rosbag
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import scipy.spatial
import scipy.interpolate

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
from torch_mpc.algos.batch_mppi import BatchMPPI
from torch_mpc.cost_functions.waypoint_costmap import WaypointCostMapCostFunction

"""
Very simple initial implementation of maxent IRL on costmaps
Expecting the following from the rosbags at this point:
    1. Trajectories on the /odom topic
    2. Map features on the /local_gridmap topic
    3. Validation costmaps on the /costmap topic
"""

def load_data(bag_fp):
    """
    Extract trajectory data from the bag
    some of the gridmap/occupancygrid metadata stuff is annoying so I'm not using rosbag to dataset rn.
    we're also assuming for now that the maps are global/static
    """
    map_features = None
    costmap = None
    metadata = None
    traj = []
    timestamps = []

    map_topic = '/global_map'
    costmap_topic = '/local_cost_map_final_occupancy_grid'
    odom_topic = '/odom'

    bag = rosbag.Bag(bag_fp, 'r')
    for topic, msg, t in bag.read_messages(topics=[map_topic, costmap_topic, odom_topic]):
        if topic == odom_topic:
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
            timestamps.append(msg.header.stamp.to_sec())
        elif topic == map_topic:
            if map_features is None:
                map_features = msg
        elif topic == costmap_topic:
            if costmap is None:
                costmap = msg

    traj = np.stack(traj, axis=0)
    timestamps = np.array(timestamps) - timestamps[0]

    #interpolate traj to get accurate timestamps
    interp_x = scipy.interpolate.interp1d(timestamps, traj[:, 0])
    interp_y = scipy.interpolate.interp1d(timestamps, traj[:, 1])
    interp_z = scipy.interpolate.interp1d(timestamps, traj[:, 2])
    
    rots = scipy.spatial.transform.Rotation.from_quat(traj[:, 3:])
    interp_q = scipy.spatial.transform.Slerp(timestamps, rots)

    targets = np.arange(start=0., stop=timestamps[-1], step=0.1)
    xs = interp_x(targets)
    ys = interp_y(targets)
    zs = interp_z(targets)
    qs = interp_q(targets).as_quat()

    traj = np.concatenate([np.stack([xs, ys, zs], axis=-1), qs], axis=-1)

    #handle transforms to deserialize map/costmap
    map_feature_data = []
    map_feature_keys = []
    mf_nx = int(map_features.info.length_x/map_features.info.resolution)
    mf_ny = int(map_features.info.length_y/map_features.info.resolution)
    for k,v in zip(map_features.layers, map_features.data):
        data = np.array(v.data).reshape(mf_nx, mf_ny)[::-1, ::-1]

        #debug test
#        data += np.random.randn(*data.shape) * 0.1

        map_feature_keys.append(k)
        map_feature_data.append(data)

    map_feature_data.append(np.ones([mf_nx, mf_ny]))
    map_feature_keys.append('bias')

    #debug test
#    map_feature_data.append(np.random.randn(mf_nx, mf_ny))
#    map_feature_keys.append('noise')

    costmap_data = np.array(costmap.data).reshape(costmap.info.width, costmap.info.height)
    map_metadata = costmap.info
    map_feature_data = np.stack(map_feature_data, axis=0)

    return {
        'traj':torch.tensor(traj).float(),
        'feature_keys': map_feature_keys,
        'map_features': torch.tensor(map_feature_data).float(),
        'costmap': torch.tensor(costmap_data).float(),
        'metadata': map_metadata
    }

def get_empirical_feature_counts(map_features, trajs, map_metadata):
    """
    Register trajectories on the costmap to get expected feature counts.
    """
    xs = trajs[...,0]
    ys = trajs[...,1]
    res = map_metadata.resolution
    ox = map_metadata.origin.position.x
    oy = map_metadata.origin.position.y

    xidxs = ((xs - ox) / res).long()
    yidxs = ((ys - oy) / res).long()

    valid_mask = (xidxs >= 0) & (xidxs < map_metadata.width) & (yidxs >= 0) & (yidxs < map_metadata.height)

    xidxs[~valid_mask] = 0
    yidxs[~valid_mask] = 0

    # map data is transposed
    features = map_features[:, yidxs, xidxs]

    feature_counts = features.mean(dim=-1)

    empirical_feature_counts = feature_counts.mean(dim=-1)

    return empirical_feature_counts

def compute_costmap(map_features, weights):
    """
    Compute a costmap as a linear combination of map_features, weighted by weights
    """
    return (map_features * weights.view(-1, 1, 1)).sum(dim=0)

def optimize_traj(mppi, initial_state, costmap):
    """
    compute an "optimal" trajectory via MPPI through the model, from the initial state, on the costmap
    """
    mppi.cost_fn.update_costmap(costmap)
    x = mppi.model.get_observations({'state':initial_state, 'steer_angle':torch.zeros(mppi.B, 1)})
    for i in range(5):
        mppi.get_control(x, step=False)

    return mppi.last_states

def maxent_irl(dataset, mppi):
    """
    Run maxent IRL to generate a costmap. This is done in the following way:
    1. Compute expert expected feature counts
    2. While not converged:
        a. Compute a costmap from the current weight vector
        b. Optimize a set of trajectories through this costmap predictor
        c. Get expected feature counts for these trajectories
        d. Update the weight vector to match expected feature counts
    """
    expert_traj = dataset['traj']
    map_features = dataset['map_features']
    map_metadata = dataset['metadata']
    weights = torch.zeros(len(dataset['feature_keys']), requires_grad=True)

    expert_feature_counts = get_empirical_feature_counts(map_features, expert_traj.unsqueeze(0), map_metadata)

    for itr in range(50):
        costmap = compute_costmap(map_features, weights)
        #Get policy feature counts
        trajs = []
        init_states = expert_traj[torch.randint(expert_traj.shape[0], (mppi.B, ))]

        with torch.no_grad():
            trajs = optimize_traj(mppi, init_states, costmap)

        learner_feature_counts = get_empirical_feature_counts(map_features, trajs, map_metadata)

        #Gradient step
        grad = expert_feature_counts - learner_feature_counts
        weights = weights - 1.0 * grad

        print('_______ITR {}_______'.format(itr + 1))
        print('Expert Feature Counts: ', expert_feature_counts)
        print('Learner Feature Counts: ', learner_feature_counts)
        print('Weights: ', weights)

        #Viz check
        if (itr % 5) == 0:
            xmin = map_metadata.origin.position.x
            xmax = map_metadata.resolution * map_metadata.width + xmin
            ymin = map_metadata.origin.position.y
            ymax = map_metadata.resolution * map_metadata.height + ymin
            plt.imshow(costmap.detach(), origin='lower', extent=(xmin, xmax, ymin, ymax))

            for traj in trajs:
                plt.plot(traj[:, 0], traj[:, 1], c='r')

            plt.plot(expert_traj[:, 0], expert_traj[:, 1], c='b')
            plt.title('Itr {}'.format(itr + 1))

            plt.show()

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_bag', type=str, required=True, help='Bag of expert data to train from')
    parser.add_argument('--test_bag', type=str, required=True, help='Bag of expert data to test on')
    args = parser.parse_args()

    train_data = load_data(args.train_bag)
    test_data = load_data(args.test_bag)

    torch.save(train_data, 'train_data.pt')
    torch.save(test_data, 'test_data.pt')

    train_data = torch.load('train_data.pt')
    test_data = torch.load('test_data.pt')

    horizon = 70
    batch_size = 100

    kbm = SteerSetpointKBM(L=1.0, v_target_lim=[1.0, 2.0], steer_lim=[-0.3, 0.3], steer_rate_lim=0.2)

    parameters = {
        'log_K_delta':torch.tensor(10.0)
    }
    kbm.update_parameters(parameters)

    map_params = {
        'height':train_data['metadata'].height * train_data['metadata'].resolution,
        'width':train_data['metadata'].width * train_data['metadata'].resolution,
        'resolution':train_data['metadata'].resolution,
        'origin':torch.tensor([train_data['metadata'].origin.position.x, train_data['metadata'].origin.position.y])
    }
    cfn = WaypointCostMapCostFunction(unknown_cost=10., goal_cost=1000., map_params=map_params)
    mppi = BatchMPPI(model=kbm, cost_fn=cfn, num_samples=2048, num_timesteps=horizon, control_params={'sys_noise':torch.tensor([1.0, 0.5]), 'temperature':0.05}, batch_size=batch_size)

    costmap_weights = maxent_irl(train_data, mppi)
