import rosbag
import torch
import numpy as np
import scipy.interpolate, scipy.spatial
    
def load_data(bag_fp, map_features_topic, odom_topic, horizon, dt, fill_value):
    """
    Extract map features and trajectory data from the bag.
    """
    map_features_list = []
    traj = []
    timestamps = []
    dataset = []

    bag = rosbag.Bag(bag_fp, 'r')
    for topic, msg, t in bag.read_messages(topics=[map_features_topic, odom_topic]):
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
        elif topic == map_features_topic:
            map_features_list.append(msg)

    traj = np.stack(traj, axis=0)
    timestamps = np.array(timestamps)

    #edge case check
    idxs = np.argsort(timestamps)

    #filter out maps that dont have enough trajectory
    map_features_list = [x for x in map_features_list if (x.info.header.stamp.to_sec() > timestamps.min()) and (x.info.header.stamp.to_sec() < timestamps.max() - (horizon*dt))]

    if len(map_features_list) == 0:
        return None, None

    #interpolate traj to get accurate timestamps
    interp_x = scipy.interpolate.interp1d(timestamps[idxs], traj[idxs, 0])
    interp_y = scipy.interpolate.interp1d(timestamps[idxs], traj[idxs, 1])
    interp_z = scipy.interpolate.interp1d(timestamps[idxs], traj[idxs, 2])
    
    rots = scipy.spatial.transform.Rotation.from_quat(traj[:, 3:])
    interp_q = scipy.spatial.transform.Slerp(timestamps[idxs], rots[idxs])

    #get a registered trajectory for each map.
    for i, map_features in enumerate(map_features_list):
        print('{}/{}'.format(i+1, len(map_features_list)), end='\r')
        map_feature_data = []
        map_feature_keys = []
        mf_nx = int(map_features.info.length_x/map_features.info.resolution)
        mf_ny = int(map_features.info.length_y/map_features.info.resolution)
        for k,v in zip(map_features.layers, map_features.data):
            #temp hack bc I don't like this feature.
#            if k not in ['roughness', 'height_high']:
            if 'npts' in k:
                continue
#                data = np.array(v.data).reshape(mf_nx, mf_ny)[::-1, ::-1]
#                data[data > 0] = np.log(data[data > 0])
#                map_feature_keys.append('log_' + k)
#                map_feature_data.append(data)

            else:
                data = np.array(v.data).reshape(mf_nx, mf_ny)[::-1, ::-1]

                map_feature_keys.append(k)
                map_feature_data.append(data)

#        map_feature_data.append(np.ones([mf_nx, mf_ny]))
#        map_feature_keys.append('bias')

        map_feature_data = np.stack(map_feature_data, axis=0)
        map_feature_data[~np.isfinite(map_feature_data)] = fill_value

        start_time = map_features.info.header.stamp.to_sec()
        targets = start_time + np.arange(horizon) * dt

        xs = interp_x(targets)
        ys = interp_y(targets)
        zs = interp_z(targets)
        qs = interp_q(targets).as_quat()

        #handle transforms to deserialize map/costmap
        traj = np.concatenate([np.stack([xs, ys, zs], axis=-1), qs], axis=-1)

        map_metadata = map_features.info
        xmin = map_metadata.pose.position.x - 0.5 * (map_metadata.length_x)
        xmax = xmin + map_metadata.length_x
        ymin = map_metadata.pose.position.y - 0.5 * (map_metadata.length_y)
        ymax = ymin + map_metadata.length_y

        metadata_out = {
            'resolution': map_metadata.resolution,
            'height': map_metadata.length_x,
            'width': map_metadata.length_y,
            'origin': torch.tensor([map_metadata.pose.position.x - 0.5 * (map_metadata.length_x), map_metadata.pose.position.y - 0.5 * (map_metadata.length_y)])
        }

        data = {
            'traj':torch.tensor(traj).float(),
            'feature_keys': map_feature_keys,
            'map_features': torch.tensor(map_feature_data).float(),
            'metadata': metadata_out
        }
        dataset.append(data)

    #convert from gridmap to occgrid metadata
    feature_keys = dataset[0]['feature_keys']
    dataset = {
        'map_features':[x['map_features'] for x in dataset],
        'traj':[x['traj'] for x in dataset],
        'metadata':[x['metadata'] for x in dataset]
    }

    return dataset, feature_keys
