import rosbag
import torch
import numpy as np
import scipy.interpolate, scipy.spatial
import cv2

def load_data(bag_fp, map_features_topic, odom_topic, image_topic, horizon, dt, fill_value):
    """
    Extract map features and trajectory data from the bag.
    """
    print(bag_fp)
    map_features_list = []
    traj = []
    vels = []
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

            twist = msg.twist.twist
            v = np.array([
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ])

            traj.append(p)
            vels.append(v)
            timestamps.append(msg.header.stamp.to_sec())
        elif topic == map_features_topic:
            map_features_list.append(msg)

    traj = np.stack(traj, axis=0)
    vels = np.stack(vels, axis=0)
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

    interp_vx = scipy.interpolate.interp1d(timestamps[idxs], vels[idxs, 0])
    interp_vy = scipy.interpolate.interp1d(timestamps[idxs], vels[idxs, 1])
    interp_vz = scipy.interpolate.interp1d(timestamps[idxs], vels[idxs, 2])

    interp_wx = scipy.interpolate.interp1d(timestamps[idxs], vels[idxs, 3])
    interp_wy = scipy.interpolate.interp1d(timestamps[idxs], vels[idxs, 4])
    interp_wz = scipy.interpolate.interp1d(timestamps[idxs], vels[idxs, 5])

    map_target_times = []

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
        map_target_times.append(start_time)

        xs = interp_x(targets)
        ys = interp_y(targets)
        zs = interp_z(targets)
        qs = interp_q(targets).as_quat()

        vxs = interp_vx(targets)
        vys = interp_vy(targets)
        vzs = interp_vz(targets)
        wxs = interp_wx(targets)
        wys = interp_wy(targets)
        wzs = interp_wz(targets)

        #handle transforms to deserialize map/costmap
        traj = np.concatenate([
            np.stack([xs, ys, zs], axis=-1),
            qs,
            np.stack([vxs, vys, vzs, wxs, wys, wzs], axis=-1)
        ], axis=-1)

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
        'metadata':[x['metadata'] for x in dataset],
    }

    #If image topic exists, add to bag
    if image_topic is not None:
        image_timestamps = []
        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            image_timestamps.append(t.to_sec())
        image_timestamps = np.array(image_timestamps)
        #get closest image to targets
        dists = np.abs(np.expand_dims(image_timestamps, axis=0) - np.expand_dims(map_target_times, axis=1))
        image_targets = np.argmin(dists, axis=1)

        images = []
        for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[image_topic])):
            n_hits = np.sum(image_targets == i)
            for j in range(n_hits):
                img = np.frombuffer(msg.data, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                images.append(torch.tensor(img).permute(2, 0, 1)[[2, 1, 0]] / 255.)

        dataset['image'] = images

    return dataset, feature_keys
