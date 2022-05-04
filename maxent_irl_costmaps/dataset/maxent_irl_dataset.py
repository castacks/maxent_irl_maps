import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.interpolate, scipy.spatial
import rosbag
import yaml

from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped

class MaxEntIRLDataset(Dataset):
    """
    Dataset for maxent IRL on costmaps that contains the following:
        1. A set of map features (plus metadata) extracted from rosbags
        2. A trajectory (registered on the map)
    """
    def __init__(self, bag_fp, config_fp, map_features_topic='/local_gridmap', odom_topic='/integrated_to_init', horizon=70, dt=0.1, fill_value=0.):
        """
        Args:
            bag_fp: The bag to get data from
            config_fp: Config file describing model input (we only listen to state)
            map_features_topic: the topic to read map features from
            odom_topic: the topic to read trajectories from
            horizon: The number of trajectory steps to register onto each map

        Note that since the map changes here, we want to treat each as an MDP and make data samples for each map.
        """
        self.map_features_topic = map_features_topic
        self.odom_topic = odom_topic
        self.horizon = horizon
        self.dt = dt
        self.fill_value = fill_value #I don't know if this is the best way to do this, but setting the fill value to 0 implies that missing features contribute nothing to the cost.

        self.dataset, self.feature_keys = self.load_data(bag_fp)
        self.normalize_map_features()

        self.expert_feature_counts = self.get_expert_feature_counts()

    def normalize_map_features(self):
        """
        normalize map features to make learning less bad.
        Planning on doing a gaussian norm + clip to 3 stddevs.
        """
        map_feats = torch.stack(self.dataset['map_features'], dim=-1)

        #get mean/stddev over last 3 dims
        map_data = map_feats.view(len(self.feature_keys), -1)
        feat_mean = map_data.mean(dim=-1).view(-1, 1, 1)
        feat_std = map_data.std(dim=-1).view(-1, 1, 1) + 1e-6
        for i in range(len(self)):
            norm_feats = (self.dataset['map_features'][i] - feat_mean) / feat_std
            self.dataset['map_features'][i] = norm_feats.clip(-3, 3)

    def get_expert_feature_counts(self):
        """
        Compute expert feature counts per MaxEnt IRL
        """
        counts = []
        for i in range(len(self)):
            traj = self.dataset['traj'][i]
            map_features = self.dataset['map_features'][i]
            map_metadata = self.dataset['metadata'][i]

            empirical_feature_counts = self.get_feature_counts(traj, map_features, map_metadata)

            counts.append(empirical_feature_counts)

        counts = torch.stack(counts, dim=0)
        return counts.mean(dim=0)

    def get_feature_counts(self, traj, map_features, map_metadata):
        xs = traj[...,0]
        ys = traj[...,1]
        res = map_metadata['resolution']
        ox = map_metadata['origin'][0]
        oy = map_metadata['origin'][1]

        xidxs = ((xs - ox) / res).long()
        yidxs = ((ys - oy) / res).long()

        valid_mask = (xidxs >= 0) & (xidxs < map_features.shape[2]) & (yidxs >= 0) & (yidxs < map_features.shape[1])

        xidxs[~valid_mask] = 0
        yidxs[~valid_mask] = 0

        # map data is transposed
        features = map_features[:, yidxs, xidxs]

        feature_counts = features.mean(dim=-1)
        return feature_counts
        
    def load_data(self, bag_fp):
        """
        Extract map features and trajectory data from the bag.
        """
        map_features_list = []
        traj = []
        timestamps = []
        dataset = []

        bag = rosbag.Bag(bag_fp, 'r')
        for topic, msg, t in bag.read_messages(topics=[self.map_features_topic, self.odom_topic]):
            if topic == self.odom_topic:
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
            elif topic == self.map_features_topic:
                map_features_list.append(msg)

        traj = np.stack(traj, axis=0)
        timestamps = np.array(timestamps)

        #filter out maps that dont have enough trajectory
        map_features_list = [x for x in map_features_list if (x.info.header.stamp.to_sec() > timestamps.min()) and (x.info.header.stamp.to_sec() < timestamps.max() - (self.horizon*self.dt))]

        #interpolate traj to get accurate timestamps
        interp_x = scipy.interpolate.interp1d(timestamps, traj[:, 0])
        interp_y = scipy.interpolate.interp1d(timestamps, traj[:, 1])
        interp_z = scipy.interpolate.interp1d(timestamps, traj[:, 2])
        
        rots = scipy.spatial.transform.Rotation.from_quat(traj[:, 3:])
        interp_q = scipy.spatial.transform.Slerp(timestamps, rots)

        #get a registered trajectory for each map.
        for i, map_features in enumerate(map_features_list):
            print('{}/{}'.format(i+1, len(map_features_list)), end='\r')
            map_feature_data = []
            map_feature_keys = []
            mf_nx = int(map_features.info.length_x/map_features.info.resolution)
            mf_ny = int(map_features.info.length_y/map_features.info.resolution)
            for k,v in zip(map_features.layers, map_features.data):
                #temp hack bc I don't like this feature.
                if k == 'npts':
                    continue

                data = np.array(v.data).reshape(mf_nx, mf_ny)[::-1, ::-1]

                map_feature_keys.append(k)
                map_feature_data.append(data)

            map_feature_data.append(np.ones([mf_nx, mf_ny]))
            map_feature_keys.append('bias')

            map_feature_data = np.stack(map_feature_data, axis=0)
            map_feature_data[~np.isfinite(map_feature_data)] = self.fill_value

            start_time = map_features.info.header.stamp.to_sec()
            targets = start_time + np.arange(self.horizon) * self.dt
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

#            if (i%50) == 0:
#                idx = map_feature_keys.index('height_high')
#                plt.title(map_feature_keys[idx])
#                plt.imshow(map_feature_data[idx], origin='lower', extent=(xmin, xmax, ymin, ymax))
#                plt.plot(traj[:, 0], traj[:, 1], c='r')
#                plt.show()

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

    def __len__(self):
        return len(self.dataset['traj'])

    def __getitem__(self, idx):
        data = {k:v[idx] for k,v in self.dataset.items()}

        return data

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    bag_fp = '../../debug_data.bag'
    config_fp = '/home/yamaha/physics_atv_ws/configs/yamaha_atv/atv_model.yaml'

    dataset = MaxEntIRLDataset(bag_fp=bag_fp, config_fp=config_fp)

    dl = DataLoader(dataset, batch_size=32, shuffle=True)

    batch = next(iter(dl))
    for i in range(32):
        map_feature_data = batch['map_features'][i]
        traj = batch['traj'][i]
        metadata = {k:v[i] for k,v in batch['metadata'].items()}

        idx = dataset.feature_keys.index('height_high')
        xmin = metadata['origin'][0]
        xmax = xmin + metadata['width']
        ymin = metadata['origin'][1]
        ymax = ymin + metadata['height']

        plt.title(dataset.feature_keys[idx])
        plt.imshow(map_feature_data[idx], origin='lower', extent=(xmin, xmax, ymin, ymax))
        plt.plot(traj[:, 0], traj[:, 1], c='r')
        plt.show()
