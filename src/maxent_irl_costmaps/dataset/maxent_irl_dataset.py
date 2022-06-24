import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped

from maxent_irl_costmaps.preprocess import load_data
from maxent_irl_costmaps.utils import dict_to

class MaxEntIRLDataset(Dataset):
    """
    Dataset for maxent IRL on costmaps that contains the following:
        1. A set of map features (plus metadata) extracted from rosbags
        2. A trajectory (registered on the map)

    Ok, the ony diff now is that there are multiple bag files and we save the trajdata to a temporary pt file.
    """
    def __init__(self, bag_fp, preprocess_fp, map_features_topic='/local_gridmap', odom_topic='/integrated_to_init', image_topic='/multisense/left/image_rect_color', horizon=70, dt=0.1, fill_value=0.):
        """
        Args:
            bag_fp: The bag to get data from
            config_fp: Config file describing model input (we only listen to state)
            map_features_topic: the topic to read map features from
            odom_topic: the topic to read trajectories from
            horizon: The number of trajectory steps to register onto each map

        Note that since the map changes here, we want to treat each as an MDP and make data samples for each map.
        """
        self.bag_fp = bag_fp
        self.preprocess_fp = preprocess_fp
        self.map_features_topic = map_features_topic
        self.odom_topic = odom_topic
        self.image_topic = image_topic
        self.horizon = horizon
        self.dt = dt
        self.fill_value = fill_value #I don't know if this is the best way to do this, but setting the fill value to 0 implies that missing features contribute nothing to the cost.
        self.N = 0
        self.device = 'cpu'

        self.initialize_dataset()

    def initialize_dataset(self):
        """
        Profile the trajectories in the bag to:
            1. Create torch files to reference later
            2. Index each sample of the dataset
            3. Get map feature statistics for normalization
        """
        preprocess = False
        if os.path.exists(self.preprocess_fp):
            x = input('{} already exists. Preprocess again? [y/N]'.format(self.preprocess_fp))
            preprocess = (x.lower() == 'y')
        else:
            os.mkdir(self.preprocess_fp)
            preprocess = True

        print('Preprocess = ', preprocess)

        #save all the data into individual files
        if preprocess:
            for tfp in os.listdir(self.bag_fp):
                raw_fp = os.path.join(self.bag_fp, tfp)
                data, feature_keys = load_data(raw_fp, self.map_features_topic, self.odom_topic, self.image_topic, self.horizon, self.dt, self.fill_value)
                if data is None:
                    continue
                for i in range(len(data['traj'])):
                    pp_fp = os.path.join(self.preprocess_fp, 'traj_{}.pt'.format(self.N))
                    subdata = {k:v[i] for k,v in data.items()}
                    subdata['feature_keys'] = feature_keys

                    #make heights relative.
                    for k in feature_keys:
                        if k in ['height_low', 'height_mean', 'height_high', 'height_max', 'terrain']:
                            idx = feature_keys.index(k)
                            mask = abs(subdata['map_features'][idx] - self.fill_value) < 1e-6
                            subdata['map_features'][idx][~mask] -= subdata['traj'][0, 2]

                    torch.save(subdata, pp_fp)
                    self.N += 1
        
        #Actually read all the data to get statistics.
        #need number of trajs, and the mean/std of all the map features.
        self.feature_keys = torch.load(os.path.join(self.preprocess_fp, 'traj_0.pt'))['feature_keys']
        self.metadata = torch.load(os.path.join(self.preprocess_fp, 'traj_0.pt'))['metadata']
        nfeats = len(self.feature_keys)
        self.feature_mean = torch.zeros(nfeats)
        self.feature_var = torch.zeros(nfeats)

        traj_fps = os.listdir(self.preprocess_fp)
        self.N = len(traj_fps)
        K = 1
        for i, fp in enumerate(traj_fps):
            print('{}/{} ({})'.format(i+1, len(traj_fps), fp), end='\r')
            traj = torch.load(os.path.join(self.preprocess_fp, fp))

            mapfeats = traj['map_features'][:, 10:-10, 10:-10].reshape(nfeats, -1)
            k = mapfeats.shape[-1]

            mean_new = (self.feature_mean + (mapfeats.sum(dim=-1) / K)) * (K / (K+k))
            x_ssd = (mapfeats - mean_new.view(-1, 1)).pow(2).sum(dim=-1)
            mean_diff = self.feature_mean - mean_new
            x_correction = (mapfeats - mean_new.view(-1, 1)).sum(dim=-1)
            d_ssd = x_ssd + mean_diff * x_correction
            var_new = (self.feature_var + (d_ssd / K)) * (K / (K+k))

            self.feature_mean = mean_new
            self.feature_var = var_new
            K += k

        self.feature_std = self.feature_var.sqrt() + 1e-6
        self.feature_std[~torch.isfinite(self.feature_std)] = 1e-6

    def visualize(self):
        """
        Get a rough sense of features
        """
        n_panes = len(self.feature_keys) + 1

        nx = int(np.sqrt(n_panes))
        ny = int(n_panes / nx) + 1

        fig, axs = plt.subplots(nx, ny, figsize=(20, 20))
        axs = axs.flatten()

        data = self[np.random.randint(len(self))]
        traj = data['traj']
        feats = data['map_features']
        metadata = data['metadata']
        xmin = metadata['origin'][0].cpu()
        ymin = metadata['origin'][1].cpu()
        xmax = xmin + metadata['width']
        ymax = ymin + metadata['height']

        
        for ax, feat, feat_key in zip(axs, feats, self.feature_keys):
            ax.imshow(feat.cpu(), origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))
            ax.plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='y')
            ax.set_title(feat_key)

        if 'image' in data.keys():
            image =  data['image']
            ax = axs[len(self.feature_keys)]
            ax.imshow(image.permute(1, 2, 0).cpu())
            ax.set_title('Image')

        return fig, axs

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        data_fp = os.path.join(self.preprocess_fp, 'traj_{}.pt'.format(idx))

        data = torch.load(data_fp, map_location=self.device)

        data['map_features'] = ((data['map_features'] - self.feature_mean.view(-1, 1, 1)) / self.feature_std.view(-1, 1, 1)).clip(-5, 5)

        return data

    def to(self, device):
        self.device = device
        self.feature_mean = self.feature_mean.to(self.device)
        self.feature_std = self.feature_std.to(self.device)
        return self

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    bag_fp = '/home/yamaha/Desktop/datasets/yamaha_maxent_irl/rosbags/'
    pp_fp = '/home/yamaha/Desktop/datasets/yamaha_maxent_irl/torch/'

    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=pp_fp)

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
