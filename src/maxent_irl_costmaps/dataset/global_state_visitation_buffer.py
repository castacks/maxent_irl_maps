"""
Dataset for registering a lot of runs onto a global map
We accomplish this by using GPS as a common reference frame and storing an
occupancy grid of all states

Note that this class is not a dataset and is just the occupancy grid and its
relevant methods
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from maxent_irl_costmaps.preprocess import load_traj
from maxent_irl_costmaps.utils import quat_to_yaw
from maxent_irl_costmaps.os_utils import walk_bags

class GlobalStateVisitationBuffer:
    def __init__(self, config_fp, bag_dir, gps_topic='/odometry/filtered_odom', dt=0.05, device='cpu'):
        """
        Args:
            config_fp: yaml file containing metadata for the state visitation map
            bag_dir: The dir to parse for rosbags
            resolution: The discretization of the map
            dt: The dt for the trajectory
        """
        self.config_fp = config_fp
        self.base_fp = bag_dir
        self.gps_topic = gps_topic
        self.dt = dt

        self.traj_fps = np.array(walk_bags(self.base_fp))
        config_data = yaml.safe_load(open(self.config_fp, 'r'))
        self.metadata = {
            'origin': torch.tensor([config_data['origin']['x'], config_data['origin']['y']]),
            'length_x': config_data['length_x'],
            'length_y': config_data['length_y'],
            'resolution': config_data['resolution']
        }
        self.initialize()
        self.device = device

    def initialize(self):
        """
        Set map bounds and occgrid
        """
        nx = int(self.metadata['length_x'] / self.metadata['resolution'])
        ny = int(self.metadata['length_y'] / self.metadata['resolution'])
        self.data = torch.zeros(nx, ny).long()
        xmin = self.metadata['origin'][0].item()
        xmax = xmin + self.metadata['length_x']
        ymin = self.metadata['origin'][1].item()
        ymax = ymin + self.metadata['length_y']

        #first pass to sanitize data
        good_traj_fps = []
        for i, tfp in enumerate(self.traj_fps):
            print('{}/{} ({})'.format(i+1, len(self.traj_fps), os.path.basename(tfp)), end='\r')
            try:
                traj = load_traj(tfp, self.gps_topic, self.dt)
            except:
                print("Couldn't load {}, skipping...".format(tfp))
                good_traj_fps.append(False)
                continue
            #check for gps jumps 
            diffs = np.linalg.norm(traj[1:, :3] - traj[:-1, :3], axis=-1)
            if any(diffs > self.dt * 50.):
                print('Jump in ({}) > 50m/s. skipping...'.format(tfp))
                good_traj_fps.append(False)

            elif traj[:, 0].min() < xmin:
                print('Traj {} x {:.2f} less than xmin {:.2f}, skipping...'.format(tfp, traj[:, 0].min(), xmin))
                good_traj_fps.append(False)

            elif traj[:, 0].max() > xmax:
                print('Traj {} x {:.2f} more than xmax {:.2f}, skipping...'.format(tfp, traj[:, 0].max(), xmax))
                good_traj_fps.append(False)

            elif traj[:, 1].min() < ymin:
                print('Traj {} y {:.2f} less than ymin {:.2f}, skipping...'.format(tfp, traj[:, 1].min(), ymin))
                good_traj_fps.append(False)

            elif traj[:, 1].max() > ymax:
                print('Traj {} y {:.2f} more than ymax {:.2f}, skipping...'.format(tfp, traj[:, 1].max(), ymax))
                good_traj_fps.append(False)

            else:
                good_traj_fps.append(True)


        self.traj_fps = self.traj_fps[good_traj_fps]

        for i, tfp in enumerate(self.traj_fps):
            print('{}/{} ({})'.format(i+1, len(self.traj_fps), os.path.basename(tfp)), end='\r')
            traj = load_traj(tfp, self.gps_topic, self.dt)

            #filter out stationary
            traj = traj[np.linalg.norm(traj[:, 7:10], axis=-1) > 1.0]

            #generate the occgrid
            txs = traj[:, 0]
            tys = traj[:, 1]

            gxs = ((txs - xmin) / self.metadata['resolution']).astype(np.int64)
            gys = ((tys - ymin) / self.metadata['resolution']).astype(np.int64)
            width = max(nx, ny)
            grid_hash = gxs * width + gys
            bins, cnts = np.unique(grid_hash, return_counts=True)
            bin_xs = bins // width
            bin_ys = bins % width

            self.data[bin_xs, bin_ys] += torch.from_numpy(cnts)

    def get_state_visitations(self, pose, crop_params, tf=None, local=True):
        """
        Args:
            pose: the (batched) pose to get the crop from [B x {7/13}]
            crop_params: the metadata of the crop to up/downsample to
            tf: optional arg: a (batched) 2d transform from the gps frame to the pose frame
                used to allow us to get GPS visitations from super odometry [B x 3]
            local: bool for whether to use the local or global rotation
                note that local  -> rotated to align with current pose x-forward
                          global -> rotated to align with pose base frame
        """
        if tf is None:
            query_pose = torch.from_numpy(pose)
        else:
            print("Sam hasnt done the tf math yet.")
            exit(1)

        poses = torch.stack([query_pose[:, 0], query_pose[:, 1], quat_to_yaw(query_pose[:, 3:7])], axis=-1)
        if not local:
            poses[:, 2] = 0

        crop_xs = torch.arange(crop_params['origin'][0], crop_params['origin'][0] + crop_params['length_x'], crop_params['resolution']).double().to(self.device)
        crop_ys = torch.arange(crop_params['origin'][1], crop_params['origin'][1] + crop_params['length_y'], crop_params['resolution']).double().to(self.device)
        crop_positions = torch.stack(torch.meshgrid(crop_xs, crop_ys, indexing="ij"), dim=-1) # HxWx2 tensor
        crop_nx = int(crop_params['length_x'] / crop_params['resolution'])
        crop_ny = int(crop_params['length_y'] / crop_params['resolution'])

        translations = poses[:, :2]  # Nx2 tensor, where each row corresponds to [x, y] position in metric space
        rotations = torch.stack([poses[:, 2].cos(), -poses[:, 2].sin(), poses[:, 2].sin(), poses[:, 2].cos()], dim=-1)  # Nx4 tensor where each row corresponds to [cos(theta), -sin(theta), sin(theta), cos(theta)]
        
        ## Reshape tensors to perform batch tensor multiplication. 

        # The goal is to obtain a tensor of size [B, H, W, 2], where B is the batch size, H and W are the dimensions fo the image, and 2 corresponds to the actual x,y positions. To do this, we need to rotate and then translate every pair of points in the meshgrid. In batch multiplication, only the last two dimensions matter. That is, usually we need the following dimensions to do matrix multiplication: (m,n) x (n,p) -> (m,p). In batch multiplication, the last two dimensions of each array need to line up as mentioned above, and the earlier dimensions get broadcasted (more details in the torch matmul page). Here, we will reshape rotations to have shape [B,1,1,2,2] where B corresponds to batch size, the two dimensions with size 1 are there so that we can broadcast with the [H,W] dimensions in crop_positions, and the last two dimensions with size 2 reshape the each row in rotations into a rotation matrix that can left multiply a position to transform it. The output of torch.matmul(rotations, crop_positions) will be a [B,H,W,2,1] tensor. We will reshape translations to be a [B,1,1,2,1] vector so that we can add it to this output and obtain a tensor of size [B,H,W,2,1], which we will then squeeze to obtain our final tensor of size [B,H,W,2]
        
        rotations = rotations.view(-1, 1, 1, 2, 2) #[B x 1 x 1 x 2 x 2]
        crop_positions = crop_positions.view(1, crop_nx, crop_ny, 2, 1) #[1 x H x W x 2 x 1]
        translations = translations.view(-1, 1, 1, 2, 1) #[B x 1 x 1 x 2 x 1]
        
        
        # Apply each transform to all crop positions (res = [B x H x W x 2])
        crop_positions_transformed = (torch.matmul(rotations, crop_positions) + translations).squeeze()

        # Obtain actual pixel coordinates
        map_origin = self.metadata['origin'].view(1, 1, 1, 2)
        pixel_coordinates = ((crop_positions_transformed - map_origin) / self.metadata['resolution']).long()

        pixel_coordinates_flipped = pixel_coordinates.swapaxes(-2,-3)

        # Obtain maximum and minimum values of map to later filter out of bounds pixels
        map_p_low = torch.tensor([0, 0]).to(self.device).view(1, 1, 1, 2)
        map_p_high = torch.tensor(self.data.shape).to(self.device).view(1, 1, 1, 2)
        invalid_mask = (pixel_coordinates_flipped < map_p_low).any(dim=-1) | (pixel_coordinates_flipped >= map_p_high).any(dim=-1)

        #Indexing method: set all invalid idxs to a valid one (i.e. 0), index, then mask out the results

        #TODO: Per-channel fill value
        fill_value = 0
        pixel_coordinates_flipped[invalid_mask] = 0

        pxlist_flipped = pixel_coordinates_flipped.reshape(-1,2)

        #[B x C x W x H]
        values = self.data[pxlist_flipped[:,0], pxlist_flipped[:,1]]  # Notice axes are flipped to account for terrain body centric coordinates.
        values = values.view(poses.shape[0], crop_nx, crop_ny)

        k1 = invalid_mask.float()
        values = ((1.-k1)*values + k1*fill_value).swapaxes(-1, -2)

        value_sums = values.sum(dim=-1, keepdims=True).sum(dim=-2, keepdims=True) + 1e-4
        value_dist = values / value_sums

        #normalize to make a proper distribution
        return value_dist
        
    def create_anim(self, save_to, local=False):
        """
        create an animation for viz purposes
        """
        traj = load_traj(np.random.choice(self.traj_fps), self.gps_topic, self.dt)
        traj = traj[np.linalg.norm(traj[:, 7:10], axis=-1) > 1.0]
        crop_params = {
            'origin': np.array([-30., -30.]),
            'length_x': 60.,
            'length_y': 60.,
            'resolution': 0.5
        }

        xmin = self.metadata['origin'][0]
        ymin = self.metadata['origin'][1]
        xmax = xmin + self.metadata['length_x']
        ymax = ymin + self.metadata['length_y']

        cxmin = crop_params['origin'][0]
        cymin = crop_params['origin'][1]
        cxmax = cxmin + crop_params['length_x']
        cymax = cymin + crop_params['length_y']

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#        plt.show(block=False)

        def get_frame(t, fig, axs):
            print(t, end='\r')

            for ax in axs:
                ax.cla()

            svs = self.get_state_visitations(traj[t:t+5], crop_params, local=local)
            dx = traj[t+5, 0] - traj[t, 0]
            dy = traj[t+5, 1] - traj[t, 1]
            n = np.sqrt(dx*dx + dy*dy)
            l = max(self.metadata['length_x'], self.metadata['length_y']) / 50.

            axs[0].imshow(self.data.T, origin='lower', extent = (xmin, xmax, ymin, ymax), vmin=0., vmax=5.)
            axs[0].arrow(traj[t, 0], traj[t, 1], l*dx/n, l*dy/n, color='r', head_width=l/2.)
            axs[1].imshow(svs[0].T, origin='lower', extent = (cxmin, cxmax, cymin, cymax), vmin=0., vmax=0.01)
            axs[1].arrow(0., 0., 5., 0., color='r', head_width=1.)

            axs[0].set_title('Global state visitations')
            axs[1].set_title('Local map of global visitations')
            axs[0].set_xlabel('X (UTM)')
            axs[0].set_ylabel('Y (UTM)')
            axs[1].set_xlabel('X (local)')
            axs[1].set_ylabel('Y (local)')

        anim = FuncAnimation(fig, lambda t: get_frame(t, fig, axs), frames = np.arange(traj.shape[0] - 10), interval=300*self.dt)

        plt.show()
#        anim.save(save_to)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True, help='path to global map config')
    parser.add_argument('--bag_dir', type=str, required=True, help='dir containing GPS for state visitations')
    parser.add_argument('--save_as', type=str, required=True, help='save buffer to this filepath')
    args = parser.parse_args()

    buf = GlobalStateVisitationBuffer(args.config_fp, args.bag_dir)

    save_fp = args.save_as if args.save_as[-3:] == '.pt' else args.save_as + '.pt'

    torch.save(buf, save_fp)
    buf.create_anim(save_to = 'aaa', local=True)

#    if not os.path.exists('gsv_figs'):
#        os.mkdir('gsv_figs')
#    for i in range(10):
#        buf.create_anim(save_to = 'gsv_figs/gsv_{}.mp4'.format(i+1), local=True)
