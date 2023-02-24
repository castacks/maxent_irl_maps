"""
A vain attempt to create something from nothing

Algo is as follows:
    1. Take a costmap from the IRL dataset
    2. Find the completed equivalent in the buffer
    3. Train a Unet to predict the completed map from the partial one
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_costmaps.networks.unet import UNet

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.dataset.global_costmap import GlobalCostmap

def quat_to_yaw(q):
    qx, qy, qz, qw = q.moveaxis(-1, 0)
    return torch.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))

class CostmapEnhancer:
    """
    I used the costmap to predict the costmap
    """
    def __init__(self, enhance_net, enhance_opt, costmap_net, global_costmap, dataset, batch_size=32):
        """
        Args:
            enhance_net: the net to predict completed costmaps
            enhance_opt: optimizer for enhance_net
            costmap_net: the base network that predicts costmaps
            global_costmap: the costmap to query for completed costmaps
            dataset: the dataet to train on
        """
        self.enhance_net = enhance_net
        self.enhance_opt = enhance_opt
        self.costmap_net = costmap_net
        self.global_costmap = global_costmap
        self.dataset = dataset

        self.batch_size=batch_size
        self.itr = 0

    def update(self, n=-1):
        self.itr += 1
        dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for i, batch in enumerate(dl):
            if n > -1 and i >= n:
                break

            #skip the last batch in the dataset as MPPI batching forces a fixed size
            if batch['traj'].shape[0] < self.batch_size:
                break

#            print('{}/{}'.format(i+1, int(len(self.dataset)/self.batch_size)), end='\r')
            self.gradient_step(batch)

        print('_____ITR {}_____'.format(self.itr))

    def gradient_step(self, batch):
        #first get the partial costmap
        with torch.no_grad():
            costmap_in = self.costmap_net.ensemble_forward(batch['map_features'])['costmap'].mean(dim=1)

        #then get the net prediction
        costmap_out = self.enhance_net.forward(costmap_in)

        #now get labels
        gps_poses = torch.zeros(self.batch_size, 3)
        gps_poses[:, :2] = batch['gps_traj'][:, 0, :2]
        yaw_offset = quat_to_yaw(batch['traj'][:, 0, 3:7]) - quat_to_yaw(batch['gps_traj'][:, 0, 3:7])
        gps_poses[:, 2] = -yaw_offset

        crop_params = {
            'origin': batch['metadata']['origin'][0] - batch['traj'][0, 0, :2],
            'length_x':batch['metadata']['height'][0],
            'length_y':batch['metadata']['width'][0],
            'resolution':batch['metadata']['resolution'][0]
        }

        costmap_gt = self.global_costmap.get_costmap(gps_poses, crop_params, local=True).swapaxes(-2, -1).to(costmap_out.device)

        loss = (costmap_out[:, 0] - costmap_gt).pow(2).mean()

        self.enhance_opt.zero_grad()
        loss.backward()
        self.enhance_opt.step()

        print('loss = {:.6f}'.format(loss.detach().item()), end='\r')

        """
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(costmap_in[0, 0].detach().cpu())
        axs[1].imshow(costmap_out[0, 0].detach().cpu())
        axs[2].imshow(costmap_gt[0].detach().cpu())
        axs[0].set_title('partial costmap')
        axs[1].set_title('predicted costmap')
        axs[2].set_title('gt costmap')
        plt.show()
        """

    def visualize(self, i=-1):
        if i < 0:
            idx = np.random.randint(len(self.dataset))
        else:
            idx = i

        batch = dataset[idx]
        with torch.no_grad():
            costmap_in = self.costmap_net.ensemble_forward(batch['map_features'].unsqueeze(0))['costmap'].mean(dim=1)

            #then get the net prediction
            costmap_out = self.enhance_net.forward(costmap_in)

            #now get labels
            gps_poses = torch.zeros(1, 3)
            gps_poses[:, :2] = batch['gps_traj'][0, :2]
            yaw_offset = quat_to_yaw(batch['traj'][[0], 3:7]) - quat_to_yaw(batch['gps_traj'][[0], 3:7])
            gps_poses[0, 2] = -yaw_offset.item()

            crop_params = {
                'origin': batch['metadata']['origin'] - batch['traj'][0, :2],
                'length_x':batch['metadata']['height'],
                'length_y':batch['metadata']['width'],
                'resolution':batch['metadata']['resolution']
            }

            costmap_gt = self.global_costmap.get_costmap(gps_poses, crop_params, local=True).swapaxes(-2, -1).to(costmap_out.device)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(costmap_in[0, 0].detach().cpu())
            axs[1].imshow(costmap_out[0, 0].detach().cpu())
            axs[2].imshow(costmap_gt[0].detach().cpu())
            axs[0].set_title('partial costmap')
            axs[1].set_title('predicted costmap')
            axs[2].set_title('gt costmap')
            plt.show()

if __name__ == '__main__':
    model_fp = '../../../models/yamaha/icra_resnet.pt'
    global_costmap_fp = '../../../models/state_visitations/fool.pt'
    bag_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train/'
    pp_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_train_h75/'

    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=pp_fp).to('cuda')

    model = torch.load(model_fp)
    model.network.eval()
    network = model.network.to('cuda')
    global_costmap = torch.load(global_costmap_fp)

    #TODO: explore whether better to go from costmap or map features
    metadata = model.expert_dataset.metadata
    nx = int(metadata['height']/metadata['resolution'])
    ny = int(metadata['width']/metadata['resolution'])

    unet = UNet([1, nx, ny], [1, nx, ny]).to('cuda')
    opt = torch.optim.Adam(unet.parameters())
    print('unet params: {}'.format(sum([x.numel() for x in unet.parameters()])))

    trainer = CostmapEnhancer(unet, opt, network, global_costmap, dataset, batch_size=4)
    
    for i in range(5):
        trainer.update()

    for i in range(100):
        trainer.visualize()
