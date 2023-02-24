import torch
from torch import nn
import numpy as np

from maxent_irl_costmaps.networks.pointpillars.voxelizer2 import pillarize
from maxent_irl_costmaps.networks.mlp import MLP
from maxent_irl_costmaps.networks.resnet import ResnetCostmapCNN

class PointPillarsCostmap(nn.Module):
    """
    Use PointPillars (Lang 2019) as a backend to generate costmaps directly from pointclouds.
    """
    def __init__(
                    self,
                    pointnet_hidden_channels=[],
                    pointnet_activation=nn.Tanh,
                    pointnet_features=64,
                    resnet_hidden_channels=[],
                    resnet_hidden_activation=nn.Tanh,
                    dropout=0.0,
                    activation_type='relu',
                    activation_scale=1.0,
                    max_npillars=5000,
                    max_npoints=100,
                    device='cpu'
                ):
        super(PointPillarsCostmap, self).__init__()

        self.pointnet = MLP(insize=8, outsize=pointnet_features, hiddens=pointnet_hidden_channels, hidden_activation=pointnet_activation, device=device)

        self.resnet = ResnetCostmapCNN(in_channels=pointnet_features, out_channels=1, hidden_channels=resnet_hidden_channels, hidden_activation=resnet_hidden_activation, activation_type=activation_type, activation_scale=activation_scale)

        self.pointnet_features = pointnet_features
        self.max_npillars = max_npillars
        self.max_npoints = max_npoints
        self.device = device

    def forward(self, pillars, pillar_idxs, metadata):
        """
        TODO normalize the pillars and vectorize
        """
        nx = int(metadata['length_x'] / metadata['resolution'])
        ny = int(metadata['length_y'] / metadata['resolution'])

        valid_mask = (torch.linalg.norm(pillars, dim=-1, keepdims=True) < 1e6).float()
        pillar_features = self.pointnet.forward(pillars)
        pillar_max_features = torch.max(pillar_features * valid_mask, dim=-2)[0]

        #start channels-last for easy scatter
        pseudo_image = torch.zeros(*pillars.shape[:-3], nx, ny, self.pointnet_features, device=self.device)

        #ok I don't have the broadcasting line off the top of my head
        for b in range(pillars.shape[0]):
            pseudo_image[b, pillar_idxs[b, :, 0], pillar_idxs[b, :, 1]] = pillar_max_features[b]

        pseudo_image = torch.moveaxis(pseudo_image, -1, -3)

        out = self.resnet.forward(pseudo_image)
        return out

    def to(self, device):
        self.device = device
        self.pointnet = self.pointnet.to(device)
        self.resnet = self.resnet.to(device)

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
    nx = int(metadata['length_x'] / metadata['resolution'])
    ny = int(metadata['length_y'] / metadata['resolution'])

    pointpillars = PointPillarsCostmap(pointnet_hidden_channels=[], resnet_hidden_channels=[32, ], activation_type='exponential')
    opt = torch.optim.Adam(pointpillars.parameters())
    pillar_list = []
    pillar_idx_list = []

    print(sum([x.numel() for x in pointpillars.parameters()]))

    #try a very simple regression to max height
    cnt = 0
    for topic, msg, t in bag.read_messages(topics=[pcl_topic, pose_topic]):
        if topic == pose_topic:
            pose = msg.pose

        if topic == pcl_topic and pose is not None:
            #dumb rosbag hack
            if cnt < 100:
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
                pillars, pillar_idxs = pillarize(pcl_tensor, pos_tensor, metadata, max_npillars=10000, max_npoints=128)
                maxheight_label = torch.zeros(nx, ny)
                pillar_maxheight = pillars[:, 2].max(dim=-1)[0]
                maxheight_label[pillar_idxs[:, 0], pillar_idxs[:, 1]] = pillar_maxheight

                res = pointpillars.forward(pillars.unsqueeze(0), pillar_idxs.unsqueeze(0), metadata)
                pred_maxheight = res['costmap'][0, 0]

                loss = (maxheight_label - pred_maxheight)[pillar_idxs[:, 0], pillar_idxs[:, 1]].pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step() 
                print('LOSS: {:.6f}'.format(loss.detach().item()))
                cnt += 1
            else:
                break

    #test
    plt.show(block=False)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for topic, msg, t in bag.read_messages(topics=[pcl_topic, pose_topic]):
        if topic == pose_topic:
            pose = msg.pose

        if topic == pcl_topic and pose is not None:
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
            pillars, pillar_idxs = pillarize(pcl_tensor, pos_tensor, metadata, max_npillars=10000, max_npoints=128)
            maxheight_label = torch.zeros(nx, ny)
            pillar_maxheight = pillars[:, 2].max(dim=-1)[0]
            maxheight_label[pillar_idxs[:, 0], pillar_idxs[:, 1]] = pillar_maxheight

            t1 = time.time()
            with torch.no_grad():
                res = pointpillars.forward(pillars.unsqueeze(0), pillar_idxs.unsqueeze(0), metadata)
            t2 = time.time()

            pred_maxheight = res['costmap'][0, 0]
            for ax in axs:
                ax.cla()
            axs[0].imshow(maxheight_label, vmin=0., vmax=50.)
            axs[1].imshow(pred_maxheight, vmin=0., vmax=50.)
            axs[0].set_title('GT')
            axs[1].set_title('Pred')
            fig.suptitle('Time = {:.4f}s'.format(t2-t1))
            plt.pause(1e-1)
