import numpy as np
import torch
from torch import nn

from maxent_irl_costmaps.networks.mlp import MLP

"""
A collection of basic CNN blocks to try.
"""

class ResnetCostmapCNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, hidden_activation=nn.Tanh, dropout=0.0, device='cpu'):
        """
        Args:
            in_channels: The number of channels in the input image
            out_channels: The number of channels in the output image
            hidden_channels: A list containing the intermediate channels

        Note that in contrast to regular resnet, there is no end MLP nor pooling
        """
        super(ResnetCostmapCNN, self).__init__()
        self.channel_sizes = [in_channels] + hidden_channels + [out_channels]

        self.cnn = nn.ModuleList()
        for i in range(len(self.channel_sizes) - 2):
            self.cnn.append(ResnetCostmapBlock(in_channels=self.channel_sizes[i], out_channels=self.channel_sizes[i+1], activation=hidden_activation))

        #last conv to avoid activation
        self.cnn.append(nn.Conv2d(in_channels=self.channel_sizes[-2], out_channels=self.channel_sizes[-1], kernel_size=1, bias=False))
        self.cnn = torch.nn.Sequential(*self.cnn)

    def forward(self, x):
        cnn_out = self.cnn.forward(x)
        return cnn_out.sigmoid()

class ResnetCostmapBlock(nn.Module):
    """
    A ResNet-style block that does VGG + residual. Like the VGG-style block, output size is half of input size.
    In contrast to the original resnet block, don't use pooling or batch norm.
    """
    def __init__(self, in_channels, out_channels, activation):
        super(ResnetCostmapBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.activation = activation()
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        _x = x
        _x = self.conv1(_x)
        _x = self.activation(_x)
        _x = self.conv2(_x)
        res = self.projection(x)
        _x = self.activation(_x + res)
        return _x
