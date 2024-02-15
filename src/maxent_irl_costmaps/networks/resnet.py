import numpy as np
import torch
from torch import nn

from maxent_irl_costmaps.networks.mlp import MLP
from maxent_irl_costmaps.networks.misc import ScaledSigmoid, Exponential, AddMin

"""
A collection of basic CNN blocks to try.
"""
class ResnetCostmapCategoricalSpeedmapCNNEnsemble2(nn.Module):
    def __init__(self, in_channels, hidden_channels, speed_nbins=20, max_speed=10., ensemble_dim=100, hidden_activation='tanh', dropout=0.0, activation_type='sigmoid', activation_scale=1.0, device='cpu'):
        """
        Args:
            in_channels: The number of channels in the input image
            out_channels: The number of channels in the output image
            hidden_channels: A list containing the intermediate channels

        Note that in contrast to regular resnet, there is no end MLP nor pooling

        Same as the first ensemble, but now make the first layer the ensemble
        """
        super(ResnetCostmapCategoricalSpeedmapCNNEnsemble2, self).__init__()
        self.channel_sizes = [in_channels] + hidden_channels + [1]

        if hidden_activation == 'tanh':
            hidden_activation = nn.Tanh
        elif hidden_activation == 'relu':
            hidden_activation = nn.ReLU

        self.ensemble_dim = ensemble_dim

        self.max_speed = max_speed
        self.speed_nbins = speed_nbins
        self.speed_bins = torch.linspace(0., self.max_speed, self.speed_nbins + 1)

        self.cnn = nn.ModuleList()
        for i in range(len(self.channel_sizes) - 2):
            if i == 0:
                self.cnn_base = nn.ModuleList([ResnetCostmapBlock(in_channels=self.channel_sizes[i], out_channels=self.channel_sizes[i+1], activation=hidden_activation, dropout=dropout) for _ in range(self.ensemble_dim)])
            else:
                self.cnn.append(ResnetCostmapBlock(in_channels=self.channel_sizes[i], out_channels=self.channel_sizes[i+1], activation=hidden_activation, dropout=dropout))

        #last conv to avoid activation (for cost head)
        self.cost_head = nn.Conv2d(in_channels=self.channel_sizes[-2], out_channels=1, kernel_size=1, bias=True)
        self.speed_head = nn.Conv2d(in_channels=self.channel_sizes[-2], out_channels=self.speed_nbins, kernel_size=1, bias=True)

        self.cnn = torch.nn.Sequential(*self.cnn)

        if activation_type == 'sigmoid':
            self.activation = ScaledSigmoid(scale=activation_scale)
        elif activation_type == 'exponential':
            self.activation = Exponential(scale=activation_scale)
        elif activation_type == 'addmin':
            self.activation = AddMin()
        elif activation_type == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_type == 'none':
            self.activation = nn.Identity()

    def forward(self, x, return_features=True):
        idx = torch.randint(self.ensemble_dim, size=(1, ))
        base_layer = self.cnn_base[idx]

        features = self.cnn.forward(base_layer.forward(x))
        costmap = self.activation(self.cost_head(features))
        speed_logits = self.speed_head(features)

        #exponentiate the mean value too, as speeds are always positive
        return {
                    'costmap': costmap,
                    'speedmap': speed_logits,
                    'features': features
                }

    def ensemble_forward(self, x, return_features=True):
        features_batch = torch.stack([layer.forward(x) for layer in self.cnn_base], dim=-4)

        #have to reshape for cnn
        data_dims = features_batch.shape[-3:]
        batch_dims = features_batch.shape[:-3]
        features_batch_flat = features_batch.view(-1, *data_dims)

        features = self.cnn.forward(features_batch_flat)

        #unsqueeze to make [B x E x C x H x W]
        costmap = self.activation(self.cost_head(features)).view(*batch_dims, 1, *data_dims[1:])
        speed_logits = self.speed_head(features).view(*batch_dims, self.speed_nbins, *data_dims[1:])

        return {
                    'costmap': costmap,
                    'speedmap': speed_logits,
                    'features': features
                }

class ResnetCostmapSpeedmapCNNEnsemble2(nn.Module):
    def __init__(self, in_channels, hidden_channels, ensemble_dim=100, hidden_activation='tanh', dropout=0.0, activation_type='sigmoid', activation_scale=1.0, device='cpu'):
        """
        Args:
            in_channels: The number of channels in the input image
            out_channels: The number of channels in the output image
            hidden_channels: A list containing the intermediate channels

        Note that in contrast to regular resnet, there is no end MLP nor pooling

        Same as the first ensemble, but now make the first layer the ensemble
        """
        super(ResnetCostmapSpeedmapCNNEnsemble2, self).__init__()
        self.channel_sizes = [in_channels] + hidden_channels + [1]

        if hidden_activation == 'tanh':
            hidden_activation = nn.Tanh
        elif hidden_activation == 'relu':
            hidden_activation = nn.ReLU

        self.ensemble_dim = ensemble_dim

        self.cnn = nn.ModuleList()
        for i in range(len(self.channel_sizes) - 2):
            if i == 0:
                self.cnn_base = nn.ModuleList([ResnetCostmapBlock(in_channels=self.channel_sizes[i], out_channels=self.channel_sizes[i+1], activation=hidden_activation, dropout=dropout) for _ in range(self.ensemble_dim)])
            else:
                self.cnn.append(ResnetCostmapBlock(in_channels=self.channel_sizes[i], out_channels=self.channel_sizes[i+1], activation=hidden_activation, dropout=dropout))

        #last conv to avoid activation (for cost head)
        self.cost_head = nn.Conv2d(in_channels=self.channel_sizes[-2], out_channels=1, kernel_size=1, bias=True)
        self.speed_head = nn.Conv2d(in_channels=self.channel_sizes[-2], out_channels=2, kernel_size=1, bias=True)

        self.cnn = torch.nn.Sequential(*self.cnn)

        if activation_type == 'sigmoid':
            self.activation = ScaledSigmoid(scale=activation_scale)
        elif activation_type == 'exponential':
            self.activation = Exponential(scale=activation_scale)
        elif activation_type == 'addmin':
            self.activation = AddMin()
        elif activation_type == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_type == 'none':
            self.activation = nn.Identity()

    def forward(self, x, return_features=True):
        idx = torch.randint(self.ensemble_dim, size=(1, ))
        base_layer = self.cnn_base[idx]

        features = self.cnn.forward(base_layer.forward(x))
        costmap = self.activation(self.cost_head(features))
        speed_logits = self.speed_head(features)

        #exponentiate the mean value too, as speeds are always positive
        speed_dist = torch.distributions.Normal(loc=speed_logits[...,0, :, :].exp(), scale=(speed_logits[..., 1, :, :].exp() + 1e-6))

        return {
                    'costmap': costmap,
                    'speedmap': speed_dist,
                    'features': features
                }

    def ensemble_forward(self, x, return_features=True):
        features_batch = torch.stack([layer.forward(x) for layer in self.cnn_base], dim=-4)

        #have to reshape for cnn
        data_dims = features_batch.shape[-3:]
        batch_dims = features_batch.shape[:-3]
        features_batch_flat = features_batch.view(-1, *data_dims)

        features = self.cnn.forward(features_batch_flat)

        #unsqueeze to make [B x E x C x H x W]
        costmap = self.activation(self.cost_head(features)).view(*batch_dims, 1, *data_dims[1:])
        speed_logits = self.speed_head(features).view(*batch_dims, 2, *data_dims[1:])

        #exponentiate the mean value too, as speeds are always positive
        speed_dist = torch.distributions.Normal(loc=speed_logits[..., 0, :, :].exp(), scale=(speed_logits[..., 1, :, :].exp() + 1e-6))

        return {
                    'costmap': costmap,
                    'speedmap': speed_dist,
                    'features': features
                }

class LinearCostmapSpeedmapEnsemble2(nn.Module):
    """
    Handle the linear case separately
    """
    def __init__(self, in_channels, ensemble_dim=100, dropout=0.0, activation_type='sigmoid', activation_scale=1.0, device='cpu'):
        """
        Args:
            in_channels: The number of channels in the input image
            out_channels: The number of channels in the output image
            hidden_channels: A list containing the intermediate channels

        Note that in contrast to regular resnet, there is no end MLP nor pooling

        Same as the first ensemble, but now make the first layer the ensemble
        """
        super(LinearCostmapSpeedmapEnsemble2, self).__init__()
        self.channel_sizes = [in_channels, 1]

        self.ensemble_dim = ensemble_dim

        #last conv to avoid activation (for cost head)
        self.cost_heads = nn.ModuleList([nn.Conv2d(in_channels=self.channel_sizes[-2], out_channels=1, kernel_size=1, bias=True) for _ in range(self.ensemble_dim)])
        self.speed_heads = nn.ModuleList([nn.Conv2d(in_channels=self.channel_sizes[-2], out_channels=2, kernel_size=1, bias=True) for _ in range(self.ensemble_dim)])

        if activation_type == 'sigmoid':
            self.activation = ScaledSigmoid(scale=activation_scale)
        elif activation_type == 'exponential':
            self.activation = Exponential(scale=activation_scale)
        elif activation_type == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_type == 'addmin':
            self.activation = AddMin()
        elif activation_type == 'none':
            self.activation = nn.Identity()

    def forward(self, x, return_features=True):
        features = x
        idx = torch.randint(self.ensemble_dim, size=(1, ))
        cost_head = self.cost_heads[idx]
        speed_head = self.speed_heads[idx]

        costmap = self.activation(cost_head(features))
        speed_logits = speed_head(features)

        #exponentiate the mean value too, as speeds are always positive
        speed_dist = torch.distributions.Normal(loc=speed_logits[...,0, :, :].exp(), scale=(speed_logits[..., 1, :, :].exp() + 1e-6))

        return {
                    'costmap': costmap,
                    'speedmap': speed_dist,
                    'features': features
                }

    def ensemble_forward(self, x, return_features=True):
        features = x
        costmap = torch.stack([self.activation(f.forward(features)) for f in self.cost_heads], dim=-4)
        speed_logits = torch.stack([f.forward(features) for f in self.speed_heads], dim=-3)

        #exponentiate the mean value too, as speeds are always positive
        speed_dist = torch.distributions.Normal(loc=speed_logits[..., 0, :, :].exp(), scale=(speed_logits[..., 1, :, :].exp() + 1e-6))

        return {
                    'costmap': costmap,
                    'speedmap': speed_dist,
                    'features': features
                }

class ResnetCostmapBlock(nn.Module):
    """
    A ResNet-style block that does VGG + residual. Like the VGG-style block, output size is half of input size.
    In contrast to the original resnet block, don't use pooling or batch norm.
    """
    def __init__(self, in_channels, out_channels, activation, dropout=0.0):
        super(ResnetCostmapBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.activation = activation()
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bnorm1 = nn.BatchNorm2d(in_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        _x = x
        _x = self.conv1(_x)
        _x = self.bnorm1(_x)
        _x = self.activation(_x)
        _x = self.dropout(_x)
        _x = self.conv2(_x)
        res = self.projection(x)
        _x = self.bnorm2(_x)
        _x = self.activation(_x + res)
        _x = self.dropout(_x)
        return _x
