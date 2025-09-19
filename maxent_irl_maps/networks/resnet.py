import numpy as np
import torch
from torch import nn

from maxent_irl_maps.networks.mlp import MLP
from maxent_irl_maps.networks.misc import ScaledSigmoid, Exponential, AddMin
from maxent_irl_maps.utils import compute_map_mean_entropy

"""
A collection of basic CNN blocks to try.
"""
class ResnetCategorical(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        speed_nbins=30,
        max_speed=15.0,
        cost_nbins=30,
        max_cost=10.0,
        hidden_activation="tanh",
        hidden_kernel_size=3,
        dropout=0.0,
        device="cpu",
    ):
        import pdb;pdb.set_trace()

class ResnetExpCostCategoricalSpeed(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        speed_nbins=20,
        max_speed=10.0,
        cost_nbins=20,
        max_cost=10.0,
        hidden_activation="tanh",
        hidden_kernel_size=3,
        dropout=0.0,
        device="cpu",
    ):
        """
        Args:
            in_channels: The number of channels in the input image
            out_channels: The number of channels in the output image
            hidden_channels: A list containing the intermediate channels
        """
        super(ResnetExpCostCategoricalSpeed, self).__init__()
        self.channel_sizes = [in_channels] + hidden_channels + [1]
        self.hidden_kernel_size = hidden_kernel_size

        if hidden_activation == "tanh":
            hidden_activation = nn.Tanh
        elif hidden_activation == "relu":
            hidden_activation = nn.ReLU

        self.max_speed = max_speed
        self.speed_nbins = speed_nbins
        self.speed_bins = torch.linspace(0.0, self.max_speed, self.speed_nbins + 1, device=device)

        self.cnn = nn.ModuleList()
        for i in range(len(self.channel_sizes) - 2):
            self.cnn.append(
                ResnetCostmapBlock(
                    in_channels=self.channel_sizes[i],
                    out_channels=self.channel_sizes[i + 1],
                    activation=hidden_activation,
                    dropout=dropout,
                    kernel_size=self.hidden_kernel_size,
                )
            )

        # last conv to avoid activation (for cost head)
        self.cost_head = nn.Conv2d(
            in_channels=self.channel_sizes[-2],
            out_channels=1,
            kernel_size=1,
            bias=True
        )
        self.speed_head = nn.Conv2d(
            in_channels=self.channel_sizes[-2],
            out_channels=self.speed_nbins,
            kernel_size=1,
            bias=True,
        )

        self.cnn = torch.nn.Sequential(*self.cnn)

    def forward(self, x, return_mean_entropy=False):
        features = self.cnn.forward(x)
        log_cost = self.cost_head(features)
        speed_logits = self.speed_head(features)

        # exponentiate the mean value too, as speeds are always positive
        res = {"costmap": log_cost.exp(), "speed_logits": speed_logits}

        if return_mean_entropy:
            res["costmap_entropy"] = torch.zeros_like(log_cost)
            res["speedmap"], res["speedmap_entropy"] = compute_map_mean_entropy(speed_logits, self.speed_bins)

        return res

    def to(self, device):
        super().to(device)
        self.speed_bins = self.speed_bins.to(device)
        return self

class ResnetCategorical(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        speed_nbins=30,
        max_speed=15.0,
        cost_nbins=30,
        max_cost=10.0,
        hidden_activation="tanh",
        hidden_kernel_size=3,
        dropout=0.0,
        device="cpu",
    ):
        """
        Args:
            in_channels: The number of channels in the input image
            out_channels: The number of channels in the output image
            hidden_channels: A list containing the intermediate channels
        """
        super(ResnetCategorical, self).__init__()
        self.channel_sizes = [in_channels] + hidden_channels + [1]
        self.hidden_kernel_size = hidden_kernel_size

        if hidden_activation == "tanh":
            hidden_activation = nn.Tanh
        elif hidden_activation == "relu":
            hidden_activation = nn.ReLU

        self.max_cost = max_cost
        self.cost_nbins = cost_nbins
        self.cost_bins = torch.linspace(np.log(1e-2), np.log(self.max_cost), self.cost_nbins + 1, device=device)

        self.max_speed = max_speed
        self.speed_nbins = speed_nbins
        self.speed_bins = torch.linspace(0.0, self.max_speed, self.speed_nbins + 1, device=device)

        self.cnn = nn.ModuleList()
        for i in range(len(self.channel_sizes) - 2):
            self.cnn.append(
                ResnetCostmapBlock(
                    in_channels=self.channel_sizes[i],
                    out_channels=self.channel_sizes[i + 1],
                    activation=hidden_activation,
                    dropout=dropout,
                    kernel_size=self.hidden_kernel_size,
                )
            )

        self.cost_head = nn.ModuleList()
        self.speed_head = nn.ModuleList()

        self.cost_head.append(
            ResnetCostmapBlock(
                    in_channels=self.channel_sizes[-2],
                    out_channels=self.channel_sizes[-2],
                    activation=hidden_activation,
                    dropout=dropout,
                    kernel_size=1,
                )
            )

        self.cost_head.append(
            nn.Conv2d(
                in_channels=self.channel_sizes[-2],
                out_channels=self.cost_nbins,
                kernel_size=1,
                bias=True
            )
        )

        self.speed_head.append(
            ResnetCostmapBlock(
                    in_channels=self.channel_sizes[-2],
                    out_channels=self.channel_sizes[-2],
                    activation=hidden_activation,
                    dropout=dropout,
                    kernel_size=1,
                )
            )

        self.speed_head.append(
            nn.Conv2d(
                in_channels=self.channel_sizes[-2],
                out_channels=self.speed_nbins,
                kernel_size=1,
                bias=True
            )
        )

        # # last conv to avoid activation (for cost head)
        # self.cost_head = nn.Conv2d(
        #     in_channels=self.channel_sizes[-2],
        #     out_channels=self.cost_nbins,
        #     kernel_size=1,
        #     bias=True
        # )
        # self.speed_head = nn.Conv2d(
        #     in_channels=self.channel_sizes[-2],
        #     out_channels=self.speed_nbins,
        #     kernel_size=1,
        #     bias=True,
        # )

        self.cnn = torch.nn.Sequential(*self.cnn)
        self.cost_head = torch.nn.Sequential(*self.cost_head)
        self.speed_head = torch.nn.Sequential(*self.speed_head)

    def forward(self, x, return_mean_entropy=False):
        features = self.cnn.forward(x)
        cost_logits = self.cost_head(features)
        speed_logits = self.speed_head(features)

        # exponentiate the mean value too, as speeds are always positive
        res = {"cost_logits": cost_logits, "speed_logits": speed_logits}

        if return_mean_entropy:
            res["costmap"], res["costmap_entropy"] = compute_map_mean_entropy(cost_logits, self.cost_bins)
            res["speedmap"], res["speedmap_entropy"] = compute_map_mean_entropy(speed_logits, self.speed_bins)

            res["costmap"] = res["costmap"].exp()

        return res

    def to(self, device):
        super().to(device)
        self.cost_bins = self.cost_bins.to(device)
        self.speed_bins = self.speed_bins.to(device)
        return self

class LinearCostmapSpeedmapEnsemble2(nn.Module):
    """
    Handle the linear case separately
    """

    def __init__(
        self,
        in_channels,
        ensemble_dim=100,
        dropout=0.0,
        activation_type="sigmoid",
        activation_scale=1.0,
        device="cpu",
    ):
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

        # last conv to avoid activation (for cost head)
        self.cost_heads = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.channel_sizes[-2],
                    out_channels=1,
                    kernel_size=1,
                    bias=True,
                )
                for _ in range(self.ensemble_dim)
            ]
        )
        self.speed_heads = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.channel_sizes[-2],
                    out_channels=2,
                    kernel_size=1,
                    bias=True,
                )
                for _ in range(self.ensemble_dim)
            ]
        )

        if activation_type == "sigmoid":
            self.activation = ScaledSigmoid(scale=activation_scale)
        elif activation_type == "exponential":
            self.activation = Exponential(scale=activation_scale)
        elif activation_type == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_type == "addmin":
            self.activation = AddMin()
        elif activation_type == "none":
            self.activation = nn.Identity()

    def forward(self, x, return_features=True):
        features = x
        idx = torch.randint(self.ensemble_dim, size=(1,))
        cost_head = self.cost_heads[idx]
        speed_head = self.speed_heads[idx]

        costmap = self.activation(cost_head(features))
        speed_logits = speed_head(features)

        # exponentiate the mean value too, as speeds are always positive
        speed_dist = torch.distributions.Normal(
            loc=speed_logits[..., 0, :, :].exp(),
            scale=(speed_logits[..., 1, :, :].exp() + 1e-6),
        )

        return {"costmap": costmap, "speedmap": speed_dist, "features": features}

    def ensemble_forward(self, x, return_features=True):
        features = x
        costmap = torch.stack(
            [self.activation(f.forward(features)) for f in self.cost_heads], dim=-4
        )
        speed_logits = torch.stack(
            [f.forward(features) for f in self.speed_heads], dim=-3
        )

        # exponentiate the mean value too, as speeds are always positive
        speed_dist = torch.distributions.Normal(
            loc=speed_logits[..., 0, :, :].exp(),
            scale=(speed_logits[..., 1, :, :].exp() + 1e-6),
        )

        return {"costmap": costmap, "speedmap": speed_dist, "features": features}


class ResnetCostmapBlock(nn.Module):
    """
    A ResNet-style block that does VGG + residual. Like the VGG-style block, output size is half of input size.
    In contrast to the original resnet block, don't use pooling or batch norm.
    """

    def __init__(self, in_channels, out_channels, activation, kernel_size=3, dropout=0.0):
        assert kernel_size %2 == 1, "need odd kernel size"
        pad = kernel_size//2
        super(ResnetCostmapBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=pad
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=pad
        )
        self.activation = activation()
        self.projection = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
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
