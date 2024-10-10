# miscellaneous network utils

import torch

from torch import nn


class AddMin(nn.Module):
    """
    Feels bad but add min to make nonnegative outputs
    """

    def __init__(self, ndims=3):
        super(AddMin, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        assert len(x.shape) >= 3, "AddMin only works w/ 3+ dims"
        leading_dims = x.shape[:-2]
        vmin = x.view(*leading_dims, -1).min(dim=-1)[0].view(*leading_dims, 1, 1)

        return x - vmin


class Exponential(nn.Module):
    """
    Try exponentiating
    """

    def __init__(self, scale=1.0):
        super(Exponential, self).__init__()
        self.scale = scale

    def forward(self, x):
        return (self.scale * x).exp()


class ScaledSigmoid(nn.Module):
    """
    Try adding a scaling factor to the sigmois activation to make it less sharp
    """

    def __init__(self, scale=1.0):
        super(ScaledSigmoid, self).__init__()
        self.scale = scale

    def forward(self, x):
        return (self.scale * x).sigmoid()
