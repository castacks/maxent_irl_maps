#miscellaneous network utils

import torch

from torch import nn

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
