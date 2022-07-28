import torch

class LethalHeightCostmap:
    def __init__(self, diff_idx=7, cost=1e8):
        self.diff_idx = diff_idx
        self.cost = 1e8

    def forward(self, map_features):
        import pdb;pdb.set_trace()
