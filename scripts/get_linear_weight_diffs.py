"""
Compute change in linear weights over time
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt

from maxent_irl_costmaps.os_utils import walk_bags

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True, help="dir of linear weights"
    )
    args = parser.parse_args()

    fps = walk_bags(args.model_dir, extension=".pt")

    weights = []

    for fp in fps:
        # linear
        #        ens_w = [x.weight.squeeze().detach().cpu() for x in torch.load(fp).network.cost_heads]

        # nn
        ens_w = [torch.load(fp).network.cost_head.weight.squeeze().detach()] * 1

        ens_w = torch.stack(ens_w, dim=0)
        weights.append(ens_w)

    weights = torch.stack(weights, dim=0)  # [T x E x W]

    weight_diffs = torch.linalg.norm(
        weights.mean(dim=1)[1:] - weights.mean(dim=1)[:-1], dim=-1
    )

    plt.plot(weight_diffs, marker=".")
    plt.xlabel("epoch")
    plt.ylabel("weight norm")
    plt.show()
