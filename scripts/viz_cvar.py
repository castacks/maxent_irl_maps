import os
import yaml
import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_maps.experiment_management.parse_configs import setup_experiment, load_net_for_eval
from maxent_irl_maps.utils import compute_map_cvar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_fp", type=str, required=True, help="model to load")
    parser.add_argument(
        "--test_fp", type=str, required=True, help="path to preproc data"
    )
    parser.add_argument(
        "--n", type=int, required=False, default=10, help="number of viz to run"
    )
    parser.add_argument(
        "--device", type=str, required=False, default="cpu", help="the device to run on"
    )
    args = parser.parse_args()

    res = load_net_for_eval(args.model_fp, device=args.device, skip_mpc=False)

    dataset = MaxEntIRLDataset(
        root_fp=args.test_fp, feature_keys=res.expert_dataset.feature_keys
    ).to(args.device)
    res.expert_dataset = dataset

    cvars = torch.linspace(-0.9, 0.9, 9, device=args.device)

    for i in range(args.n):
        idx = np.random.randint(len(res.expert_dataset))

        with torch.no_grad():
            data = res.expert_dataset[idx]

            # hack back to single dim
            map_features = torch.stack([data["map_features"]] * res.mppi.B, dim=0)
            metadata = data["metadata"]
            xmin = metadata["origin"][0].cpu()
            ymin = metadata["origin"][1].cpu()
            xmax = xmin + metadata["length"][0].cpu()
            ymax = ymin + metadata["length"][1].cpu()
            expert_traj = data["traj"]

            cres = res.network.ensemble_forward(map_features, return_mean_entropy=True)

            costmaps = []
            for cvar in cvars:
                #[B x 1 x W x H]
                cmap = compute_map_cvar(cres["costmap"].swapaxes(0, 1), cvar)
                costmaps.append(cmap)

        #plot
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))

        img = data["image"].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu()

        axs[0, 0].set_title('FPV')
        axs[0, 0].imshow(img)

        cvar_diff = costmaps[-1] - costmaps[0]

        axs[1, 0].set_title("CvaR {:.2f} - CVaR {:.2f}".format(cvars[-1].item(), cvars[0].item()))
        axs[1, 0].imshow(cvar_diff[0, 0].T.cpu(), origin="lower", extent=(xmin, xmax, ymin, ymax))
        axs[1, 0].set_xlabel('X(m)')
        axs[1, 0].set_ylabel('Y(m)')

        for ax, cvar, cmap in zip(axs[:, 1:].flatten(), cvars, costmaps):
            ax.set_title("CVaR {:.2f}".format(cvar.item()))
            ax.imshow(
                cmap[0, 0].T.cpu(),
                origin="lower",
                cmap="jet",
                extent=(xmin, xmax, ymin, ymax),
            )
            ax.set_xlabel('X(m)')
            ax.set_ylabel('Y(m)')

        plt.show()