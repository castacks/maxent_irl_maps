import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.experiment_management.parse_configs import setup_experiment
from maxent_irl_costmaps.os_utils import maybe_mkdir


def compute_cvar(costmaps, q):
    costmap_q = torch.quantile(costmaps, q, dim=0)
    mask = costmaps >= costmap_q.unsqueeze(0)
    costmap_cvar = (costmaps * mask).sum(dim=0) / mask.sum(dim=0)
    return costmap_cvar


def visualize_ensemble(model, idx):
    """
    Look at network feature maps to see what the activations are doing
    """
    with torch.no_grad():
        if idx == -1:
            idx = np.random.randint(len(model.expert_dataset))

        data = model.expert_dataset[idx]

        map_features = data["map_features"]
        metadata = data["metadata"]
        xmin = metadata["origin"][0].cpu().item()
        ymin = metadata["origin"][1].cpu().item()
        xmax = xmin + metadata["width"].cpu().item()
        ymax = ymin + metadata["height"].cpu().item()
        expert_traj = data["traj"]

        # compute costmap
        # resnet cnn
        mosaic = """
        ABCDE
        FGHIJ
        KLMNO
        """

        fig = plt.figure(tight_layout=True, figsize=(16, 12))
        ax_dict = fig.subplot_mosaic(mosaic)
        all_axs = list(ax_dict.values())
        axs1 = all_axs[:5]
        axs2 = all_axs[5:]

        # plot image, mean costmap, std costmap, and a few samples
        t1 = time.time()
        res = model.network.ensemble_forward(
            map_features.view(1, *map_features.shape), ne=-1
        )
        costmaps = res["costmap"][0, :, 0]
        t2 = time.time()

        costmap_mean = costmaps.mean(dim=0)
        costmap_std = costmaps.std(dim=0)

        idx = model.expert_dataset.feature_keys.index("diff")
        axs1[0].imshow(data["image"].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
        axs1[1].imshow(
            map_features[idx].cpu(),
            origin="lower",
            cmap="gray",
            extent=(xmin, xmax, ymin, ymax),
        )
        r2 = axs1[2].imshow(
            costmap_mean.cpu(),
            origin="lower",
            cmap="plasma",
            extent=(xmin, xmax, ymin, ymax),
        )
        r3 = axs1[3].imshow(
            costmap_std.cpu(),
            origin="lower",
            cmap="plasma",
            extent=(xmin, xmax, ymin, ymax),
        )

        fig.suptitle("Inference = {:.4f}s".format(t2 - t1))
        axs1[0].set_title("FPV")
        axs1[1].set_title("Diff")
        axs1[2].set_title("Costmap Mean")
        axs1[3].set_title("Costmap std")

        vmin = torch.quantile(costmaps, 0.1)
        vmax = torch.quantile(costmaps, 0.9)

        for i, q in enumerate(torch.linspace(0.05, 0.95, len(axs2))):
            cm = compute_cvar(costmaps, q.to(costmaps.device))
            r = axs2[i].imshow(
                cm.cpu(),
                origin="lower",
                cmap="plasma",
                extent=(xmin, xmax, ymin, ymax),
                vmin=vmin,
                vmax=vmax,
            )
            axs2[i].set_title("CVaR_{:.2f}".format(q.item()))
            axs2[i].get_xaxis().set_visible(False)
            axs2[i].get_yaxis().set_visible(False)


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

    param_fp = os.path.join(os.path.split(args.model_fp)[0], "_params.yaml")
    res = setup_experiment(param_fp)["algo"].to(args.device)

    res.network.load_state_dict(torch.load(args.model_fp))
    res.network.eval()

    dataset = MaxEntIRLDataset(
        root_fp=args.test_fp, feature_keys=res.expert_dataset.feature_keys
    ).to(args.device)
    res.expert_dataset = dataset

    for i in range(len(dataset)):
        visualize_ensemble(res, idx=-1)
        plt.show()
