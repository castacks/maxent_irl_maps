import os
import torch
import argparse
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.experiment_management.parse_configs import setup_experiment

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

    for i in range(args.n):
        res.visualize()
        plt.show()
