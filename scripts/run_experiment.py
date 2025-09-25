import argparse
import torch

from maxent_irl_maps.experiment_management.parse_configs import setup_experiment

import matplotlib
matplotlib.use("TkAgg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup_fp", type=str, required=True, help="path to the experiment yaml"
    )
    args = parser.parse_args()

    res = setup_experiment(args.setup_fp)

    print("dataset size: {}".format(len(res["dataset"])))

    res["experiment"].run()
