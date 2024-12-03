import argparse
import torch

from maxent_irl_maps.experiment_management.parse_configs import setup_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup_fp", type=str, required=True, help="path to the experiment yaml"
    )
    args = parser.parse_args()

    res = setup_experiment(args.setup_fp)

    print("dataset size: {}".format(len(res["dataset"])))

    print(
        {
            k: v.shape if isinstance(v, torch.Tensor) else v
            for k, v in res["dataset"][1].items()
        }
    )

    #    for i in range(10):
    #        res['dataset'].visualize()
    #        plt.show()

    res["experiment"].run()
