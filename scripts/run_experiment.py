import argparse
import torch

from maxent_irl_maps.experiment_management.parse_configs import setup_experiment
from maxent_irl_maps.experiment_management.setup_torch_coordinator_irl import setup_torch_coordinator_irl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup_fp", type=str, required=True, help="path to the experiment yaml"
    )
    parser.add_argument('--torch_coordinator', action='store_true', help='set this flag if using a torch coordinator config')
    args = parser.parse_args()

    if args.torch_coordinator:
        res = setup_torch_coordinator_irl(args.setup_fp)

        print("dataset size: {}".format(len(res["dataset"])))

        res["experiment"].run()
    else:
        res = setup_experiment(args.setup_fp)

        print("dataset size: {}".format(len(res["dataset"])))

        print(
            {
                k: v.shape if isinstance(v, torch.Tensor) else v
                for k, v in res["dataset"][1].items()
            }
        )

        res["experiment"].run()
