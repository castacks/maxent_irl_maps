import os
import tqdm
import torch
import argparse

from sklearn.linear_model import QuantileRegressor

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.experiment_management.parse_configs import setup_experiment
from maxent_irl_costmaps.utils import get_state_visitations, get_speedmap

"""
Compute a simple speed baseline
   for alter:
    fit a linear fn bet. speed and SVD2
   for sematnics:
    compute the average speed per class
"""

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_fp", type=str, required=True, help="path to save figs to"
    )
    parser.add_argument(
        "--model_fp", type=str, required=True, help="Costmap weights file"
    )
    parser.add_argument(
        "--test_fp", type=str, required=True, help="dir to save preprocessed data to"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="device to run script on",
    )
    args = parser.parse_args()

    param_fp = os.path.join(os.path.split(args.model_fp)[0], "_params.yaml")
    model = setup_experiment(param_fp)["algo"].to(args.device)

    model.network.load_state_dict(torch.load(args.model_fp))
    model.network.eval()

    dataset = MaxEntIRLDataset(
        root_fp=args.test_fp, feature_keys=model.expert_dataset.feature_keys
    ).to(args.device)
    model.expert_dataset = dataset

    res = {"speed": [], "svd2": [], "semantic_class": []}

    svd2_idx = dataset.feature_keys.index("SVD2")
    svd2_mean = dataset.feature_mean[svd2_idx]
    svd2_std = dataset.feature_std[svd2_idx]

    ganav_idxs = [dataset.feature_keys.index("ganav_{}".format(i)) for i in range(12)]

    for i in tqdm.tqdm(range(len(dataset))):
        dpt = dataset[i]

        map_features = dpt["map_features"].permute(1, 2, 0)
        map_metadata = dpt["metadata"]
        expert_traj = dpt["traj"]
        expert_speeds = torch.linalg.norm(expert_traj[:, 7:10], dim=-1)

        speedmap = get_speedmap(
            expert_traj.unsqueeze(0), expert_speeds.unsqueeze(0), map_metadata
        )

        mask = speedmap > 0.1

        speeds = speedmap[mask]

        feats = map_features[mask]
        svd2s = (feats[:, svd2_idx] * svd2_std) + svd2_mean
        sclasses = feats[:, ganav_idxs].argmax(dim=-1)

        res["speed"].append(speeds)
        res["svd2"].append(svd2s)
        res["semantic_class"].append(sclasses)

    res = {k: torch.cat(v) for k, v in res.items()}

    # alter
    # (simple linreg in closed form)
    #    qreg = QuantileRegressor(quantile=0.95, alpha=0, solver="highs")
    qreg = QuantileRegressor(quantile=0.5, alpha=0, solver="highs")
    qreg.fit(res["svd2"].unsqueeze(-1).cpu().numpy(), res["speed"].cpu().numpy())

    print("ALTER")
    print(
        "speed = {:.2f} + {:.2f}*svd2".format(qreg.intercept_.item(), qreg.coef_.item())
    )

    # semantics
    semantic_speeds = []
    for i in range(12):
        class_speeds = res["speed"][res["semantic_class"] == i]
        #        semantic_speeds.append(torch.quantile(class_speeds, 0.95))
        semantic_speeds.append(torch.quantile(class_speeds, 0.5))

    print("SEMANTICS")
    for i in range(12):
        print("class {}:{:.2f}".format(i, semantic_speeds[i]))
