import os
import yaml
import torch
import argparse
import matplotlib.pyplot as plt

import matplotlib; matplotlib.use("TkAgg")

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_maps.experiment_management.parse_configs import setup_experiment, load_net_for_eval

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

    model_base_dir = os.path.split(args.model_fp)[0]
    config_fp = os.path.join(model_base_dir, '_params.yaml')
    config = yaml.safe_load(open(config_fp, 'r'))

    config['dataset']['common']['root_dir'] = args.test_fp

    res = setup_experiment(config, skip_norms=True)["algo"].to(args.device)
    res.network.load_state_dict(torch.load(args.model_fp, weights_only=True, map_location=args.device))
    res.network.eval()

    idxs = torch.randperm(len(res.expert_dataset))[:args.n]

    for idx in idxs:
        fig, axs = res.visualize(idx=idx)

        ## show feat image
        from physics_atv_visual_mapping.utils import normalize_dino

        axs[-1].cla()
        img = res.expert_dataset[idx]["feature_image"]
        img = img['data'].unsqueeze(0)
        with torch.no_grad():
            feat_img = res.network.voxel_recolor.feat_net(img)[0].permute(1,2,0)

        U,S,V = torch.pca_lowrank(feat_img.flatten(end_dim=-2))

        feat_img_pca = feat_img @ V

        axs[-1].imshow(normalize_dino(feat_img_pca).cpu().numpy())

        plt.show()
