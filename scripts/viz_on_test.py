import torch
import matplotlib.pyplot as plt
import argparse

from maxent_irl_costmaps.dataset.preprocess_pointpillars_dataset import PreprocessPointpillarsDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fp', type=str, required=True, help='model to load')
    parser.add_argument('--preprocess_fp', type=str, required=True, help='path to preproc data')
    parser.add_argument('--n', type=int, required=False, default=10, help='number of viz to run')
    parser.add_argument('--device', type=str, required=False, default='cpu', help='the device to run on')
    args = parser.parse_args()

    res = torch.load(args.model_fp).to(args.device)
    dataset = PreprocessPointpillarsDataset(preprocess_fp=args.preprocess_fp, gridmap_type=res.expert_dataset.gridmap_type, feature_mean=res.expert_dataset.feature_mean, feature_std=res.expert_dataset.feature_std).to(args.device)
    res.expert_dataset = dataset

    for i in range(args.n):
        res.visualize()
        plt.show()
