import torch
import matplotlib.pyplot as plt
import argparse

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fp', type=str, required=True, help='model to load')
    parser.add_argument('--test_fp', type=str, required=True, help='path to preproc data')
    parser.add_argument('--n', type=int, required=False, default=10, help='number of viz to run')
    parser.add_argument('--device', type=str, required=False, default='cpu', help='the device to run on')
    args = parser.parse_args()

    res = torch.load(args.model_fp).to(args.device)

    dataset = MaxEntIRLDataset(
        root_fp = args.test_fp,
        feature_keys = res.expert_dataset.feature_keys
    ).to(args.device)
    res.expert_dataset = dataset

    for i in range(args.n):
        res.visualize()
        plt.show()
