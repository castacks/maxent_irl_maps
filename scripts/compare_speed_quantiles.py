import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.experiment_management.parse_configs import setup_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fp', type=str, required=True, help='model to load')
    parser.add_argument('--test_fp', type=str, required=True, help='path to preproc data')
    parser.add_argument('--n', type=int, required=False, default=10, help='number of viz to run')
    parser.add_argument('--device', type=str, required=False, default='cpu', help='the device to run on')
    args = parser.parse_args()

    param_fp = os.path.join(os.path.split(args.model_fp)[0], '_params.yaml')
    res = setup_experiment(param_fp)['algo'].to(args.device)

    res.network.load_state_dict(torch.load(args.model_fp))
    res.network.eval()

    dataset = MaxEntIRLDataset(
        root_fp = args.test_fp,
        feature_keys = res.expert_dataset.feature_keys
    ).to(args.device)
    res.expert_dataset = dataset

    for i in range(args.n):
        idx = np.random.randint(len(dataset))
        dpt = dataset[idx]

        speedmap_probs = res.network.forward(dpt['map_features'].unsqueeze(0))['speedmap'][0].softmax(dim=0)
        speedmap_cdf = torch.cumsum(speedmap_probs, dim=0)

        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        axs = axs.flatten()
        axs[0].imshow(dpt['image'][[2,1,0]].permute(1,2,0).cpu())

        quantiles = torch.linspace(0.1, 0.9, 9)
        for q, ax in zip(quantiles, axs[1:]):
            qdiff = q - speedmap_cdf
            qdiff[qdiff < 0] = 1e10
            qidx = qdiff.argmin(dim=0)

            speedmap_low = res.network.speed_bins[qidx]
            speedmap_high = res.network.speed_bins[qidx+1]

            ax.imshow(speedmap_high.cpu().T, origin='lower', vmin=res.network.speed_bins[0], vmax=res.network.speed_bins[-1], cmap='jet')
            ax.set_title('q = {:.2f}'.format(q.item()))

        plt.show()
