import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.experiment_management.parse_configs import setup_experiment

def compute_speedmap_quantile(speedmap_cdf, speed_bins, q):
    """
    Given a speedmap (parameterized as cdf + bins) and a quantile,
        compute the speed corresponding to that quantile

    Args:
        speedmap_cdf: BxWxH Tensor of speedmap cdf
        speed_bins: B+1 Tensor of bin edges
        q: float containing the quantile 
    """
    B = len(speed_bins)
    #need to stack zeros to front
    _cdf = torch.cat([
        torch.zeros_like(speedmap_cdf[[0]]),
        speedmap_cdf
    ], dim=0)

    _cdf = _cdf.view(B, -1)

    _cdiffs = q - _cdf
    _mask = _cdiffs <= 0

    _qidx = (_cdiffs + 1e10*_mask).argmin(dim=0)

    _cdf_low = _cdf.T[torch.arange(len(_qidx)), _qidx]
    _cdf_high = _cdf.T[torch.arange(len(_qidx)), _qidx+1]

    _k = (q - _cdf_low) / (_cdf_high - _cdf_low)

    _speedmap_low = speed_bins[_qidx]
    _speedmap_high = speed_bins[_qidx+1]

    _speedmap = (1-_k) * _speedmap_low + _k * _speedmap_high

    return _speedmap.reshape(*speedmap_cdf.shape[1:])

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

        with torch.no_grad():
            speedmap_probs = res.network.forward(dpt['map_features'].unsqueeze(0))['speedmap'][0].softmax(dim=0)
            speedmap_cdf = torch.cumsum(speedmap_probs, dim=0)

        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        axs = axs.flatten()
        axs[0].imshow(dpt['image'][[2,1,0]].permute(1,2,0).cpu())

        quantiles = torch.linspace(0.1, 0.9, 9)
        for q, ax in zip(quantiles, axs[1:]):
            speedmap = compute_speedmap_quantile(speedmap_cdf, res.network.speed_bins.to(args.device), q)

            ax.imshow(speedmap.cpu().T, origin='lower', vmin=res.network.speed_bins[0], vmax=res.network.speed_bins[-1], cmap='jet')
            ax.set_title('q = {:.2f}'.format(q.item()))

        plt.show()
