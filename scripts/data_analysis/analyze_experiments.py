"""
Script to analyze the results of experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def walk_experiments(fp):
    res = []
    for root, dirs, files in os.walk(fp):
        for f in files:
            if f == 'metrics.pt':
                res.append(os.path.join(root, f))
    return res

def make_rolling_metrics(metrics, rolling=30):
    rolling_metrics = {k:{kk:[] for kk in metrics[k].keys()} for k in metrics.keys()}
    for ek in rolling_metrics.keys():
        for mk in rolling_metrics[ek].keys():
            rolling_metrics[ek][mk] = torch.stack([metrics[ek][mk][i:-(rolling-i)] for i in range(rolling)], dim=0).mean(dim=0)

    return rolling_metrics

def make_plots(metrics, rolling=30, fig=None, axs=None):
    """
    Make simple datapt-by-datapt plots of each metric. Assume that all metrics are on the same dataset and contain the same keys.
    """
    N = len(list(metrics.values())[0].keys())
    h = 2
    w = int(N/h) + ((N%h)!=0)
    fig, axs = plt.subplots(h, w, figsize=(4 * w, 4 * h))
    axs = axs.flatten()

    #create shorter labels for readability
    short_keys = list(metrics.keys())
    prefix_idx = len(os.path.commonpath(short_keys)) + 1
    suffix_idx = len(os.path.commonpath([x[::-1] for x in short_keys])) + 1
    short_keys = [x[prefix_idx:-suffix_idx] for x in short_keys]

    for ei, (ek, ev) in enumerate(metrics.items()):
        for mi, (mk, mv) in enumerate(ev.items()):
            axs[mi].plot(mv, label=short_keys[ei])
            if ei == len(metrics.keys()) - 1:
                axs[mi].set_title(mk)
                if mi == 0:
                    axs[mi].legend()

    return fig, axs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True, help='dir contaning experiment metrics')
    args = parser.parse_args()

    fps = walk_experiments(args.experiment_dir)

    print('Generating results for:')
    for fp in fps:
        print('\t' + fp)

    metrics = {k:torch.load(k) for k in fps}
    for k in metrics.keys():
        print(k)
        for kk, vv in metrics[k].items():
            print('\t{}:{:.4f}'.format(kk, vv.mean().item()))

    rolling_metrics = make_rolling_metrics(metrics, rolling=50)
    fig, axs = make_plots(rolling_metrics)
    plt.show()
