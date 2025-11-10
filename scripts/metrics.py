import os
import yaml
import argparse
import numpy as np

from tabulate import tabulate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_config', type=str, required=True, help='path to metrics config')
    parser.add_argument('--latex', action='store_true', help='set this flag for latex output')
    args = parser.parse_args()

    metrics_config = yaml.safe_load(open(args.metrics_config, 'r'))

    metrics_keys = [
        'mhd',
        'expert_log_goal',
        'expert_speed_prob',
        'expert_speed_emd2'
    ]

    res = [['Experiment'] + metrics_keys]

    for label, mdir in metrics_config.items():
        metrics = np.load(os.path.join(mdir, 'metrics.npz'))

        res.append([label] + [f"{metrics[k].mean():.04f}+-{metrics[k].std():.04f}" for k in metrics_keys])

    if args.latex:
        pass
    else:
        tabdata = res
        print(tabulate(tabdata))