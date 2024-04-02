import os
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='dir of experiments to compare')
    parser.add_argument('--metrics_dir', type=str, required=False, default='metrics', help='name of metrics dir')
    args = parser.parse_args()

    efps = os.listdir(args.root_dir)

    #we can just hack out the timestamp
    efp_names = [x[20:] for x in efps]

    group_keys = set(efp_names)
    res = {k:{} for k in group_keys}

    for efp in efps:
        group = efp[20:]
        metrics = torch.load(os.path.join(args.root_dir, efp, args.metrics_dir, 'metrics.pt'))

        for k,v in metrics.items():
            if k not in res[group].keys():
                res[group][k] = []

            res[group][k].append(v.mean())

    res = {k:{kk:torch.stack(vv) for kk, vv in v.items()} for k,v in res.items()}

    #pprint
    for group, metrics in res.items():
        print(group)
        for mk, mv in metrics.items():
            mu = mv.mean()
            sig = mv.std()
            print('\t{}:{:.4f}+-{:.4f}'.format(mk, mu, sig))

        print('_____')
