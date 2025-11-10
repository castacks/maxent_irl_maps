import argparse
import matplotlib.pyplot as plt

from tartandriver_utils.os_utils import load_yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_fp', type=str, required=True)
    args = parser.parse_args()

    res = load_yaml(args.label_fp)

    agg = {k:0 for k in res['classes']}

    for dpt in res['dpts']:
        agg[dpt['class_label']] += 1

    print(agg)