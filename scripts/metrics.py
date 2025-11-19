import os
import yaml
import argparse
import numpy as np

from tabulate import tabulate

from tartandriver_utils.os_utils import load_yaml

def get_classif_idxs(metrics, classif_data):
    """
    Args:
        metrics: metrics dir to get idxs for. Should contain rdir, subidx keys
        classif_data: classif yaml that contains class and rdir, subidx
    """
    classif = {k:np.array(list(v.keys())) for k,v in classif_data['dpts'].items()}
    res = []
    for rdir, subidx in zip(metrics['rdir'], metrics['subidx']):
        _rdir = os.path.basename(rdir)
        assert _rdir in classif.keys(), f"classif missing run {rdir}!"
        target_subidxs = classif[_rdir]

        mindist = np.abs(target_subidxs - subidx).min()
        minidx = np.abs(target_subidxs - subidx).argmin()
        minidx = target_subidxs[minidx]

        assert mindist < 25, "got mindist to subidx {subidx} > 25. Something probably wrong!"

        class_id = classif_data['dpts'][_rdir][minidx]['class_id']
        res.append(class_id)

    return np.array(res)

def get_classif_data(dataset_dir):
    run_dirs = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
    run_dirs = [x for x in run_dirs if os.path.exists(os.path.join(x, 'irl_classif.yaml'))]

    res = {
        'classes': None,
        'dpts': {}
    }

    for rdir in run_dirs:
        classif = load_yaml(os.path.join(rdir, 'irl_classif.yaml'))
        if res['classes'] is None:
            res['classes'] = classif['classes']

        assert res['classes'] == classif['classes'], f"expected classes {res['classes']}, got {classif['classes']}"

        res['dpts'][os.path.basename(rdir)] = classif['dpts']

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_config', type=str, required=True, help='path to metrics config')
    parser.add_argument('--latex', action='store_true', help='set this flag for latex output')
    args = parser.parse_args()

    metrics_config = load_yaml(args.metrics_config)

    for exp_key, exp_conf in metrics_config.items():
        metrics_keys = [
            'mhd',
            'expert_log_goal',
            'expert_speed_prob',
            'expert_speed_emd2'
        ]

        classif_data = get_classif_data(exp_conf['classif'])

        res = []

        for label, mdir in exp_conf['metrics'].items():
            metrics = np.load(os.path.join(mdir, 'metrics.npz'))
            classif_idxs = get_classif_idxs(metrics, classif_data)

            for class_id in range(1, len(classif_data['classes'])):
                class_label = classif_data['classes'][class_id]

                mask = (classif_idxs == class_id)
                row = [label, f"{class_label} ({mask.sum()} dpts)"]
                for mk in metrics_keys:
                    mdata = metrics[mk][mask]
                    if mask.sum() > 1:
                        row.append(f"{mdata.mean():.04f}+-{mdata.std():.04f}")
                    elif mask.sum() == 1:
                        row.append(f"{mdata.mean():.04f}+-N/A")
                    else:
                        row.append("N/A")

                res.append(row)

            ## also add total
            row = [label, f" total ({len(mask)} dpts)"]
            for mk in metrics_keys:
                mdata = metrics[mk]
                if mask.sum() > 1:
                    row.append(f"{mdata.mean():.04f}+-{mdata.std():.04f}")
                elif mask.sum() == 1:
                    row.append(f"{mdata.mean():.04f}+-N/A")
                else:
                    row.append("N/A")

            res.append(row)

        ## sort by terrain type instead of experiment
        res = sorted(res, key=lambda x: x[1] + x[0])

        res.insert(0, ['Experiment'] + ['Terrain Class'] + metrics_keys)

        ##add line breaks lol
        prev = None
        line_breaks = []
        for i, row in enumerate(res):
            if prev != row[1]:
                line_breaks.append(i)

            prev = row[1]

        for i in reversed(line_breaks):
            res.insert(i, [' '] * len(res[i]))

        if args.latex:
            pass
        else:
            tabdata = res
            print(f"{exp_key}")
            print(tabulate(tabdata))
            print('\n\n')