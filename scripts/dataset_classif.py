import os
import argparse

import matplotlib.pyplot as plt

from tartandriver_utils.os_utils import load_yaml, save_yaml

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset

#NULL is here for weird dpts/to make real classes start at 1
dpt_classes = ['none', 'trail_dense', 'trail_open', 'no_trail_dense', 'no_trail_open']

def viz_dpt(dpt, fig, axs):
    img = dpt['image']['data'][[2,1,0]].permute(1,2,0).cpu().numpy()
    traj = dpt['odometry']['data'][:, :2].cpu().numpy()
    bev_data = dpt['bev_data']['data'][0].cpu().numpy()
    bev_extent = dpt['bev_data']['metadata'].extent()

    for ax in axs:
        ax.cla()

    axs[0].imshow(img)

    axs[1].imshow(bev_data.T, extent=bev_extent, origin='lower', cmap='gray')
    axs[1].plot(traj[:, 0], traj[:, 1], c='y')
    axs[1].set_xlim(bev_extent[0], bev_extent[1])
    axs[1].set_ylim(bev_extent[2], bev_extent[3])

    return fig, axs

def get_input(dpt_classes):
    for i, cls in enumerate(dpt_classes):
        print(f"{i}: {cls}")
    
    x = "aaa"

    while True:
        x = input(f"Input dpt class (-1 to undo): ")
        if check_input(x, dpt_classes):
            break
        else:
            print('bad input')

    return int(x)

def check_input(x, dpt_classes):
    try:
        _x = int(x)
    except:
        return False
    
    if _x < -1 or _x >= len(dpt_classes):
        return False
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, required=True, help='path to dataset')
    parser.add_argument('--dataset_fp', type=str, required=False, default=None, help='set this arg to point to another dataset using the same config (useful for labeling test sets)')
    args = parser.parse_args()

    config = load_yaml(args.dataset_config)

    if args.dataset_fp is not None:
        config['common']['root_dir'] = args.dataset_fp
    
    dataset = MaxEntIRLDataset(config)

    save_fp = os.path.join(dataset.root_dir, 'irl_dataset_classif.yaml')
    if os.path.exists(save_fp):
        x = input(f"{save_fp} exists. Overwrite? [Y/n]")
        if x == 'n':
            exit(0)

    ## main loop ##
    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
    plt.show(block=False)

    res = {
        'classes': dpt_classes,
        'dpts': [None] * len(dataset)
    }

    curr_idx = 0

    while curr_idx < len(dataset):
        print(f"dpt {curr_idx+1}/{len(dataset)}")

        dpt = dataset[curr_idx]

        viz_dpt(dpt, fig, axs)
        plt.pause(1e-2)
        inp = get_input(dpt_classes)

        if inp > 0:
            dpt_dict = {
                'idx': curr_idx,
                'rdir': f"{dpt['rdir']}",
                'subidx': f"{dpt['subidx']:08d}",
                'class_id': inp,
                'class_label': dpt_classes[inp]
            }

            res['dpts'][curr_idx] = dpt_dict
            save_yaml(res, save_fp)

            curr_idx += 1

        else:
            curr_idx = max(0, curr_idx-1)

    print('done!')