import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tartandriver_utils.os_utils import load_yaml, save_yaml, is_kitti_dir, kitti_n_frames

from ros_torch_converter.datatypes.image import ImageTorch
from ros_torch_converter.datatypes.bev_grid import BEVGridTorch
from ros_torch_converter.datatypes.rb_state import OdomRBStateTorch

#NULL is here for weird dpts/to make real classes start at 1
dpt_classes = [
    'none',
    'trail_open',       #Discernable trail and clear for >3 ATV widths
    'trail_sparse',     #Discernable trail and 0-2 obstacles/hittable objects within 3 ATV widths
    'trail_dense',      #Discernable trail and 3+ obstacles/things within 3 atv widths
    'no_trail_open',    # same as above, but no discernable trail
    'no_trail_sparse',
    'no_trail_dense',
]

def viz_dpt(rdir, subidx, fig, axs):
    N = kitti_n_frames(rdir)

    it = ImageTorch.from_kitti(os.path.join(rdir, 'image'), subidx)
    bgt = BEVGridTorch.from_kitti(os.path.join(rdir, 'bev_map_reduce'), subidx)
    ot = OdomRBStateTorch.from_kitti_multi(
        os.path.join(rdir, 'odometry'),
        range(subidx, min(subidx+75, N))
    )
    traj = np.stack([x.state[:2] for x in ot])

    bev_data = bgt.bev_grid.data[..., bgt.bev_grid.feature_keys.index('num_voxels')]
    bev_extent = bgt.bev_grid.metadata.extent()

    for ax in axs:
        ax.cla()

    axs[0].imshow(it.image[..., [2,1,0]])

    axs[1].imshow(bev_data.T, extent=bev_extent, origin='lower', cmap='gray')
    axs[1].plot(traj[:, 0], traj[:, 1], c='y')
    axs[1].set_xlim(bev_extent[0], bev_extent[1])
    axs[1].set_ylim(bev_extent[2], bev_extent[3])

    fig.suptitle(f'rdir = {rdir}, frame {subidx+1}/{N}')

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

def update_yaml(dpt_dict):
    yaml_fp = os.path.join(dpt_dict['rdir'], 'irl_classif.yaml')
    if not os.path.exists(yaml_fp):
        yaml_data = {
            'classes': dpt_classes,
            'dpts': {}
        }
        save_yaml(yaml_data, yaml_fp)

    yaml_data = load_yaml(yaml_fp)

    yaml_data['dpts'][dpt_dict['subidx']] = dpt_dict
    save_yaml(yaml_data, yaml_fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='path to dataset root')
    parser.add_argument('--label_every', type=int, required=False, default=25)
    args = parser.parse_args()

    run_dirs = [os.path.join(args.root_dir, x) for x in os.listdir(args.root_dir)] + [args.root_dir]
    run_dirs = [x for x in run_dirs if is_kitti_dir(x)]

    frame_list = []
    print('Running labeling script for the following dirs:')
    for x in run_dirs:
        print(f'\t{x}')
        N = kitti_n_frames(x)
        idxs = range(N)[::args.label_every]
        frame_list.extend([(x, i) for i in idxs])

    print(f'{len(frame_list)} total frames to label')
    
    ## main loop ##
    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
    plt.show(block=False)

    curr_idx = 0

    while curr_idx < len(frame_list):
        print(f"dpt {curr_idx+1}/{len(frame_list)}")

        rdir, subidx = frame_list[curr_idx]

        viz_dpt(rdir, subidx, fig, axs)

        plt.pause(1e-2)
        inp = get_input(dpt_classes)

        if inp > 0:
            dpt_dict = {
                'rdir': rdir,
                'subidx': subidx,
                'class_id': inp,
                'class_label': dpt_classes[inp]
            }

            update_yaml(dpt_dict)

            curr_idx += 1

        else:
            curr_idx = max(0, curr_idx-1)

    print('done!')