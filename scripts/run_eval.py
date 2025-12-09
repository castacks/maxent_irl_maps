import os
import yaml
import tqdm
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_maps.experiment_management.parse_configs import setup_experiment, load_net_for_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_fp", type=str, required=True, help="model to load")
    parser.add_argument(
        "--test_fp", type=str, required=False, default=None, help="path to preproc data (leave blank to just invert the gps filter)"
    )
    parser.add_argument('--label', type=str, required=False, default='')
    parser.add_argument('--randomize', action='store_true', help='set this flag to gen plots in random order')
    parser.add_argument(
        "--device", type=str, required=False, default="cpu", help="the device to run on"
    )
    args = parser.parse_args()

    ## hack setup ##
    model_base_dir, model_name = os.path.split(args.model_fp)
    save_dir = os.path.join(model_base_dir, 'metrics'+args.label)
    os.makedirs(os.path.join(save_dir, 'viz'), exist_ok=True)

    res = torch.load(args.model_fp, weights_only=False).to(args.device)
    res.network.eval()

    ## setup dataset ##
    if args.test_fp is not None:
        #setup dataset as normal if theres a path
        dconf = res.dataset.config
        del dconf['common']['gps_filter']
        del dconf['datatypes']['gps_odometry']
        dconf['common']['root_dir'] = args.test_fp
        res.dataset = MaxEntIRLDataset(dconf).to(args.device)
    else:
        # invert the gps mask otherwise
        dconf = res.dataset.config
        dconf['common']['gps_filter']['invert'] = not dconf['common']['gps_filter']['invert']
        res.dataset = MaxEntIRLDataset(dconf).to(res.device)

    print(f'evaluating on dataset {res.dataset.root_dir}...')

    ## real setup ##
    # model_base_dir, model_name = os.path.split(args.model_fp)
    # config_fp = os.path.join(model_base_dir, '_params.yaml')
    # config = yaml.safe_load(open(config_fp, 'r'))

    # config['dataset']['common']['root_dir'] = args.test_fp

    # save_dir = os.path.join(model_base_dir, 'metrics', model_name.strip('.pt') + args.label)
    # os.makedirs(os.path.join(save_dir, 'viz'), exist_ok=True)

    # res = setup_experiment(config, skip_norms=True)["algo"].to(args.device)
    # res.network.load_state_dict(torch.load(args.model_fp, weights_only=True, map_location=args.device))
    # res.network.eval()

    N = len(res.dataset)

    idxs = torch.randperm(N) if args.randomize else torch.arange(N)

    metrics_all = None

    for idx in tqdm.tqdm(idxs):
        results = res.visualize(idx=idx)
        fig, axs = results['viz']
        metrics = {k:np.array(v).reshape(1) for k,v in results['metrics'].items()}

        if metrics_all is None:
            metrics_all = metrics
        else:
            metrics_all = {k:np.concatenate([metrics_all[k], metrics[k]]) for k in metrics_all.keys()}

        np.savez(os.path.join(save_dir, 'metrics.npz'), **metrics_all)
        plt.savefig(os.path.join(save_dir, 'viz', f'{idx:08d}.png'))
        plt.close()

        if False:
            ## show feat image
            from physics_atv_visual_mapping.utils import normalize_dino

            axs[-1].cla()
            img = res.dataset[idx]["feature_image"]
            img = img['data'].unsqueeze(0)
            with torch.no_grad():
                feat_img = res.network.voxel_recolor.feat_net(img)[0].permute(1,2,0)

            U,S,V = torch.pca_lowrank(feat_img.flatten(end_dim=-2))

            feat_img_pca = feat_img @ V

            axs[-1].imshow(normalize_dino(feat_img_pca).cpu().numpy())

        if False:
            ## LSS viz code (prob shouldnt be here lol)
            from physics_atv_visual_mapping.utils import normalize_dino

            with torch.no_grad():
                dpt = res.dataset.getitem_batch([idx])
                #unsqueeze for now for single-cam (TODO need to use a camlist arg)
                images = dpt["feature_image"]["data"].unsqueeze(1)

                #assume all intrinsics the same for now
                intrinsics = dpt["feature_image_intrinsics"]["data"][0]
                pose_H = dpt["tf_odom_to_cam"]["data"]
                bev_metadata = dpt["bev_data"]["metadata"]
                max_depth = torch.linalg.norm(dpt["bev_data"]["metadata"].length[0]) / 2.

                #TODO unhack the unsqueeze when we switch to camlist
                cam_pts = res.network.sample_camera_frustums(
                    pose=pose_H,
                    image=images,
                    intrinsics=intrinsics,
                    n_bins=res.network.lss.n_depth_bins,
                    max_depth=max_depth
                ).unsqueeze(1)

                _metadata = torch.stack([
                    bev_metadata.origin,
                    bev_metadata.length,
                    bev_metadata.resolution
                ], dim=1)

                lss_viz_data = res.network.lss.viz_forward(images, cam_pts, _metadata)

                ## pca the feat img for viz ##
                feat_img = lss_viz_data['feat_img']
                U, S, V = torch.pca_lowrank(feat_img.flatten(end_dim=-2), center=True)
                feat_img_pca = feat_img @ V

                fig, axs = plt.subplots(2, 3)
                axs[0, 0].set_title('geom feats')
                axs[0, 0].imshow(dpt['bev_data']['data'][0, 0].cpu().numpy())

                axs[0, 1].set_title('bev_feats')
                axs[0, 1].imshow(normalize_dino(lss_viz_data['bev_feats'][0]).cpu().numpy())

                axs[0, 2].set_title('lss occ')
                axs[0, 2].imshow(lss_viz_data['bev_occ'][0].cpu().numpy(), cmap='gray')

                axs[1, 0].set_title('raw image')
                axs[1, 0].imshow(dpt['image']['data'][0].permute(1,2,0).cpu().numpy())

                axs[1, 1].set_title('feat image')
                axs[1, 1].imshow(normalize_dino(feat_img_pca[0, 0]).cpu().numpy())

                axs[1, 2].set_title('depth image')
                axs[1, 2].imshow(lss_viz_data['depth_img'][0, 0].cpu().numpy(), cmap='jet')

                plt.show()
