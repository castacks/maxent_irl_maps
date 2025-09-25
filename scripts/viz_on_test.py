import os
import yaml
import torch
import argparse
import matplotlib.pyplot as plt

import matplotlib; matplotlib.use("TkAgg")

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_maps.experiment_management.parse_configs import setup_experiment, load_net_for_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_fp", type=str, required=True, help="model to load")
    parser.add_argument(
        "--test_fp", type=str, required=True, help="path to preproc data"
    )
    parser.add_argument(
        "--n", type=int, required=False, default=10, help="number of viz to run"
    )
    parser.add_argument(
        "--device", type=str, required=False, default="cpu", help="the device to run on"
    )
    args = parser.parse_args()

    model_base_dir = os.path.split(args.model_fp)[0]
    config_fp = os.path.join(model_base_dir, '_params.yaml')
    config = yaml.safe_load(open(config_fp, 'r'))

    config['dataset']['common']['root_dir'] = args.test_fp

    res = setup_experiment(config, skip_norms=True)["algo"].to(args.device)
    res.network.load_state_dict(torch.load(args.model_fp, weights_only=True, map_location=args.device))
    res.network.eval()

    idxs = torch.randperm(len(res.expert_dataset))

    for i in range(args.n):
        idx = idxs[i]

        fig, axs = res.visualize(idx=idx)

        if False:
            ## show feat image
            from physics_atv_visual_mapping.utils import normalize_dino

            axs[-1].cla()
            img = res.expert_dataset[idx]["feature_image"]
            img = img['data'].unsqueeze(0)
            with torch.no_grad():
                feat_img = res.network.voxel_recolor.feat_net(img)[0].permute(1,2,0)

            U,S,V = torch.pca_lowrank(feat_img.flatten(end_dim=-2))

            feat_img_pca = feat_img @ V

            axs[-1].imshow(normalize_dino(feat_img_pca).cpu().numpy())

        plt.show()

        if False:
            ## LSS viz code (prob shouldnt be here lol)
            from physics_atv_visual_mapping.utils import normalize_dino

            with torch.no_grad():
                dpt = res.expert_dataset.getitem_batch([idx])
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
