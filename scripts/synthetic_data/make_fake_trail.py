import torch
import matplotlib.pyplot as plt
import argparse
import os

from torch_mpc.models.skid_steer import SkidSteer

"""
Simple code for generating fake 'trail' features and trajectories.
"""

def expand_trajs(trajs):
    """
    go from x,y,th to
    x,y,z,qx,qy,qz,qw
    """
    x, y, th = trajs.moveaxis(-1, 0)
    z = torch.zeros_like(x)
    qw = torch.cos(th/2.)
    qx = torch.zeros_like(x)
    qy = torch.zeros_like(y)
    qz = torch.sin(th/2.)
    return torch.stack([x, y, z, qx, qy, qz, qw], axis=-1)

def get_feature_maps_from_trajs(trajs, metadata, obstacle_radius = 1.0):
    """
    Get obstacle and position maps from trajs assuming that everything more than
    obstacle radius away from the trajs is obstacle

    Args:
        trajs: The trajectories to infer obstacles from
        metadata: The metadata of the maps to produce
        obstacle_radius: Cells within this distance of any traj will be considered free
    """
    feature_keys = [
        'height_high'
    ]

    res = metadata['resolution']
    width = metadata['width']
    height = metadata['height']
    ox = metadata['origin'][0]
    oy = metadata['origin'][1]

    xs = ox + torch.arange(start=0., end=width, step=res)
    ys = oy + torch.arange(start=0., end=height, step=res)

    positions = torch.meshgrid(xs, ys, indexing='ij')
    positions = torch.stack(positions, axis=-1) #[x by y]

    distances = torch.linalg.norm(positions, axis=-1)
    pos_x = positions[..., 0]
    pos_y = positions[..., 1]

    obstacle_acc = torch.ones_like(distances).bool()

    #iterate through trajectories to save memory
    for traj in trajs:
        traj_poses = traj[:, :2].unsqueeze(1) #[T x G x 2]
        map_poses = positions.view(-1, 2).unsqueeze(0) #[T x G x 2]
        dists = (traj_poses - map_poses).pow(2).sum(dim=-1).sqrt().min(dim=0)[0]
        dists_map = dists.view(obstacle_acc.shape)
        mask = (dists_map > obstacle_radius)
        obstacle_acc = obstacle_acc & mask

    obstacles = obstacle_acc.float()

    features = torch.stack([
        obstacles.T
    ], axis=0)

    return features, feature_keys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--both', action='store_true')
    parser.add_argument('--save_dir', required=True, help='location to save the temp data')
    args = parser.parse_args()
    
    model = SkidSteer()

    x0 = torch.zeros(model.observation_space().shape[0])
    U1 = torch.zeros(100, 2)
    U1[:, 0] = 1.5
    U1[30:50, 1] = 0.2

    U2 = U1.clone()
    U2[:, 1] *= -1.


    metadata = {
        'width': torch.tensor(30.),
        'height': torch.tensor(30.),
        'resolution': torch.tensor(0.25),
        'origin': torch.tensor([-15., -15.])
    }

    left_trajs = []
    right_trajs = []

    for i in range(10):
        z = torch.randn_like(U1) * 0.05
        traj = model.rollout(x0, U1 + z)
        traj2 = model.rollout(x0, U2 + z)
        left_trajs.append(traj)
        right_trajs.append(traj2)

    
    left_trajs = expand_trajs(torch.stack(left_trajs, dim=0))
    right_trajs = expand_trajs(torch.stack(right_trajs, dim=0))
    all_trajs = torch.cat([left_trajs, right_trajs], dim=0)

    feature_maps, feature_keys = get_feature_maps_from_trajs(all_trajs, metadata)
    left_feature_maps, feature_keys = get_feature_maps_from_trajs(left_trajs, metadata)
    right_feature_maps, feature_keys = get_feature_maps_from_trajs(right_trajs, metadata)

    feature_keys = ['height_high', 'height_high_left', 'height_high_right']
    feature_maps = torch.cat([feature_maps, left_feature_maps, right_feature_maps], axis=0)

    dataset = []
    for traj in left_trajs:
        datapt = {
            'metadata': metadata,
            'feature_keys': feature_keys,
            'map_features': feature_maps,
            'traj': traj
        }
        dataset.append(datapt)

    if args.both:
        for traj in right_trajs:
            datapt = {
                'metadata': metadata,
                'feature_keys': feature_keys,
                'map_features': feature_maps,
                'traj': traj
            }
            dataset.append(datapt)

    if os.path.exists(args.save_dir):
        x = input('{} already exists. Preprocess again? [y/N]'.format(args.save_dir))
    else:
        os.mkdir(args.save_dir)

    for i, x in enumerate(dataset):
        fp = os.path.join(args.save_dir, 'traj_{}.pt'.format(i))
        torch.save(x, fp)
