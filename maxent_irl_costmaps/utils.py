import torch

def get_feature_counts(traj, map_features, map_metadata):
    xs = traj[...,0]
    ys = traj[...,1]
    res = map_metadata['resolution']
    ox = map_metadata['origin'][0]
    oy = map_metadata['origin'][1]

    xidxs = ((xs - ox) / res).long()
    yidxs = ((ys - oy) / res).long()

    valid_mask = (xidxs >= 0) & (xidxs < map_features.shape[2]) & (yidxs >= 0) & (yidxs < map_features.shape[1])

    xidxs[~valid_mask] = 0
    yidxs[~valid_mask] = 0

    # map data is transposed
    features = map_features[:, yidxs, xidxs]

    feature_counts = features.mean(dim=-1)
    return feature_counts
