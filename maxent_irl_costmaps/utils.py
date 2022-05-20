import torch

def get_state_visitations(trajs, map_metadata, weights = None):
    """
    Given a set of trajectories and map metadata, compute the visitations on the map for each traj.
    Args:
        trajs: The trajs to get visit counts of
        map_metadata: The map parameters to get visit counts from
        weights: (optional) the amount to weight each traj by
    """
    if weights is None:
        weights = torch.ones(trajs.shape[0]) / trajs.shape[0]

    xs = trajs[...,0]
    ys = trajs[...,1]
    res = map_metadata['resolution']
    ox = map_metadata['origin'][0]
    oy = map_metadata['origin'][1]
    nx = int(map_metadata['width'] / res)
    ny = int(map_metadata['height'] / res)

    xidxs = ((xs - ox) / res).long()
    yidxs = ((ys - oy) / res).long()

    valid_mask = (xidxs >= 0) & (xidxs < ny) & (yidxs >= 0) & (yidxs < nx)

    binweights = torch.ones(trajs.shape[:-1]) * weights.view(-1, 1)
    flat_binweights = binweights.flatten()

    flat_visits = (nx * yidxs + xidxs).flatten().clamp(0, nx*ny - 1).long()
    flat_visit_counts = torch.bincount(flat_visits, weights=flat_binweights)
    bins = torch.zeros(nx*ny)
    bins[:len(flat_visit_counts)] += flat_visit_counts
    visit_counts = bins.view(nx, ny)
    visitation_probs = visit_counts / visit_counts.sum()

    #debug
#    with torch.no_grad():
#        import matplotlib.pyplot as plt
#        fig, axs = plt.subplots(1, 2)
#        for traj in trajs:
#            axs[0].plot(traj[:, 0], traj[:, 1], c='b', alpha=0.1)
#        axs[0].imshow(visitation_probs, origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
#        axs[1].imshow(visitation_probs, origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
#        plt.show()

    return visitation_probs

def get_feature_counts(traj, map_features, map_metadata):
    """
    Given a (set) of trajectories and map features, compute the features of that trajectory
    """
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
