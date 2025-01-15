import torch

def compute_map_mean_entropy(logits, bin_edges):
    """
    Compute the mean and entropy of a map represented as a categorical

    Args:
        logits: a BxLxWxH tensor of map logits
    """
    _bin_edges = (bin_edges[1:]+bin_edges[:-1])/2.
    _bin_edges = _bin_edges.view(1,-1,1,1)

    probs = logits.softmax(dim=1)

    mean = (probs * _bin_edges).sum(dim=1, keepdim=True)
    entropy = (probs * -probs.log()).sum(dim=1, keepdim=True)

    return mean, entropy

def quat_to_yaw(quat):
    """
    Convert a quaternion (as [x, y, z, w]) to yaw
    """
    if len(quat.shape) < 2:
        return quat_to_yaw(quat.unsqueeze(0)).squeeze()

    return torch.atan2(
        2 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1]),
        1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2),
    )

def compute_map_cvar(maps, cvar):
    if cvar < 0.0:
        map_q = torch.quantile(maps, 1.0 + cvar, dim=0)
        mask = maps <= map_q.view(1, *map_q.shape)
    else:
        map_q = torch.quantile(maps, cvar, dim=0)
        mask = maps >= map_q.view(1, *map_q.shape)

    cvar_map = (maps * mask).sum(dim=0) / mask.sum(dim=0)
    return cvar_map

def compute_speedmap_quantile(speedmap_cdf, speed_bins, q):
    """
    Given a speedmap (parameterized as cdf + bins) and a quantile,
        compute the speed corresponding to that quantile

    Args:
        speedmap_cdf: BxWxH Tensor of speedmap cdf
        speed_bins: B+1 Tensor of bin edges
        q: float containing the quantile
    """
    B = len(speed_bins)
    # need to stack zeros to front
    _cdf = torch.cat([torch.zeros_like(speedmap_cdf[[0]]), speedmap_cdf], dim=0)

    _cdf = _cdf.view(B, -1)

    _cdiffs = q - _cdf
    _mask = _cdiffs <= 0

    _qidx = (_cdiffs + 1e10 * _mask).argmin(dim=0)

    _cdf_low = _cdf.T[torch.arange(len(_qidx)), _qidx]
    _cdf_high = _cdf.T[torch.arange(len(_qidx)), _qidx + 1]

    _k = (q - _cdf_low) / (_cdf_high - _cdf_low)

    _speedmap_low = speed_bins[_qidx]
    _speedmap_high = speed_bins[_qidx + 1]

    _speedmap = (1 - _k) * _speedmap_low + _k * _speedmap_high

    return _speedmap.reshape(*speedmap_cdf.shape[1:])


def get_speedmap(trajs, speeds, map_metadata, weights=None):
    """
    Given a set of trajectories, produce a map where each cell contains the speed the traj in that cell
    Args:
        trajs: the trajs to compute speeds over
        speeds: the speeds corresponding to pos
        map_metadata: The map params to get speeds over
        weights: optional weighting on each trajectory
    """
    if weights is None:
        weights = torch.ones(trajs.shape[0], device=trajs.device) / trajs.shape[0]

    xs = trajs[..., 0]
    ys = trajs[..., 1]
    ox = map_metadata["origin"][0].item()
    oy = map_metadata["origin"][1].item()

    if isinstance(map_metadata["resolution"], torch.Tensor):
        res = map_metadata["resolution"].item()
        nx = round(map_metadata["width"].item() / res)
        ny = round(map_metadata["height"].item() / res)
    else:
        res = map_metadata["resolution"]
        nx = round(map_metadata["width"] / res)
        ny = round(map_metadata["height"] / res)

    width = max(nx, ny)

    xidxs = ((xs - ox) / res).long()
    yidxs = ((ys - oy) / res).long()

    # 1 iff. valid
    valid_mask = (xidxs >= 0) & (xidxs < ny) & (yidxs >= 0) & (yidxs < nx)

    # I'm pretty sure all I have to do is replace the ones from svs w/ the actual speeds
    binweights = (
        torch.ones(trajs.shape[:-1], device=trajs.device)
        * weights.view(-1, 1)
        * valid_mask.float()
    )
    speed_binweights = speeds * binweights

    flat_binweights = binweights.flatten()
    flat_speed_binweights = speed_binweights.flatten()

    flat_speeds = (ny * xidxs + yidxs).flatten().clamp(0, nx * ny - 1).long()
    flat_speed_counts = torch.bincount(flat_speeds, weights=flat_speed_binweights)

    flat_visits = (ny * xidxs + yidxs).flatten().clamp(0, nx * ny - 1).long()
    flat_visit_counts = torch.bincount(flat_visits, weights=flat_binweights) + 1e-6

    bins = torch.zeros(nx * ny, device=trajs.device)
    bins[: len(flat_speed_counts)] += flat_speed_counts / flat_visit_counts
    speed_counts = bins.view(nx, ny)

    # # debug
    # with torch.no_grad():
    #     idx_diffs = (((nx * yidxs + xidxs)[..., 1:] - (nx * yidxs + xidxs)[..., :-1]).abs() > 1e-4).float()

    #     import matplotlib.pyplot as plt
    #     fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    #     axs = axs.flatten()
    #     for traj, sp, diff in zip(trajs, speeds, idx_diffs):
    #         axs[0].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='b', alpha=0.5)
    #         axs[2].plot(sp.cpu())
    #         axs[2].scatter(torch.arange(len(diff)) + 1, diff.cpu() * sp[1:].cpu(), c='r', s=2.)
    #     axs[0].imshow(speed_counts.T.cpu(), origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
    #     axs[1].imshow(speed_counts.T.cpu(), origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
    #     plt.show()

    return speed_counts

def world_to_grid(trajs, map_metadata):
    xs = trajs[..., 0]
    ys = trajs[..., 1]
    ox = map_metadata["origin"][0].item()
    oy = map_metadata["origin"][1].item()

    if isinstance(map_metadata["resolution"], torch.Tensor):
        res = map_metadata["resolution"].item()
        nx = round(map_metadata["width"].item() / res)
        ny = round(map_metadata["height"].item() / res)
    else:
        res = map_metadata["resolution"]
        nx = round(map_metadata["width"] / res)
        ny = round(map_metadata["height"] / res)

    width = max(nx, ny)

    xidxs = ((xs - ox) / res).long()
    yidxs = ((ys - oy) / res).long()

    # 1 iff. valid
    coords = torch.stack([xidxs, yidxs], dim=-1)
    valid_mask = (xidxs >= 0) & (xidxs < ny) & (yidxs >= 0) & (yidxs < nx)
    return coords, valid_mask


def get_state_visitations(trajs, map_metadata, weights=None):
    """
    Given a set of trajectories and map metadata, compute the visitations on the map for each traj.
    Args:
        trajs: The trajs to get visit counts of
        map_metadata: The map parameters to get visit counts from
        weights: (optional) the amount to weight each traj by
    """
    if weights is None:
        weights = torch.ones(trajs.shape[0], device=trajs.device) / trajs.shape[0]

    xs = trajs[..., 0]
    ys = trajs[..., 1]
    ox = map_metadata["origin"][0].item()
    oy = map_metadata["origin"][1].item()

    if isinstance(map_metadata["resolution"], torch.Tensor):
        res = map_metadata["resolution"].item()
        nx = round(map_metadata["width"].item() / res)
        ny = round(map_metadata["height"].item() / res)
    else:
        res = map_metadata["resolution"]
        nx = round(map_metadata["width"] / res)
        ny = round(map_metadata["height"] / res)

    width = max(nx, ny)

    xidxs = ((xs - ox) / res).long()
    yidxs = ((ys - oy) / res).long()

    # 1 iff. valid
    valid_mask = (xidxs >= 0) & (xidxs < ny) & (yidxs >= 0) & (yidxs < nx)

    binweights = (
        torch.ones(trajs.shape[:-1], device=trajs.device)
        * weights.view(-1, 1)
        * valid_mask.float()
    )
    flat_binweights = binweights.flatten()

    flat_visits = (ny * xidxs + yidxs).flatten().clamp(0, nx * ny - 1).long()
    flat_visit_counts = torch.bincount(flat_visits, weights=flat_binweights)
    bins = torch.zeros(nx * ny, device=trajs.device)
    bins[: len(flat_visit_counts)] += flat_visit_counts
    visit_counts = bins.view(nx, ny)
    visitation_probs = visit_counts / visit_counts.sum()

    # debug
    #    with torch.no_grad():
    #        import matplotlib.pyplot as plt
    #        fig, axs = plt.subplots(1, 2)
    #        for traj in trajs:
    #            axs[0].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='b', alpha=0.1)
    #        axs[0].imshow(visitation_probs.cpu(), origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
    #        axs[1].imshow(visitation_probs.cpu(), origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
    #        plt.show()

    return visitation_probs

def dict_to(d1, device):
    if isinstance(d1, dict):
        return {k: dict_to(v, device) for k, v in d1.items()}
    else:
        return d1.to(device)
