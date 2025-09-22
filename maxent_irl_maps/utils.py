import torch
import torch_scatter

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

    _k = (q - _cdf_low) / ((_cdf_high - _cdf_low)+1e-8)

    _speedmap_low = speed_bins[_qidx]
    _speedmap_high = speed_bins[_qidx + 1]

    _speedmap = (1 - _k) * _speedmap_low + _k * _speedmap_high

    return _speedmap.reshape(*speedmap_cdf.shape[1:])

def get_speedmap(trajs, speeds, metadata):
    """
    Given a set of trajectories, produce a map where each cell contains the speed the traj in that cell
    Args:
        trajs: the trajs to compute speeds over
        speeds: the speeds corresponding to pos
        map_metadata: The map params to get speeds over
    """
    assert (metadata.N[:, 0] * metadata.N[:, 1] == metadata.N[0, 0] * metadata.N[0, 1]).all()

    B = trajs.shape[0]
    N = trajs.shape[-1]
    ncells = metadata.N[0, 0] * metadata.N[0, 1]

    trajs_flat = trajs.view(B, -1, N)
    speeds_flat = speeds.view(B, -1)
    
    xys = trajs_flat[:, :, :2]

    gxys, valid_mask = world_to_grid(xys, metadata)

    gxys[~valid_mask] = 0
    speeds_flat[~valid_mask] = 0.

    #TODO double-check i didnt transpose the raster
    raster_idxs = gxys[:, :, 0] * metadata.N[:, 1].view(B, 1) + gxys[:, :, 1]

    # better to enforce that unvisited = 0
    speedmap_flat = torch.zeros(B, ncells, device=trajs.device)

    torch_scatter.scatter(
        src = speeds_flat,
        index = raster_idxs,
        out = speedmap_flat,
        dim=-1,
        reduce = 'max'
    )

    speedmap = speedmap_flat.reshape(B, metadata.N[0, 0], metadata.N[0, 1])

    return speedmap

def world_to_grid(trajs, metadata):
    """
    Args:
        trajs: [B x ... x N] Tensor of trajs
        metadata: [B] stack of metadata
    """
    tshape = trajs.shape
    B = tshape[0]

    xys = trajs[..., :2]
    xys_flat = xys.view(B, -1, 2)
    
    _o = metadata.origin.unsqueeze(1)
    _l = metadata.length.unsqueeze(1)
    _r = metadata.resolution.unsqueeze(1)

    gxys_flat = (xys_flat - _o) / _r

    valid_mask_flat = (xys_flat > (_o+1e-6)).all(dim=-1) & (xys_flat < (_o+_l-1e-6)).all(dim=-1)

    gxys = gxys_flat.reshape(*tshape[:-1], 2)
    valid_mask = valid_mask_flat.reshape(*tshape[:-1])

    return gxys.long(), valid_mask

def get_state_visitations(trajs, metadata, weights=None):
    """
    Given a set of trajectories and map metadata, compute the visitations on the map for each traj.
    Args:
        trajs: [B x ... x N] Tensor of trajs to compute visit counts for (we will collapse all the middle dims)
        map_metadata: The map parameters to get visit counts from
        weights: [B x ...] Tensor (optional) the amount to weight each traj by
    """
    assert (metadata.N[:, 0] * metadata.N[:, 1] == metadata.N[0, 0] * metadata.N[0, 1]).all()

    B = trajs.shape[0]
    N = trajs.shape[-1]
    ncells = metadata.N[0, 0] * metadata.N[0, 1]

    trajs_flat = trajs.view(B, -1, N)
    
    if weights is None:
        weights = torch.ones(B, trajs_flat.shape[1], device=trajs.device)
    else:
        weights = weights.view(B, -1)

    xys = trajs_flat[:, :, :2]

    gxys, valid_mask = world_to_grid(xys, metadata)

    gxys[~valid_mask] = 0
    weights[~valid_mask] = 0.

    #TODO double-check i didnt transpose the raster
    raster_idxs = gxys[:, :, 0] * metadata.N[:, 1].view(B, 1) + gxys[:, :, 1]

    flat_visitations = torch.zeros(B, ncells, device=trajs.device)

    torch_scatter.scatter(
        src = weights,
        index = raster_idxs,
        out = flat_visitations,
        dim=-1,
        reduce = 'sum'
    )

    flat_visitations /= flat_visitations.sum(dim=-1, keepdims=True)

    visitations = flat_visitations.reshape(B, metadata.N[0, 0], metadata.N[0, 1])

    return visitations

def clip_to_map_bounds(traj, metadata):
    """
    Given traj, find last point (temporally) in the map bounds

    Args:
        traj: [BxTxN] tensor of trajs
        metadata: [B] stack of LocalMapperMetadata
    """
    B, T, _ = traj.shape

    xs = traj[..., 0]
    ys = traj[..., 1]
    #unpack along T
    xmin = metadata.origin[..., [0]]
    ymin = metadata.origin[..., [1]]
    xmax = xmin + metadata.length[..., [0]]
    ymax = ymin + metadata.length[..., [1]]

    in_bounds = (xs >= xmin) & (xs < xmax) & (ys >= ymin) & (ys < ymax)
    idxs = torch.arange(xs.shape[-1], device=traj.device).unsqueeze(0).tile(B, 1)

    idx = (idxs * in_bounds).argmax(dim=-1)
    ibs = torch.arange(B, device=traj.device)

    return traj[ibs, idx]

def dict_to(d1, device):
    if isinstance(d1, dict):
        return {k: dict_to(v, device) for k, v in d1.items()}
    else:
        return d1.to(device)
