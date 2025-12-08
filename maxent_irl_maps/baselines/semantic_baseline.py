import torch

def get_terrainnet_costmap(bev_data, bev_fks, cost_config, alpha=0.5, diff_lim=[0., 2.]):
    """
    Implement cost similar to that in TerrainNet (Meng et al. 2023), e.g.
    cost = (1 + alpha*diff) * semantic_cost
    Args:
        bev_data: [BxCxWxH] Tensor of BEV data
        bev_fks: FeatureKeyList of the channels of bev_data
        cost_config: semantic costing config

    Note that it is a deliberate decision to not use BEVGridTorch so that
        we can embed this util in network inference
    """
    assert 'diff' in bev_fks.label

    semantic_costmap = get_semantic_costmap(bev_data, bev_fks, cost_config)

    diff_idx = bev_fks.index('diff')
    _diff = bev_data[:, diff_idx].clip(*diff_lim)
    _diff_scale = 1. + alpha * _diff

    costmap = semantic_costmap * _diff_scale

    return costmap

def get_semantic_costmap(bev_data, bev_fks, cost_config):
    """
    Implement a simple semantics->costing baseline
    Args:
        bev_data: [BxCxWxH] Tensor of BEV data
        bev_fks: FeatureKeyList of the channels of bev_data
        cost_config: semantic costing config

    Note that it is a deliberate decision to not use BEVGridTorch so that
        we can embed this util in network inference
    """
    semantic_classes = list(cost_config['costs'].keys())
    assert all([k in bev_fks.label for k in semantic_classes])

    _scosts = torch.tensor(
        [cost_config['costs'][k] for k in semantic_classes],
        device = bev_data.device
    ).view(1, -1, 1, 1)

    semantic_idxs = [bev_fks.index(k) for k in semantic_classes]
    _semantic_map = bev_data[:, semantic_idxs].softmax(dim=1)

    costmap = (_semantic_map * _scosts).sum(dim=1)

    return costmap