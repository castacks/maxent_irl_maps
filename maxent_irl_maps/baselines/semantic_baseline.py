import torch

def get_terrainnet_costmap(bev_data, cost_coeffs, alpha, bev_fks, cost_fks, diff_lim=[0., 2.]):
    """
    Implement cost similar to that in TerrainNet (Meng et al. 2023), e.g.
    cost = (1 + alpha*diff) * semantic_cost
    Args:
        bev_data: [BxCxWxH] Tensor of BEV data
        cost_coeffs: [C] Tensor of cost coeffs
        alpha: Tensor of dif scale coeff (shoud be nonnegative)
        bev_fks: FeatureKeyList of the channels of bev_data
        cost_fks: FeatureKeyList of the channels of cost_coeffs

    Note that it is a deliberate decision to not use BEVGridTorch so that
        we can embed this util in network inference
    """
    assert 'diff' in bev_fks.label

    semantic_costmap = get_semantic_costmap(bev_data, cost_coeffs, bev_fks, cost_fks)

    diff_idx = bev_fks.index('diff')
    _diff = bev_data[:, diff_idx].clip(*diff_lim)
    _diff_scale = 1. + alpha * _diff

    costmap = semantic_costmap * _diff_scale

    return costmap

def get_semantic_costmap(bev_data, cost_coeffs, bev_fks, cost_fks):
    """
    Implement a simple semantics->costing baseline
    Args:
        bev_data: [BxCxWxH] Tensor of BEV data (containing semantic LOGITS)
        cost_coeffs: [C] Tensor of cost coeffs
        bev_fks: FeatureKeyList of the channels of bev_data
        cost_fks: FeatureKeyList of the channels of cost_coeffs

    Note that it is a deliberate decision to not use BEVGridTorch so that
        we can embed this util in network inference
    """
    assert all([k in bev_fks.label for k in cost_fks.label])

    _scosts = cost_coeffs.view(1, -1, 1, 1)

    semantic_idxs = [bev_fks.label.index(k) for k in cost_fks.label]
    _semantic_map = bev_data[:, semantic_idxs].softmax(dim=-3)

    costmap = (_semantic_map * _scosts).sum(dim=-3)

    return costmap