"""
Collection of metrics for evaluating performance of mexent IRL
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from maxent_irl_maps.dataset.global_state_visitation_buffer import (
    GlobalStateVisitationBuffer,
)
from maxent_irl_maps.networks.baseline_lethal_height import LethalHeightCostmap
from maxent_irl_maps.utils import (
    get_state_visitations,
    quat_to_yaw,
    compute_speedmap_quantile,
    world_to_grid,
)
from maxent_irl_maps.geometry_utils import apply_footprint

def speed_error(
    costmap,
    expert_traj,
    learner_traj,
    expert_state_visitations,
    learner_state_visitations,
):
    midx = min(expert_traj.shape[0], learner_traj.shape[0])
    return (expert_traj[:midx, 3] - learner_traj[:midx, 3]).abs().mean()


def speed_modified_hausdorff_distance(
    costmap,
    expert_traj,
    learner_traj,
    expert_state_visitations,
    learner_state_visitations,
):
    ap = expert_traj[:, 3]
    bp = learner_traj[:, 3]
    dist_mat = (ap.unsqueeze(0) - bp.unsqueeze(1)).abs()
    mhd1 = dist_mat.min(dim=0)[0].mean()
    mhd2 = dist_mat.min(dim=1)[0].mean()
    return max(mhd1, mhd2)


def expert_cost(
    costmap,
    expert_traj,
    learner_traj,
    expert_state_visitations,
    learner_state_visitations,
):
    return (costmap * expert_state_visitations).sum()


def learner_cost(
    costmap,
    expert_traj,
    learner_traj,
    expert_state_visitations,
    learner_state_visitations,
):
    return (costmap * learner_state_visitations).sum()


def position_distance(
    costmap,
    expert_traj,
    learner_traj,
    expert_state_visitations,
    learner_state_visitations,
):
    return torch.linalg.norm(expert_traj[:, :2] - learner_traj[:, :2], dim=-1).sum()


def kl_divergence(
    costmap,
    expert_traj,
    learner_traj,
    expert_state_visitations,
    learner_state_visitations,
):
    # We want learner onto expert
    # KL(p||q) = sum_p[p(x) * log(p(x)/q(x))]
    return (
        learner_state_visitations
        * torch.log(
            (learner_state_visitations + 1e-6) / (expert_state_visitations + 1e-6)
        )
    ).sum()


def modified_hausdorff_distance(
    costmap,
    expert_traj,
    learner_traj,
    expert_state_visitations,
    learner_state_visitations,
):
    ap = expert_traj[:, :2]
    bp = learner_traj[:, :2]
    dist_mat = torch.linalg.norm(ap.unsqueeze(0) - bp.unsqueeze(1), dim=-1)
    mhd1 = dist_mat.min(dim=0)[0].mean()
    mhd2 = dist_mat.min(dim=1)[0].mean()
    return max(mhd1, mhd2)


def pos_speed_modified_hausdorff_distance(
    costmap,
    expert_traj,
    learner_traj,
    expert_state_visitations,
    learner_state_visitations,
):
    ap = expert_traj[:, [0, 1, 3]]
    bp = learner_traj[:, [0, 1, 3]]
    dist_mat = torch.linalg.norm(ap.unsqueeze(0) - bp.unsqueeze(1), dim=-1)
    mhd1 = dist_mat.min(dim=0)[0].mean()
    mhd2 = dist_mat.min(dim=1)[0].mean()
    return max(mhd1, mhd2)
