"""
Simple script to test some hausdorff distance stuff

Do this by sampling a number of trajectories and comparing hausdorff distance
"""

import torch
import matplotlib.pyplot as plt

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
from torch_mpc.models.steer_setpoint_throttle_kbm import SteerSetpointThrottleKBM


def hausdorff_distance(A, B, directed_distance=0, symmetric_distance=0):
    """
    A: [B x 2+] set of points
    B: [B x 2+] set of points
    directed_distance: 1-6 for the following:
        0: min d(a, B)
        1: q0.5 d(a, B)
        2: q0.75 d(a, B)
        3: q0.9 d(a, B)
        4: max d(a, B)
        5: mean d(a, B)
    symmetric_distance: 1-4 for the following
        0: min(d(A, B), d(B, A))
        1: max(d(A, B), d(B, A))
        2: 0.5 * (d(A, B) + d(B, A))
        3: (Na * d(A, B) + Nb * d(B, A)) / Na + Nb
    """
    ab_dist_matrix = torch.linalg.norm(
        A.unsqueeze(1)[..., :2] - B.unsqueeze(0)[..., :2], dim=-1
    )
    a_to_B_dist = torch.min(ab_dist_matrix, dim=1)[0]
    b_to_A_dist = torch.min(ab_dist_matrix, dim=0)[0]

    if directed_distance == 0:
        A_to_B_dist = a_to_B_dist.min()
        B_to_A_dist = b_to_A_dist.min()
    elif directed_distance == 1:
        A_to_B_dist = torch.quantile(a_to_B_dist, 0.5)
        B_to_A_dist = torch.quantile(b_to_A_dist, 0.5)
    elif directed_distance == 2:
        A_to_B_dist = torch.quantile(a_to_B_dist, 0.75)
        B_to_A_dist = torch.quantile(b_to_A_dist, 0.75)
    elif directed_distance == 3:
        A_to_B_dist = torch.quantile(a_to_B_dist, 0.9)
        B_to_A_dist = torch.quantile(b_to_A_dist, 0.9)
    elif directed_distance == 4:
        A_to_B_dist = a_to_B_dist.max()
        B_to_A_dist = b_to_A_dist.max()
    elif directed_distance == 5:
        A_to_B_dist = a_to_B_dist.mean()
        B_to_A_dist = b_to_A_dist.mean()

    if symmetric_distance == 0:
        dist = min(A_to_B_dist, B_to_A_dist)
    elif symmetric_distance == 1:
        dist = max(A_to_B_dist, B_to_A_dist)
    elif symmetric_distance == 2:
        dist = 0.5 * (A_to_B_dist + B_to_A_dist)
    elif symmetric_distance == 3:
        dist = (A_to_B_dist * A.shape[0] + B_to_A_dist * B.shape[0]) / (
            A.shape[0] + B.shape[0]
        )

    return dist


if __name__ == "__main__":
    kbm = SteerSetpointThrottleKBM(
        L=3.0,
        throttle_lim=[1.5, 3.5],
        steer_lim=[-0.52, 0.52],
        steer_rate_lim=0.2,
        dt=0.15,
    )

    X0 = torch.tensor([0.0, 0.0, 0.0, 3.5, 0.0])
    action_seqs = torch.rand([10, 75, 2])
    action_seqs[..., 1] -= 0.5
    action_seqs[..., 1] *= 2.0

    #    action_seqs[:, :, 0] = 3.5
    #    for i,s in enumerate(torch.linspace(-0.2, 0.2, 10)):
    #        action_seqs[i, :, 1] = s

    trajs = torch.stack(
        [kbm.rollout(X0, action_seqs[i]) for i in range(action_seqs.shape[0])], dim=0
    )

    traj_dists = torch.zeros(trajs.shape[0], 24)

    # plot hausdorff distances to the first traj
    for k, traj in enumerate(trajs):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for t in trajs:
            axs[0].plot(t[:, 0], t[:, 1], marker=".", c="b")

        axs[0].plot(trajs[0, :, 0], trajs[0, :, 1], marker=".", c="r")
        axs[0].plot(trajs[k, :, 0], trajs[k, :, 1], marker=".", c="g")

        dists = []
        for i in range(6):
            for j in range(4):
                dist = hausdorff_distance(trajs[0], traj, i, j)
                dists.append(dist)
                print("traj {}: dist metric {} = {:.4f}".format(k, 4 * i + j + 1, dist))
                traj_dists[k, 4 * i + j] = dist

        axs[1].bar(range(1, 25), dists)

        plt.title("Distance by traj pair")
        plt.show()

    for i in range(traj_dists.shape[1]):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for j, traj in enumerate(trajs):
            axs[0].plot(traj[:, 0], traj[:, 1], c="r" if j == 0 else "b")
            axs[0].text(
                traj[-1, 0], traj[-1, 1], "{}, ({:.2f})".format(j, traj_dists[j, i])
            )
        axs[1].bar(range(trajs.shape[0]), traj_dists[:, i])
        plt.title("Distance by metric {}".format(i + 1))
        plt.show()
