import os
import tqdm
import yaml
import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_maps.experiment_management.parse_configs import setup_experiment, load_net_for_eval
from maxent_irl_maps.utils import get_state_visitations
from torch_mpc.cost_functions.cost_terms.utils import apply_footprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_fp", type=str, required=True, help="model to load")
    parser.add_argument(
        "--test_fp", type=str, required=True, help="path to preproc data"
    )
    parser.add_argument(
        "--device", type=str, required=False, default="cpu", help="the device to run on"
    )
    args = parser.parse_args()

    res = load_net_for_eval(args.model_fp, device=args.device, skip_mpc=False)

    dataset = MaxEntIRLDataset(
        root_fp=args.test_fp, feature_keys=res.expert_dataset.feature_keys
    ).to(args.device)
    res.expert_dataset = dataset

    expert_costs = []

    for i in tqdm.tqdm(range(len(res.expert_dataset))):
        idx = np.random.randint(len(res.expert_dataset))

        with torch.no_grad():
            data = res.expert_dataset[idx]

            # hack back to single dim
            map_features = torch.stack([data["map_features"]] * res.mppi.B, dim=0)
            metadata = data["metadata"]
            xmin = metadata["origin"][0].cpu()
            ymin = metadata["origin"][1].cpu()
            xmax = xmin + metadata["length"][0].cpu()
            ymax = ymin + metadata["length"][1].cpu()
            expert_traj = data["traj"]

            cres = res.network.forward(map_features, return_mean_entropy=True)

            costmap = cres["costmap"]

            X_expert = {
                "state": expert_traj,
                "steer_angle": data["steer"].unsqueeze(-1)
                if "steer" in data.keys()
                else torch.zeros(res.mppi.B, res.mppi.T, 1, device=initial_states.device),
            }
            expert_kbm_traj = res.mppi.model.get_observations(X_expert)

            footprint_expert_traj = apply_footprint(
                expert_kbm_traj.unsqueeze(0), res.footprint
            ).view(1, -1, 2)

            esv = get_state_visitations(footprint_expert_traj, metadata)

            """
            #debug viz
            fig, axs = plt.subplots(1, 2)

            axs[0].imshow(costmap[0,0].T.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
            axs[0].plot(expert_kbm_traj[:, 0], expert_kbm_traj[:, 1], c='y')

            axs[1].imshow(esv.T.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
            axs[1].plot(expert_kbm_traj[:, 0], expert_kbm_traj[:, 1], c='y')

            plt.show()
            """

            mask = esv > 1e-4

            ecosts = costmap[0, 0][mask]

            expert_costs.append(ecosts.flatten())

    expert_costs = torch.cat(expert_costs).cpu().numpy()

    cost_q95 = np.quantile(expert_costs, 0.95)
    cost_q99 = np.quantile(expert_costs, 0.99)
    cost_q995 = np.quantile(expert_costs, 0.995)

    plt.title("Expert cost distribution")
    plt.hist(expert_costs, cumulative=True, histtype='step', density=True, bins=100)

    plt.axvline(cost_q95, color='r', linestyle='dotted', label='expert q95 ({:.4f})'.format(cost_q95))
    plt.axvline(cost_q99, color='r', linestyle='dashed', label='expert q99 ({:.4f})'.format(cost_q99))
    plt.axvline(cost_q995, color='r', linestyle='solid', label='expert q995 ({:.4f})'.format(cost_q995))

    plt.legend()

    plt.xlabel('Cost')
    plt.ylabel('Expert CDF')
    plt.show()

    #viz loop
    for i in range(len(res.expert_dataset)):
        idx = np.random.randint(len(res.expert_dataset))

        with torch.no_grad():
            data = res.expert_dataset[idx]

            # hack back to single dim
            map_features = torch.stack([data["map_features"]] * res.mppi.B, dim=0)
            metadata = data["metadata"]
            xmin = metadata["origin"][0].cpu()
            ymin = metadata["origin"][1].cpu()
            xmax = xmin + metadata["length"][0].cpu()
            ymax = ymin + metadata["length"][1].cpu()
            expert_traj = data["traj"]

            cres = res.network.forward(map_features, return_mean_entropy=True)

            costmap = cres["costmap"]

            fig, axs = plt.subplots(2, 2, figsize=(16, 16))

            plt.title('Expert should not go through red cells')

            axs[0, 0].set_title('Costmap')
            axs[0, 0].imshow(costmap[0,0].T.cpu(), origin='lower', cmap='jet', extent=(xmin, xmax, ymin, ymax))
            axs[0, 0].plot(expert_traj[:, 0].cpu().numpy(), expert_traj[:, 1].cpu().numpy(), c='y')

            axs[0, 1].set_title('Costmap q95 ({:.4f})'.format(cost_q95))
            axs[0, 1].imshow(costmap[0,0].T.cpu(), origin='lower', cmap='jet', extent=(xmin, xmax, ymin, ymax), vmax=cost_q95)
            axs[0, 1].plot(expert_traj[:, 0].cpu().numpy(), expert_traj[:, 1].cpu().numpy(), c='y')

            axs[1, 0].set_title('Costmap q99 ({:.4f})'.format(cost_q99))
            axs[1, 0].imshow(costmap[0,0].T.cpu(), origin='lower',  cmap='jet', extent=(xmin, xmax, ymin, ymax), vmax=cost_q99)
            axs[1, 0].plot(expert_traj[:, 0].cpu().numpy(), expert_traj[:, 1].cpu().numpy(), c='y')

            axs[1, 1].set_title('Costmap q995 ({:.4f})'.format(cost_q995))
            axs[1, 1].imshow(costmap[0,0].T.cpu(), origin='lower', cmap='jet', extent=(xmin, xmax, ymin, ymax), vmax=cost_q995)
            axs[1, 1].plot(expert_traj[:, 0].cpu().numpy(), expert_traj[:, 1].cpu().numpy(), c='y')

            plt.show()