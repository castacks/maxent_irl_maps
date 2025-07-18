import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import scipy.spatial
import scipy.interpolate

from torch.utils.data import DataLoader

from torch_mpc.cost_functions.cost_terms.utils import apply_footprint

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_maps.utils import get_state_visitations, get_speedmap, compute_map_mean_entropy, compute_speedmap_quantile

class MPPIIRLSpeedmaps:
    """
    This is the same as MPPI IRL, but in addition to the IRL, also learn a speed map via MLE to expert speed
    Speedmap Learning:
        1. Run the network to get the per-cell speed distribution
        2. Create a speed label for each cell that the expert visited (have to register speeds/traj onto the map)
        3. Compute a masked MLE for the cells that were visited
        4. backward pass w/ the IRL grad
        5. win

    MPPI IRL:
        Costmap learner that uses expert data + MPPI optimization to learn costmaps.
        The algorithm is as follows:
        1. Get empirical feature counts across the entire dataset for expert
        2. Iteratively
            a. Sample a batch of data from the expert's dataset
            b. Compute a set of costmaps from the current weight vector.
            c. Use MPPI to optimize a trajectory on the costmap
            d. Get empirical feature counts from the MPPI solver (maybe try the weighted trick)
        e. Match feature expectations
    """

    def __init__(
        self,
        network,
        opt,
        expert_dataset,
        mppi,
        footprint,
        mppi_itrs=10,
        batch_size=64,
        speed_coeff=1.0,
        reg_coeff=1e-2,
        grad_clip=1.0,
        device="cpu",
    ):
        """
        Args:
            network: the network to use for predicting costmaps
            opt: the optimizer for the network
            expert_dataset: The dataset containing expert demonstrations to imitate
            footprint: "smear" state visitations with this
            mppi: The MPPI object to optimize with
        """
        self.expert_dataset = expert_dataset
        self.footprint = footprint
        self.mppi = mppi
        self.mppi_itrs = mppi_itrs

        self.network = network

        print(self.network)
        print("({} params)".format(sum([x.numel() for x in self.network.parameters()])))
        print(expert_dataset.feature_keys)
        self.network_opt = opt

        self.batch_size = batch_size
        self.reg_coeff = reg_coeff
        self.speed_coeff = speed_coeff
        self.grad_clip = grad_clip

        self.itr = 0
        self.device = device

    def update(self, n=-1):
        """
        High-level method that runs training for one epoch.
        """
        self.itr += 1
        dl = DataLoader(self.expert_dataset, batch_size=self.batch_size, shuffle=True)
        for i, batch in enumerate(dl):
            if n > -1 and i >= n:
                break

            # skip the last batch in the dataset as MPPI batching forces a fixed size
            if batch["traj"].shape[0] < self.batch_size:
                break

            print(
                "{}/{}".format(i + 1, int(len(self.expert_dataset) / self.batch_size)),
                end="\r",
            )
            self.gradient_step(batch)

        print("_____ITR {}_____".format(self.itr))

    def gradient_step(self, batch):
        """
        Apply the MaxEnt update to the network given a batch
        """
        assert (
            self.batch_size == 1 or batch["metadata"]["resolution"].std() < 1e-4
        ), "got mutliple resolutions in a batch, which we currently don't support"

        grads = []
        speed_loss = []

        # first generate all the costmaps
        res = self.network.forward(batch["map_features"], return_mean_entropy=True)
        costmaps = res["costmap"]
        speedmaps = res["speedmap"]

        # initialize metadata for cost function
        map_params = batch["metadata"]
        # initialize goals for cost function
        expert_traj = batch["traj"]

        # initialize initial state for MPPI
        initial_states = expert_traj[:, 0]
        x0 = {
            "state": initial_states.clone(),
            "steer_angle": batch["steer"][:, [0]]
            if "steer" in batch.keys()
            else torch.zeros(initial_states.shape[0], device=initial_states.device),
        }
        x = self.mppi.model.get_observations(x0)

        # also get KBM states for expert
        X_expert = {
            "state": expert_traj.clone(),
            "steer_angle": batch["steer"].unsqueeze(-1)
            if "steer" in batch.keys()
            else torch.zeros(self.mppi.B, self.mppi.T, 1, device=initial_states.device),
        }
        expert_kbm_traj = self.mppi.model.get_observations(X_expert)

        # initialize solver
        initial_state = expert_traj[:, 0]
        x0 = {
            "state": initial_state,
            "steer_angle": batch["steer"][:, [0]]
            if "steer" in batch.keys()
            else torch.zeros(1, device=initial_state.device),
        }
        x = self.mppi.model.get_observations(x0)

        goals = []
        for bi in range(expert_traj.shape[0]):
            map_params_b = {
                "resolution": batch["metadata"]["resolution"][bi],
                "length": batch["metadata"]["length"][bi],
                "origin": batch["metadata"]["origin"][bi],
            }
            etraj = expert_kbm_traj[bi]
            goals.append(self.clip_to_map_bounds(etraj[:, :2], map_params_b).view(1, 2))

        self.mppi.reset()
        self.mppi.cost_fn.data["waypoints"] = goals
        self.mppi.cost_fn.data["local_navmap"] = {"data": costmaps, "metadata": map_params, "feature_keys":["cost"]}

        for ii in range(self.mppi_itrs):
            with torch.no_grad():
                self.mppi.get_control(x, step=False)

        # weighting version
        trajs = self.mppi.noisy_states.clone()
        weights = self.mppi.last_weights.clone()

        # afaik, this op is not batch-able because of torch.bincount
        # so just loop it - this is not the speed bottleneck
        learner_state_visitations = []
        expert_state_visitations = []
        for bi in range(trajs.shape[0]):
            map_params_b = {
                "resolution": batch["metadata"]["resolution"][bi],
                "length": batch["metadata"]["length"][bi],
                "origin": batch["metadata"]["origin"][bi],
            }

            footprint_learner_traj = apply_footprint(trajs[bi], self.footprint).view(
                self.mppi.K, -1, 2
            )
            footprint_expert_traj = apply_footprint(
                expert_kbm_traj[bi].unsqueeze(0), self.footprint
            ).view(1, -1, 2)

            lsv = get_state_visitations(
                footprint_learner_traj, map_params_b, weights[bi]
            )

            esv = get_state_visitations(footprint_expert_traj, map_params_b)
            learner_state_visitations.append(lsv)
            expert_state_visitations.append(esv)

            """
            fig, axs = plt.subplots(1, 3)
            axs[0].plot(trajs[bi][weights[bi].argmax(), :, 0].cpu(), trajs[bi][weights[bi].argmax(), :, 1].cpu(), c='r', marker='.')
            axs[0].imshow(lsv.T.cpu(), origin='lower', extent=(
                map_params_b['origin'][0].item(),
                map_params_b['origin'][0].item() + map_params_b['length'][0].item(),
                map_params_b['origin'][1].item(),
                map_params_b['origin'][1].item() + map_params_b['length'][1].item(),
            ))
            axs[0].set_title('learner')

            axs[1].plot(expert_kbm_traj[bi][:, 0].cpu(), expert_kbm_traj[bi][:, 1].cpu(), c='r', marker='.')
            axs[1].plot(batch['traj'][bi, :, 0].cpu(), batch['traj'][bi, :, 1].cpu(), c='b', marker='.')
            axs[1].imshow(esv.T.cpu(), origin='lower', extent=(
                map_params_b['origin'][0].item(),
                map_params_b['origin'][0].item() + map_params_b['length'][0].item(),
                map_params_b['origin'][1].item(),
                map_params_b['origin'][1].item() + map_params_b['length'][1].item(),
            ))
            axs[1].set_title('expert')

            axs[2].imshow((esv - lsv).T.cpu(), origin='lower', extent=(
                map_params_b['origin'][0].item(),
                map_params_b['origin'][0].item() + map_params_b['length'][0].item(),
                map_params_b['origin'][1].item(),
                map_params_b['origin'][1].item() + map_params_b['length'][1].item(),
            ))
            plt.show()
            """

        learner_state_visitations = torch.stack(learner_state_visitations, dim=0)
        expert_state_visitations = torch.stack(expert_state_visitations, dim=0)

        grads = (expert_state_visitations - learner_state_visitations) / trajs.shape[0]
        grads = grads.unsqueeze(1) #grad shape needs to match costmap shape

        if not torch.isfinite(grads).all():
            import pdb; pdb.set_trace()

        # Speedmaps here:
        expert_speedmaps = []
        for bi in range(expert_traj.shape[0]):
            map_params_b = {
                "resolution": batch["metadata"]["resolution"][bi],
                "length": batch["metadata"]["length"][bi],
                "origin": batch["metadata"]["origin"][bi],
            }
            # no footprint
            #            epos = expert_traj[bi][:, :2].unsqueeze(0)
            #            espeeds = torch.linalg.norm(expert_traj[bi][:, 7:10], dim=-1).unsqueeze(0)
            #            esm = get_speedmap(epos, espeeds, map_params_b).view(costmaps[bi].shape)

            # footprint
            epos = apply_footprint(
                expert_kbm_traj[bi].unsqueeze(0), self.footprint
            ).view(1, -1, 2)
            espeeds = torch.linalg.norm(expert_traj[bi][:, 7:10], dim=-1).unsqueeze(0)
            espeeds = espeeds.unsqueeze(2).tile(1, 1, len(self.footprint)).view(1, -1)
            esm = get_speedmap(epos, espeeds, map_params_b).view(costmaps[bi].shape)

            expert_speedmaps.append(esm)

            #debug viz
            """
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(expert_kbm_traj[bi][:, 0].cpu(), expert_kbm_traj[bi][:, 1].cpu(), c='r')
            axs[0].imshow(esm.T.cpu(), origin='lower', extent=(
                map_params_b['origin'][0].item(),
                map_params_b['origin'][0].item() + map_params_b['length'][0].item(),
                map_params_b['origin'][1].item(),
                map_params_b['origin'][1].item() + map_params_b['length'][1].item(),
            ))
            axs[0].set_title('expert speedmap')
            plt.show()
            """

        expert_speedmaps = torch.stack(expert_speedmaps, dim=0)

        speedmap_probs = res["speed_logits"].softmax(axis=1)

        # bin expert speeds
        _sbins = self.network.speed_bins[:-1].to(self.device).view(1, -1, 1, 1)
        sdiffs = expert_speedmaps - _sbins
        sdiffs[sdiffs < 0] = 1e10
        expert_speed_idxs = sdiffs.argmin(dim=1) + 1
        expert_speed_idxs = expert_speed_idxs.clip(0, self.network.speed_nbins-1).long()

        mask = (
            expert_speedmaps[:, 0] > 1e-6
        )  # only want the cells that the expert drove in
        ce = torch.nn.functional.cross_entropy(
            speedmap_probs, expert_speed_idxs, reduction="none"
        )[mask]

        # try regularizing speeds to zero
        neg_labels = torch.zeros_like(expert_speed_idxs)
        ce_neg = torch.nn.functional.cross_entropy(
            speedmap_probs, neg_labels, reduction="none"
        )[~mask]

        neg_ratio = mask.sum() / (~mask | mask).sum()

        speed_loss = self.speed_coeff * (ce.mean() + 0.1 * neg_ratio * ce_neg.mean())

        print('IRL GRAD:   {:.4f}'.format(torch.linalg.norm(grads).detach().cpu().item()))
        print('SPEED LOSS: {:.4f}'.format(speed_loss.detach().item()))

        # add regularization
        reg = self.reg_coeff * costmaps

        # kinda jank, but since we're multi-headed and have a loss and a gradient,
        # I think we need two backward passes through the computation graph.
        self.network_opt.zero_grad()
        costmaps.backward(gradient=(grads + reg), retain_graph=True)
        speed_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.network_opt.step()

    def visualize(self, idx=-1):
        """
        Create a visualization of MaxEnt IRL inputs/outputs for the idx-th datapoint.
        """
        if idx == -1:
            idx = np.random.randint(len(self.expert_dataset))

        with torch.no_grad():
            data = self.expert_dataset[idx]

            # hack back to single dim
            map_features = torch.stack([data["map_features"]] * self.mppi.B, dim=0)
            metadata = data["metadata"]
            xmin = metadata["origin"][0].cpu()
            ymin = metadata["origin"][1].cpu()
            xmax = xmin + metadata["length"][0].cpu()
            ymax = ymin + metadata["length"][1].cpu()
            expert_traj = data["traj"]

            res = self.network.forward(map_features, return_mean_entropy=True)

            costmap = res["costmap"]
            speedmap = res["speedmap"]
            costmap_unc = res["costmap_entropy"]
            speedmap_unc = res["speedmap_entropy"]

            #mess with cost quantiles
            costmap_cdf = torch.cumsum(res['cost_logits'].softmax(dim=1), dim=1)
            costmap_quantile = compute_speedmap_quantile(costmap_cdf[0], self.network.cost_bins, q=0.8).exp()

            costmap_quantile = (costmap_quantile.unsqueeze(0).tile(self.mppi.B, 1, 1)).view(*costmap.shape)

            # initialize solver
            initial_state = expert_traj[0]
            x0 = {
                "state": initial_state,
                "steer_angle": data["steer"][[0]]
                if "steer" in data.keys()
                else torch.zeros(1, device=initial_state.device),
            }
            x = torch.stack([self.mppi.model.get_observations(x0)] * self.mppi.B, dim=0)

            X_expert = {
                "state": expert_traj,
                "steer_angle": data["steer"].unsqueeze(-1)
                if "steer" in data.keys()
                else torch.zeros(self.mppi.B, self.mppi.T, 1, device=initial_states.device),
            }
            expert_kbm_traj = self.mppi.model.get_observations(X_expert)

            map_params = {
                "origin": torch.stack([metadata["origin"]] * self.mppi.B, dim=0),
                "length": torch.stack([metadata["length"]] * self.mppi.B, dim=0),
                "resolution": torch.stack([metadata["resolution"]] * self.mppi.B, dim=0),
            }

            goals = [
                self.clip_to_map_bounds(expert_kbm_traj[:, :2], metadata).view(1, 2)
            ] * self.mppi.B

            self.mppi.reset()
            self.mppi.cost_fn.data["waypoints"] = goals
            self.mppi.cost_fn.data["local_navmap"] = {
                # "data": costmap_quantile,
                "data": costmap,
                "metadata": map_params,
                "feature_keys": ["cost"]
            }

            # solve for traj
            for ii in range(self.mppi_itrs):
                self.mppi.get_control(x, step=False)

            tidx = self.mppi.last_cost.argmin()
            traj = self.mppi.last_states[tidx]

            footprint_learner_traj = apply_footprint(traj, self.footprint).view(1, -1, 2)
            footprint_expert_traj = apply_footprint(
                expert_kbm_traj.unsqueeze(0), self.footprint
            ).view(1, -1, 2)

            lsv = get_state_visitations(footprint_learner_traj, metadata)
            esv = get_state_visitations(footprint_expert_traj, metadata)

            learner_cost = (lsv * costmap).sum()
            expert_cost = (esv * costmap).sum()

            metadata = data["metadata"]
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            axs = axs.flatten()

            fk = None
            fklist = ["num_voxels", "max_elevation", "step", "diff", "dino_0"]
            for f in fklist:
                if f in self.expert_dataset.feature_keys:
                    fk = f
                    idx = self.expert_dataset.feature_keys.index(fk)
                    break

            img = data["image"].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu()

            fig.suptitle("Expert cost = {:.4f}, Learner cost = {:.4f}".format(expert_cost.item(), learner_cost.item()))

            axs[0].imshow(img)
            axs[1].imshow(
                map_features[0][idx].T.cpu(),
                origin="lower",
                cmap="gray",
                extent=(xmin, xmax, ymin, ymax),
            )
            m1 = axs[2].imshow(
                costmap[0, 0].T.cpu(),
                origin="lower",
                cmap="jet",
                extent=(xmin, xmax, ymin, ymax),
            )

            m2 = axs[3].imshow(
                speedmap[0, 0].T.cpu(),
                origin="lower",
                cmap="jet",
                extent=(xmin, xmax, ymin, ymax),
                vmin=0.0,
                vmax=10.0,
            )
            m3 = axs[4].imshow(
                costmap_unc[0, 0].T.cpu(),
                origin="lower",
                cmap="viridis",
                extent=(xmin, xmax, ymin, ymax),
            )

            # m4 = axs[5].imshow(
            #     speedmap_unc[0, 0].T.cpu(),
            #     origin="lower",
            #     cmap="viridis",
            #     extent=(xmin, xmax, ymin, ymax),
            # )

            m4 = axs[5].imshow(
                costmap_quantile[0, 0].T.cpu(),
                origin="lower",
                cmap="jet",
                vmax=costmap.max(),
                extent=(xmin, xmax, ymin, ymax),
            )

            #dont plot the initial state bc learner traj doesnt contain initial
            axs[1].plot(
                expert_kbm_traj[1:, 0].cpu(), expert_kbm_traj[1:, 1].cpu(), c="y", label="expert"
            )
            axs[2].plot(expert_kbm_traj[1:, 0].cpu(), expert_kbm_traj[1:, 1].cpu(), c="y")
            axs[3].plot(expert_kbm_traj[1:, 0].cpu(), expert_kbm_traj[1:, 1].cpu(), c="y")
            axs[4].plot(expert_kbm_traj[1:, 0].cpu(), expert_kbm_traj[1:, 1].cpu(), c="y")
            axs[5].plot(expert_kbm_traj[1:, 0].cpu(), expert_kbm_traj[1:, 1].cpu(), c="y")

            #            axs[0].plot(traj_px[:, 0], traj_px[:, 1], c='g')
            axs[1].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c="g", label="learner")
            axs[2].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c="g")
            axs[3].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c="g")
            axs[4].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c="g")
            axs[5].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c="g")

            for ax in axs[1:]:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

            axs[0].set_title("FPV")
            axs[1].set_title("gridmap {}".format(fk))
            #            axs[2].set_title('irl cost (clipped)')
            axs[2].set_title("irl cost mean")
            axs[3].set_title("speedmap mean")
            axs[4].set_title("irl cost unc")
            # axs[5].set_title("speedmap unc")
            axs[5].set_title("cost quantile")

            for i in [1, 2, 4, 5]:
                axs[i].set_xlabel("X(m)")
                axs[i].set_ylabel("Y(m)")

            axs[1].legend()

            plt.colorbar(m1, ax=axs[2])
            plt.colorbar(m2, ax=axs[3])
            plt.colorbar(m3, ax=axs[4])
            plt.colorbar(m4, ax=axs[5])
        return fig, axs

    def clip_to_map_bounds(self, traj, metadata):
        """
        Given traj, find last point (temporally) in the map bounds
        """
        ox = metadata["origin"][0]
        oy = metadata["origin"][1]
        lx = metadata["length"][0]
        ly = metadata["length"][1]
        xs = traj[:, 0]
        ys = traj[:, 1]

        in_bounds = (xs > ox) & (xs < ox + lx) & (ys > oy) & (ys < oy + ly)
        idx = (torch.arange(in_bounds.shape[0], device=self.device) * in_bounds).argmax(
            dim=0
        )

        return traj[idx]

    def to(self, device):
        self.device = device
        self.expert_dataset = self.expert_dataset.to(device)
        self.mppi = self.mppi.to(device)
        self.network = self.network.to(device)
        self.footprint = self.footprint.to(device)
        return self
