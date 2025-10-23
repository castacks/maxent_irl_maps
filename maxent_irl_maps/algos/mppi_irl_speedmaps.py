import numpy as np
import torch
import matplotlib.pyplot as plt

from tartandriver_perception_infra.trainers.base import Trainer

from torch_mpc.cost_functions.cost_terms.utils import apply_footprint

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_maps.utils import get_state_visitations, get_speedmap, clip_to_map_bounds, modified_hausdorff_distance

class MPPIIRLSpeedmaps(Trainer):
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
        dataset,
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
            dataset: The dataset containing expert demonstrations to imitate
            footprint: "smear" state visitations with this
            mppi: The MPPI object to optimize with
        """
        super().__init__(dataset, batch_size, network, opt)
        self.footprint = footprint
        self.mppi = mppi
        self.mppi_itrs = mppi_itrs

        if 'bev_data' in dataset[0].keys():
            print(dataset[0]["bev_data"]["feature_keys"])

        self.reg_coeff = reg_coeff
        self.speed_coeff = speed_coeff
        self.grad_clip = grad_clip

        self.device = device

    def get_loss(self, batch):
        """
        Apply the MaxEnt update to the network given a batch
        """
        assert (
            self.batch_size == 1 or batch["bev_data"]["metadata"].resolution.std() < 1e-4
        ), "got mutliple resolutions in a batch, which we currently don't support"

        grads = []
        speed_loss = []

        metadata = batch["bev_data"]["metadata"]
        map_features = batch["bev_data"]["data"]

        ## get network outputs ##
        # res = self.network.forward(batch, return_mean_entropy=True)

        # if res is None:
        #     return

        # costmap = res["costmap"]
        # speedmap = res["speedmap"]
        # costmap_unc = res["costmap_entropy"]
        # speedmap_unc = res["speedmap_entropy"]

        ## get network outputs (new) ##
        res = self.network.forward(batch)
        costmap = res['cost']['preds']

        ## Run solver ##
        expert_kbm_traj = self.get_expert_state_traj(batch)

        with torch.no_grad():
            learner_trajs, weights, learner_best_traj, cost_results = self.run_solver_on_costmap(costmap, metadata, expert_kbm_traj)

        #take the initial state out of expert traj
        expert_kbm_traj = expert_kbm_traj[:, 1:]

        footprint_learner_traj = apply_footprint(learner_best_traj, self.footprint)
        footprint_expert_traj = apply_footprint(expert_kbm_traj, self.footprint)

        learner_state_visitations = get_state_visitations(footprint_learner_traj, metadata)
        expert_state_visitations = get_state_visitations(footprint_expert_traj, metadata)

        """
        for bi in range(map_features.shape[0]):
            ltraj = learner_best_traj[bi]
            etraj = expert_kbm_traj[bi]
            lsv = learner_state_visitations[bi]
            esv = expert_state_visitations[bi]

            extent = (
                metadata.origin[bi, 0].item(),
                metadata.origin[bi, 0].item() + metadata.length[bi, 0].item(),
                metadata.origin[bi, 1].item(),
                metadata.origin[bi, 1].item() + metadata.length[bi, 1].item(),
            )

            fig, axs = plt.subplots(1, 3)
            axs[0].plot(ltraj[:, 0].cpu(), ltraj[:, 1].cpu(), c='r', marker='.')
            axs[0].imshow(lsv.T.cpu(), origin='lower', extent=extent)
            axs[0].set_title('learner')

            axs[1].plot(etraj[:, 0].cpu(), etraj[:, 1].cpu(), c='r', marker='.')
            axs[1].plot(batch['odometry']['data'][bi, :, 0].cpu(), batch['odometry']['data'][bi, :, 1].cpu(), c='b', marker='.')
            axs[1].imshow(esv.T.cpu(), origin='lower', extent=extent)
            axs[1].set_title('expert')

            axs[2].imshow((esv - lsv).T.cpu(), origin='lower', extent=extent)
            plt.show()
        """

        grads = (expert_state_visitations - learner_state_visitations) / map_features.shape[0]
        grads = grads.unsqueeze(1) #grad shape needs to match costmap shape

        if not torch.isfinite(grads).all():
            import pdb; pdb.set_trace()

        # Speedmaps here:

        ## get expert speed from state if possible, else compute from odom
        if expert_kbm_traj.shape[-1] >= 4:
            espeeds = expert_kbm_traj[:, :, 3]
        else:
            espeeds = torch.linalg.norm(batch["odometry"]["data"][:, 1:, 7:10], dim=-1)

        #tile espeeds to match footprint
        espeeds = espeeds.unsqueeze(2).tile(1, 1, self.footprint.shape[0])
        
        expert_speedmaps = get_speedmap(footprint_expert_traj, espeeds, metadata)

        speedmap_probs = res["speed"]["logits"].softmax(axis=1)

        # bin expert speeds
        _sbins = self.network.heads["speed"].bins[:-1].to(self.device).view(1, -1, 1, 1)
        sdiffs = expert_speedmaps.unsqueeze(1) - _sbins
        sdiffs[sdiffs < 0] = 1e10
        expert_speed_idxs = sdiffs.argmin(dim=1)
        expert_speed_idxs = expert_speed_idxs.clip(0, self.network.heads["speed"].nbins-1).long()

        #debug viz
        """
        for bi in range(map_features.shape[0]):
            etraj = expert_kbm_traj[bi]
            esm = expert_speedmaps[bi]
            ebins = expert_speed_idxs[bi]

            extent = (
                metadata.origin[bi, 0].item(),
                metadata.origin[bi, 0].item() + metadata.length[bi, 0].item(),
                metadata.origin[bi, 1].item(),
                metadata.origin[bi, 1].item() + metadata.length[bi, 1].item(),
            )
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(etraj[:, 0].cpu(), etraj[:, 1].cpu(), c='r')
            axs[0].imshow(esm.T.cpu(), origin='lower', extent=extent)
            axs[0].set_title('expert speedmap')

            axs[1].plot(etraj[:, 0].cpu(), etraj[:, 1].cpu(), c='r')
            axs[1].imshow(ebins.T.cpu(), origin='lower', extent=extent)
            axs[1].set_title('expert speed bins')

            plt.show()
        """
        
        #only want cells that the expert drove in
        mask = expert_speedmaps > 1e-6

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
        reg = self.reg_coeff * costmap
        irl_grad = grads + reg

        #return dict here is a little 
        return {
            'irl_info': { 
                'grad': irl_grad,
                'tensor': costmap
            },
            'speed_loss': speed_loss
        }

    def update_network(self, loss):
        """
        Implement custom loss bc irl is loss and grad
        """
        try:
            self.opt.zero_grad()
            loss['irl_info']['tensor'].backward(gradient=loss['irl_info']['grad'], retain_graph=True)
            loss['speed_loss'].backward()

        except:
            import pdb;pdb.set_trace()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.opt.step()

    def get_parameters(self):
        return {
            'network': self.network.state_dict()
        }

    def get_expert_state_traj(self, dpt):
        """
        Get the expert trajectory (in MPC states) from datapoint
        """
        inp = {
            'state': dpt["odometry"]["data"],
            'steer_angle': dpt["steer_angle"]["data"].unsqueeze(-1)
        }
        return self.mppi.model.get_observations(inp)

    def run_solver_on_costmap(self, costmap, metadata, expert_traj, clip_goals=False, return_cost_results=False):
        """
        Run MPPI on the corrent costmap from the expert initial state to final state
        Args:
            costmap: [BxWxH] Tensor of costs
            metadata: LocalMapperMetadata object corresponding to the costmap
            expert_traj: [BxTxN] Tensor of expert trajectories
        
        Returns:
            trajs: [BxKxTxN] Tensor of all MPPI samples
            weights: [BxK] Tensor of MPPI sampling weights
            best_traj: [BxTxN] Tensor of the lowest-cost MPPI traj

        Also note that this function will modify the state of the MPPI object
        """
        initial_states = expert_traj[:, 0]
        map_params = {
            "origin": metadata.origin,
            "length": metadata.length,
            "resolution": metadata.resolution,
        }

        if clip_goals:
            goals = clip_to_map_bounds(expert_traj[..., :2], metadata).unsqueeze(1)
        else:
            goals = expert_traj[:, [-1], :2]

        self.mppi.reset()
        self.mppi.cost_fn.data["waypoints"] = goals
        self.mppi.cost_fn.data["local_navmap"] = {
            "data": costmap,
            "metadata": map_params,
            "feature_keys": ["cost"]
        }

        cost_results_all = []

        # solve for traj
        for ii in range(self.mppi_itrs):
            _, cost_results = self.mppi.get_control(initial_states, step=False)
            cost_results_all.append(cost_results)

        cost_results_all = {k: torch.stack([x[k]['cost'] for x in cost_results_all], dim=1) for k in cost_results_all[0].keys()}

        best_traj = self.mppi.last_states
        weights = self.mppi.last_weights
        all_trajs = self.mppi.noisy_states

        return all_trajs, weights, best_traj, cost_results_all
    
    def get_expert_cost(self, costmap, metadata, expert_traj):
        initial_states = expert_traj[:, 0]
        map_params = {
            "origin": metadata.origin,
            "length": metadata.length,
            "resolution": metadata.resolution,
        }

        goals = expert_traj[:, [-1], :2]

        self.mppi.reset()
        self.mppi.cost_fn.data["waypoints"] = goals
        self.mppi.cost_fn.data["local_navmap"] = {
            "data": costmap,
            "metadata": map_params,
            "feature_keys": ["cost"]
        }

        ## assume control costs not relevant here
        dummy_controls = self.mppi.noisy_controls[:, [0]].clone()

        _, _, cost_results = self.mppi.cost_fn.cost(expert_traj.unsqueeze(1), dummy_controls)

        cost_results = {k:v['cost'][:, 0] for k,v in cost_results.items()}

        return cost_results

    def visualize(self, idx=-1, return_metrics=True):
        """
        Create a visualization of MaxEnt IRL inputs/outputs for the idx-th datapoint.
        Also hijacking this script to return a few metrics, namely:
            1. MHD bet. best learner trajectory and expert
            2. Expert log-prob

        Expert log-prob computation:
            - Under maxent IRL, p(tau) \propto exp(-J(tau))
            - p(tau) = (1/Z) exp(-J(tau)), where
            - Z = sum_{all tau} [exp(-J(tau))]
            - We will say that Z ~ sum{all_mppi} [exp(-J(tau))]

            Thus:
            log(p(tau_E)) =
            log(exp(-J(tau_E))) - log(Z) ~= 
            -J(tau_E) - logsumexp(-J(tau_mppi))
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
            
        if idx == -1:
            idx = np.random.randint(len(self.dataset))

        with torch.no_grad():
            dpt = self.dataset.getitem_batch([idx])

            metadata = dpt["bev_data"]["metadata"]
            map_features = dpt["bev_data"]["data"]

            # ## get network outputs ##
            # res = self.network.forward(dpt, return_mean_entropy=True)

            # costmap = res["costmap"].tile(self.mppi.B, 1, 1, 1)
            # speedmap = res["speedmap"].tile(self.mppi.B, 1, 1, 1)
            # costmap_unc = res["costmap_entropy"].tile(self.mppi.B, 1, 1, 1)
            # speedmap_unc = res["speedmap_entropy"].tile(self.mppi.B, 1, 1, 1)

            ## get network outputs (new) ##
            res = self.network.forward(dpt)
            costmap = res['cost']['preds'].tile(self.mppi.B, 1, 1, 1)
            speedmap = res['speed']['preds'].tile(self.mppi.B, 1, 1, 1)
            costmap_unc = res['cost']['entropy'].tile(self.mppi.B, 1, 1, 1)
            speedmap_unc= res['speed']['entropy'].tile(self.mppi.B, 1, 1, 1)

            ## Run solver ##
            expert_kbm_traj = self.get_expert_state_traj(dpt).tile(self.mppi.B, 1, 1)

            learner_trajs, weights, learner_best_traj, learner_cost_results = self.run_solver_on_costmap(costmap, metadata, expert_kbm_traj)

            #take the initial state out of expert traj
            expert_kbm_traj = expert_kbm_traj[:, 1:]
            expert_cost_results = self.get_expert_cost(costmap, metadata, expert_kbm_traj)

            learner_best_cost_results = self.get_expert_cost(costmap, metadata, learner_best_traj)

            ## compute expert log prob
            learner_rewards = -learner_cost_results['costmap_projection'].reshape(self.mppi.B, -1)
            expert_rewards = -expert_cost_results['costmap_projection']
            all_rewards = torch.cat([learner_rewards, expert_rewards.unsqueeze(-1)], dim=-1)
        
            partition_fn = torch.logsumexp(all_rewards, dim=-1)

            expert_log_prob = (expert_rewards - partition_fn).mean()

            learner_rewards = -learner_cost_results['FINAL'].reshape(self.mppi.B, -1)
            expert_rewards = -expert_cost_results['FINAL']

            ## fair game to throw the expert traj into the partition fn
            all_rewards = torch.cat([learner_rewards, expert_rewards.unsqueeze(-1)], dim=-1)

            partition_fn = torch.logsumexp(all_rewards, dim=-1)

            expert_log_prob_goal = (expert_rewards - partition_fn).mean()

            ## compute costmap costs
            expert_costmap_cost = expert_cost_results['costmap_projection'].mean()
            best_learner_costmap_cost = learner_best_cost_results['costmap_projection'].mean()

            ## compute MHD
            mhd = torch.stack([modified_hausdorff_distance(et, lt) for et, lt in zip(expert_kbm_traj, learner_best_traj)]).mean()

            metrics = {
                'expert_log_prob': expert_log_prob.item(),
                'expert_log_goal': expert_log_prob_goal.item(),
                'expert_costmap_cost': expert_costmap_cost.item(),
                'learner_costmap_cost': best_learner_costmap_cost.item(),
                'mhd': mhd.item(),
                'idx': idx
            }

            ## viz ##
            extent = (
                metadata.origin[0, 0].item(),
                metadata.origin[0, 0].item() + metadata.length[0, 0].item(),
                metadata.origin[0, 1].item(),
                metadata.origin[0, 1].item() + metadata.length[0, 1].item(),
            )

            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            axs = axs.flatten()

            fk = None
            fklist = ["num_voxels", "max_elevation", "step", "diff", "dino_0"]
            for f in fklist:
                if f in dpt["bev_data"]["feature_keys"].label:
                    fk = f
                    fidx = dpt["bev_data"]["feature_keys"].index(fk)
                    break

            img = dpt["image"]["data"][0].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu()

            fig.suptitle("dpt {}: MHD={:.4f} Log prob={:.4f} Log prob goal={:.4f} Expert costmap cost={:.4f} Learner costmap cost={:.4f}".format(
                idx,
                mhd.item(),
                expert_log_prob.item(),
                expert_log_prob_goal.item(),
                expert_costmap_cost.item(),
                best_learner_costmap_cost.item()
            ))

            axs[0].imshow(img)
            axs[1].imshow(
                map_features[0][fidx].T.cpu(),
                origin="lower",
                cmap="gray",
                extent=extent,
            )
            m1 = axs[2].imshow(
                costmap[0, 0].T.cpu(),
                origin="lower",
                cmap="jet",
                extent=extent,
            )

            m2 = axs[3].imshow(
                speedmap[0, 0].T.cpu(),
                origin="lower",
                cmap="jet",
                extent=extent,
                vmin=0.0,
                vmax=10.0,
            )
            m3 = axs[4].imshow(
                costmap_unc[0, 0].T.cpu(),
                origin="lower",
                cmap="viridis",
                extent=extent,
            )

            #dont plot the initial state bc learner traj doesnt contain initial
            for i, ax_i in enumerate([1,2,3,4,5]):
                axs[ax_i].plot(expert_kbm_traj[0, :, 0].cpu(), expert_kbm_traj[0, :, 1].cpu(), c="y", label="expert" if i == 0 else None)
                axs[ax_i].plot(learner_best_traj[0, :, 0].cpu(), learner_best_traj[0, :, 1].cpu(), c="g", label="learner" if i == 0 else None)

            for ax in axs[1:]:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])

            axs[0].set_title("FPV")
            axs[1].set_title("gridmap {}".format(fk))
            axs[2].set_title("irl cost mean")
            axs[3].set_title("speedmap mean")
            axs[4].set_title("irl cost unc")
            axs[5].set_title("cost quantile")

            for i in [1, 2, 4, 5]:
                axs[i].set_xlabel("X(m)")
                axs[i].set_ylabel("Y(m)")

            axs[1].legend()

            plt.colorbar(m1, ax=axs[2])
            plt.colorbar(m2, ax=axs[3])
            plt.colorbar(m3, ax=axs[4])

        return {
            'viz': (fig, axs),
            'metrics': metrics
        }

    def to(self, device):
        self.device = device
        self.dataset = self.dataset.to(device)
        self.mppi = self.mppi.to(device)
        self.network = self.network.to(device)
        self.footprint = self.footprint.to(device)
        return self
