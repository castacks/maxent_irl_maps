import torch
import numpy as np
import matplotlib.pyplot as plt

"""
Inspired by https://arxiv.org/pdf/2312.16016.pdf,
use cosine distance to traversability prototypes as cost
"""

class TraversabilityPrototypes:
    def __init__(self, prototypes, device='cpu'):
        self.device = device
        self.prototypes = torch.tensor(prototypes, device=device)
        self.expert_dataset = None

    def forward(self, x):
        """
        Args:
            x: [B x C x W x H] Tensor of map features
        """
        _x = x.view(-1, *x.shape)
        _pr = self.prototypes.view(self.prototypes.shape[0], 1, self.prototypes.shape[1], 1, 1)

        sim = (_x * _pr).sum(dim=2)
        maxsim = sim.max(dim=0)[0]
        cost = maxsim
        return cost - cost.min()

    def visualize(self, idx=-1, fig=None, axs=None):
        if idx == -1:
            idx = np.random.randint(len(self.expert_dataset))

        with torch.no_grad():
            data = self.expert_dataset[idx]

            #hack back to single dim
            map_features = data['map_features'].unsqueeze(0)
            metadata = data['metadata']
            xmin = metadata['origin'][0].cpu()
            ymin = metadata['origin'][1].cpu()
            xmax = xmin + metadata['length_x'].cpu()
            ymax = ymin + metadata['length_y'].cpu()

            expert_traj = data['traj']

            #compute costmap
            costmap = self.forward(map_features)

            if fig is None or axs is None:
                fig, axs = plt.subplots(2, 3, figsize=(18, 12))
                axs = axs.flatten()

            axs[0].imshow(data['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            axs[1].imshow(map_features[0][0].cpu(), origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))
            m1 = axs[2].imshow(costmap[0].cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax), vmax=50.)

            axs[1].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y', label='expert')
            axs[2].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y')

            axs[0].set_title('FPV')
            axs[1].set_title('gridmap f0')
            axs[2].set_title('irl cost (clipped)')

            for i in [1, 2]:
                axs[i].set_xlabel('X(m)')
                axs[i].set_ylabel('Y(m)')

            axs[1].legend()

            plt.colorbar(m1, ax=axs[2])
        return fig, axs

    def to(self, device):
        self.device = device
        self.prototypes = self.prototypes.to(device)

        if self.expert_dataset:
            self.expert_dataset = self.expert_dataset.to(device)

        return self
