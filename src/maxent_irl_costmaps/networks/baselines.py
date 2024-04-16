import torch

class AlterBaseline:
    """
    Implementation of the cost function from ALTER as a geometric baseline. This cfn is

    10 if diff > 1.
    6 - SVD2 otherwise
    """
    def __init__(self, dataset, diff_thresh=0.1, device='cpu'):
        """
        Args:
            dataset: need to pass dataset to unnormalize features and find diff and SVD2 idxs
        """
        assert 'diff' in dataset.feature_keys, "need 'diff' feature for alter"
        assert 'SVD2' in dataset.feature_keys, "need 'SVD2' feature for alter"

        self.diff_idx = dataset.feature_keys.index('diff')
        self.svd2_idx = dataset.feature_keys.index('SVD2')

        self.diff_mean = dataset.feature_mean[self.diff_idx].item()
        self.diff_std = dataset.feature_std[self.diff_idx].item()

        self.svd2_mean = dataset.feature_mean[self.svd2_idx].item()
        self.svd2_std = dataset.feature_std[self.svd2_idx].item()

        self.diff_thresh = diff_thresh
        self.speed_bins = torch.arange(11)

        self.device = device

    def forward(self, x, return_features=True):
        diffs = (x[:, self.diff_idx] * self.diff_std) + self.diff_mean
        svd2s = (x[:, self.svd2_idx] * self.svd2_std) + self.svd2_mean

        #theres a 2x multiplier in the paper
        svd2s *= 2

        diff_mask = (diffs > self.diff_thresh).float()

        costmap = (diff_mask) * 10. + (1.-diff_mask) * (6.-svd2s)

        #fake a speedmap
        speedmap = torch.stack([torch.ones_like(costmap)] * 10, dim=1)

        return {
            'costmap': costmap.unsqueeze(1).to(self.device),
            'speedmap': speedmap.to(self.device)
        }

    def ensemble_forward(self, x, return_features=True):
        res = self.forward(x, return_features)
        return {k:v.unsqueeze(0) for k,v in res.items()}

    def to(self, device):
        self.device = device
        self.speed_bins = self.speed_bins.to(device)
        return self

class SemanticBaseline:
    """
    Implementation of a semantic-based cost function (TerrainNet-like)

    For now, here's our semantic cost mapping
    0  -> 10 (obstacle)
    1  -> 3 (unknown)
    2  -> 7  (high veg)
    3  -> 0  (sky)
    4  -> 0  (trail)
    5  -> 0  (trail)
    6  -> 10 (obstacle)
    7  -> 3  (unknown)
    8  -> 10 (obstacle)
    9  -> 5  (low veg)
    10 -> 2  (grass)
    11 -> 10 (obstacle)
    """
    def __init__(self, dataset, device='cpu'):
        self.cost_mappings = torch.tensor([
            10.,
            3.,
            7.,
            0.,
            0.,
            0.,
            10.,
            3.,
            10.,
            5.,
            2.,
            10.,
        ])
        self.fidxs = []
        for i in range(12):
            assert 'ganav_{}'.format(i) in dataset.feature_keys, "need 'ganav_{}' in dataset keys for baseline".format(i)
            self.fidxs.append(dataset.feature_keys.index('ganav_{}'.format(i)))

        self.speed_bins = torch.arange(11)
        self.device = device

    def forward(self, x, return_features=True):
        semantics = x[:, self.fidxs]
        semantic_argmax = semantics.argmax(dim=1)

        costmap = self.cost_mappings[semantic_argmax]

        #fake a speedmap
        speedmap = torch.stack([torch.ones_like(costmap)] * 10, dim=1)

        return {
            'costmap': costmap.unsqueeze(1).to(self.device),
            'speedmap': speedmap.to(self.device)
        }

    def ensemble_forward(self, x, return_features=True):
        res = self.forward(x, return_features)
        return {k:v.unsqueeze(0) for k,v in res.items()}

    def to(self, device):
        self.device = device
        self.cost_mappings = self.cost_mappings.to(device)
        self.speed_bins = self.speed_bins.to(device)
        return self

class AlterSemanticBaseline:
    """
    combine alter and semantics into one baseline
    """
    def __init__(self, dataset, diff_thresh=0.1, device='cpu'):
        self.alter = AlterBaseline(dataset, diff_thresh, device)
        self.semantics = SemanticBaseline(dataset, device)
        self.speed_bins = torch.arange(11)
        self.device = device

    def forward(self, x, return_features=True):
        res_alter = self.alter.forward(x, return_features)
        res_semantics = self.semantics.forward(x, return_features)

        return {
            'costmap': torch.maximum(res_alter['costmap'], res_semantics['costmap']),
            'speedmap': res_alter['speedmap']
        }

    def ensemble_forward(self, x, return_features=True):
        res = self.forward(x, return_features)
        return {k:v.unsqueeze(0) for k,v in res.items()}

    def to(self, device):
        self.device = device
        self.alter = self.alter.to(device)
        self.semantics = self.semantics.to(device)
        return self

if __name__ == '__main__':
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from maxent_irl_costmaps.experiment_management.parse_configs import setup_experiment

    root_fp = '/home/physics_atv/workspace/experiments/yamaha_irl_dino/visual_ablations_50cm/2024-04-05-19-45-38_visual_ablations_semantics/itr_5.pt'

    param_fp = os.path.join(os.path.split(root_fp)[0], '_params.yaml')
    res = setup_experiment(param_fp)
    res['algo'].network.load_state_dict(torch.load(root_fp))
    res['algo'].network.eval()
    dataset = res['dataset']

#    alter_baseline = AlterBaseline(dataset).to(res['algo'].device)
#    res['algo'].network = alter_baseline

    semantic_baseline = SemanticBaseline(dataset).to(res['algo'].device)
    res['algo'].network = semantic_baseline

#    alter_semantic_baseline = AlterSemanticBaseline(dataset).to(res['algo'].device)
#    res['algo'].network = alter_semantic_baseline

    for _ in range(100):
        idx = np.random.randint(len(dataset))
        fig, axs = res['algo'].visualize(idx=idx)
        plt.show()
