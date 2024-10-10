import torch
import numpy as np
from scipy.ndimage import gaussian_filter


class LethalHeightCostmap(torch.nn.Module):
    """
    Identical to the costmapper node in physics_atv_lidar_mapping
    """

    def __init__(
        self, dataset, lethal_height=0.5, blur_sigma=2.0, sharpness=2.0, clip_low=0.8
    ):
        """
        Args:
            dataset: need to give the dataset to extract the normalization constants
        """
        super(LethalHeightCostmap, self).__init__()

        #        self.cost = 1e8
        self.cost = 30.0

        self.lethal_height = lethal_height
        self.blur_sigma = blur_sigma
        self.sharpness = sharpness
        self.clip_low = clip_low

        self.diff_idx = dataset.feature_keys.index("diff")
        self.diff_mean = dataset.feature_mean[self.diff_idx].item()
        self.diff_std = dataset.feature_std[self.diff_idx].item()

        self.device = "cpu"

    def forward(self, map_features):
        device = map_features.device
        diffs = map_features[..., self.diff_idx, :, :]

        # denormalize back to regular values
        diffs = (diffs * self.diff_std) + self.diff_mean

        # equivalent to the baseline costmap code
        costmap = (
            (diffs > self.lethal_height).cpu().numpy().astype(np.float32)
        )  # find cells above lethal height
        costmap = gaussian_filter(costmap, sigma=self.blur_sigma)  # blur for inflation
        costmap = self.sharpness * costmap  # multiply by a scaling factor to sharpen
        costmap[costmap < self.clip_low] = 0.0  # apply hysteresis
        costmap[costmap > self.clip_low] = self.cost
        costmap = torch.tensor(costmap).to(device).unsqueeze(1)

        return {
            "costmap": costmap,
            "speedmap": torch.distributions.Normal(
                loc=torch.zeros_like(costmap), scale=torch.ones_like(costmap)
            ),
        }

    def to(self, device):
        self.device = device
        return self
