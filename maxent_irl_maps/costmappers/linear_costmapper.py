import torch


class LinearCostMapper:
    """
    Generate a costmap as a linear combination of weights
    """

    def __init__(self, weights):
        """
        Args:
            weights: the weights to take a linear combo of features of.
        """
        self.weights = weights

    def get_costmap(self, map_features):
        """
        Return a costmap according to the current weights
        Args:
            map_features: the features to use as input. Expect batch of [B1 x ... x Bn x C x W x H]
        """
        map_data = map_features["map_features"]
        n_leading_dims = len(map_data.shape[:-3])
        expand_shape = [1] * n_leading_dims + [len(self.weights)] + [1, 1]
        costmap_data = (map_data * self.weights.view(*expand_shape)).sum(dim=-3)
        return costmap_data
