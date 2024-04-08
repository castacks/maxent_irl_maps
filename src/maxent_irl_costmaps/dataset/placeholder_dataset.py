import torch

from torch.utils.data import Dataset

class PlaceHolderDataset:
    """
    Temporary dataset object that holds all the metadata for a dataset
    This is necessary because we need to keep track of feature keys and normalizations
    """
    def __init__(self, dataset):
        """
        Args:
            The dataset to copy to make a placeholder
        """
