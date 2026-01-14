import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset

#Get a random portion of the unlabel data to label
class RandomSampling():
    def __init__(self, dataset):
        self.dataset = dataset

    def query(self, n):
        unlabeled_idxs = len(self.dataset)
        # selected_idxs = np.random.choice(unlabeled_idxs, n, replace=False)
        return np.random.choice(unlabeled_idxs, n, replace=False)
    