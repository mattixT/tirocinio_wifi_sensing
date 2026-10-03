"""
Loading of the NTU-Fi HAR dataset, and PyTorch Dataset/DataLoader definitions for training/testing.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing import load_sample

DATASET = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/03_DATASET/NTU-Fi_HAR")
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]

def load_split(split):
    """Load all the samples of a split ('train_amp' or 'test_amp') and returns X: (N, 3, 114, 500) float32, y: (N,) int64"""
    X, y = [], []
    for label, act in enumerate(ACTIVITIES):
        for f in sorted((DATASET / split / act).glob("*.mat")):
            X.append(load_sample(f))
            y.append(label)
    return np.stack(X), np.array(y, dtype=np.int64)

def list_files(split):
    """Names of the files of a split ('activity/file.mat'), in the same order used by load_split"""
    return [f"{act}/{f.name}"
            for act in ACTIVITIES
            for f in sorted((DATASET / split / act).glob("*.mat"))]

class CSIDataset(Dataset):
    """PyTorch Dataset on arrays already in memory: sample i is the pair (X[i], y[i])"""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    
    def __len__(self):
        """Number of samples in the dataset"""
        return len(self.y)
    
    def __getitem__(self, i):
        """Sample i: the DataLoader asks for 64 samples  and stack them into a batch"""
        return self.X[i], self.y[i]

def get_loaders(batch_size = 64):
    """Train and test DataLoaders with SenseFi setup"""
    X_train, y_train = load_split("train_amp")
    X_test, y_test = load_split("test_amp")

    # shuffle=True only for the training set, so the model does not see the classes in blocks
    train_loader = DataLoader(CSIDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CSIDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ----------------- TEST -----------------

if __name__ == "__main__":
    train_loader, test_loader = get_loaders()

    print("Samples: train", len(train_loader.dataset), "- test", len(test_loader.dataset))
    print("Batches per epoch: train", len(train_loader), "- test", len(test_loader))

    xb, yb = next(iter(train_loader))
    print("One batch -> X:", xb.shape, "| y:", yb.shape, yb.dtype)
    print("Labels in the first batch:", yb.tolist())