"""
Loading of the NTU-Fi HAR dataset, and PyTorch Dataset/DataLoader definitions for training/testing.
Optional validation set: with a stratified 20% of the training set is used for validation.
"""

from pathlib import Path

import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from preprocessing import load_sample

DATASET = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/03_DATASET/NTU-Fi_HAR")
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]
VAL_FRACTION = 0.2
SPLIT_SEED = 0
VAL_BLOCK = range(140, 160) # block validation

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

def split_validation(y, fraction=VAL_FRACTION, seed=SPLIT_SEED):
    """Stratified random split of the training set"""
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split( idx, test_size=fraction, stratify=y, random_state=seed)
    return np.sort(train_idx), np.sort(val_idx)

def split_validation_block(files, block=VAL_BLOCK):
    """"""
    numbers = [int(re.search(r"\d+", f.split("/")[1]).group()) for f in files]   # 'box/box142.mat' -> 142
    val_idx = np.array([i for i, n in enumerate(numbers) if n in block])
    train_idx = np.array([i for i, n in enumerate(numbers) if n not in block])
    return train_idx, val_idx

class CSIDataset(Dataset):
    """PyTorch Dataset on arrays already in memory: sample i is the pair (X[i], y[i]).
    files (optional): name of the file of every sample, same order as X"""
    def __init__(self, X, y, files=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.files = files

    def __len__(self):
        """Number of samples in the dataset"""
        return len(self.y)

    def __getitem__(self, i):
        """Sample i: the DataLoader asks for 64 samples and stacks them into a batch"""
        return self.X[i], self.y[i]

def get_loaders(batch_size=64, val=None):
    """Train and test DataLoaders with SenseFi setup"""
    X_train, y_train = load_split("train_amp")
    X_test, y_test = load_split("test_amp")
    train_files = list_files("train_amp")

    val_loader = None
    if val is not None:
        if val == "random":
            train_idx, val_idx = split_validation(y_train)
        elif val == "block":
            train_idx, val_idx = split_validation_block(train_files)
        else:
            raise ValueError(f"unknown validation mode: {val}")
        val_files = [train_files[i] for i in val_idx]
        val_loader = DataLoader(CSIDataset(X_train[val_idx], y_train[val_idx], val_files),
                                batch_size=batch_size, shuffle=False)
        X_train, y_train = X_train[train_idx], y_train[train_idx]
    
    # shuffle=True only for the training set, so the model does not see the classes in blocks
    train_loader = DataLoader(CSIDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CSIDataset(X_test, y_test, list_files("test_amp")), 
                             batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ----------------- TEST -----------------

if __name__ == "__main__":
    for val in [None, "random", "block"]:
        train_loader, val_loader, test_loader = get_loaders(val=val)
        n_val = len(val_loader.dataset) if val_loader else 0
        print(f"val={val}: train {len(train_loader.dataset)} - val {n_val} - test {len(test_loader.dataset)}")
        if val_loader:
            print(" validation samples per class:", np.bincount(val_loader.dataset.y.numpy()).tolist())
            print(" first validation samples:", val_loader.dataset.files[:5])