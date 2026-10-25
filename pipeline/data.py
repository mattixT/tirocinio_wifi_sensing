"""
Loading of NTU-Fi HAR and PyTorch Dataset/DataLoader for training, validation and test.
The whole dataset is loaded in memory once (1200 samples, preprocessed by load_sample). Files are
read in a fixed order (activities as in ACTIVITIES, files sorted by name), so every activity
always gets the same label: 0=box, 1=circle, 2=clean, 3=fall, 4=run, 5=walk.

get_loaders(val=..., norm=...) returns (train_loader, val_loader, test_loader), batches of 64 as in SenseFi:
    - val=None      no validation set: 936 train / 264 test, val_loader is None
    - val="random"  stratified random 20% of the training set: 748 train / 188 val
    - val="block"   files numbered 140-159 of every activity: 816 train / 120 val
    - norm="sensefi" (default), "sample" or "subcarrier": normalization of the samples. With
      "subcarrier" every one of the 342 rows (antenna pair x subcarrier) is normalized with its own mean
      and std, computed on the training set only and then applied unchanged to validation and test.

This script prints the size of the three sets for the three options.
Usage (from the repository root): python pipeline/data.py
"""

from pathlib import Path

import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from preprocessing import load_sample, NORMS

DATASET = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/03_DATASET/NTU-Fi_HAR")
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]
VAL_FRACTION = 0.2
SPLIT_SEED = 0
VAL_BLOCK = range(140, 160) # block validation

def load_split(split, norm="sensefi"):
    """Load and preprocess all the samples of a split ('train_amp' or 'test_amp').
    norm is passed to load_sample ("sensefi", "sample" or "none").
    Returns X: (N, 3, 114, 500) float32 and y: (N,) int64 (label = index in ACTIVITIES)."""
    X, y = [], []
    for label, act in enumerate(ACTIVITIES):
        for f in sorted((DATASET / split / act).glob("*.mat")):
            X.append(load_sample(f, norm))
            y.append(label)
    return np.stack(X), np.array(y, dtype=np.int64)

def list_files(split):
    """Names of the files of a split ('activity/file.mat'), in the same order used by load_split."""
    return [f"{act}/{f.name}"
            for act in ACTIVITIES
            for f in sorted((DATASET / split / act).glob("*.mat"))]

def split_validation(y, fraction=VAL_FRACTION, seed=SPLIT_SEED):
    """Stratified random split of the training set: a fraction of every activity goes to validation.
    The seed is fixed, so every run uses the same files. Returns sorted (train_idx, val_idx)."""
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(idx, test_size=fraction, stratify=y, random_state=seed)
    return np.sort(train_idx), np.sort(val_idx)

def split_validation_block(files, block=VAL_BLOCK):
    """Block split of the training set: the files numbered in `block` (default 140-159) of every
    activity go to validation, all the others stay in training.
    files: names 'activity/file.mat' in the same order as the samples. Returns (train_idx, val_idx)."""
    numbers = [int(re.search(r"\d+", f.split("/")[1]).group()) for f in files]   # 'box/box142.mat' -> 142
    val_idx = np.array([i for i, n in enumerate(numbers) if n in block])
    train_idx = np.array([i for i, n in enumerate(numbers) if n not in block])
    return train_idx, val_idx

def normalize_subcarrier(X_train, *others):
    """Per-subcarrier z-score, in place: every row (antenna pair, subcarrier) gets its own mean and std,
    computed on the training set only (all its samples and time steps). The same numbers are then
    applied to the other arrays (validation, test), so no information comes from them.
    Arrays: (N, 3, 114, T) float32. Returns the mean and std, both of shape (1, 3, 114, 1)."""
    mean = X_train.mean(axis=(0, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    std = X_train.std(axis=(0, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    for X in (X_train, *others):
        X -= mean
        X /= std
    return mean, std

class CSIDataset(Dataset):
    """PyTorch Dataset on arrays already in memory: sample i is the pair (X[i], y[i])."""
    def __init__(self, X, y, files=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.files = files

    def __len__(self):
        """Number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, i):
        """Sample i as the pair (X[i], y[i]); the DataLoader collects batch_size of them into a batch."""
        return self.X[i], self.y[i]

def get_loaders(batch_size=64, val=None, norm="sensefi"):
    """DataLoaders as in SenseFi (batch 64, shuffle only on training) with an optional validation set.
    val: None, "random" or "block"; norm: "sensefi", "sample" or "subcarrier" (see the module docstring).
    Returns (train_loader, val_loader, test_loader); val_loader is None without validation."""
    if norm not in NORMS:
        raise ValueError(f"unknown normalization: {norm}")
    # per-subcarrier: load without normalization, normalize after the split with training statistics
    load_norm = "none" if norm == "subcarrier" else norm
    X_train, y_train = load_split("train_amp", load_norm)
    X_test, y_test = load_split("test_amp", load_norm)
    train_files = list_files("train_amp")

    X_val = y_val = val_files = None
    if val is not None:
        if val == "random":
            train_idx, val_idx = split_validation(y_train)
        elif val == "block":
            train_idx, val_idx = split_validation_block(train_files)
        else:
            raise ValueError(f"unknown validation mode: {val}")
        val_files = [train_files[i] for i in val_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_train, y_train = X_train[train_idx], y_train[train_idx]

    if norm == "subcarrier":
        normalize_subcarrier(X_train, X_test, *([] if X_val is None else [X_val]))

    val_loader = None
    if X_val is not None:
        val_loader = DataLoader(CSIDataset(X_val, y_val, val_files), batch_size=batch_size, shuffle=False)
    
    # shuffle=True only for the training set, so the model does not see the classes in blocks
    train_loader = DataLoader(CSIDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CSIDataset(X_test, y_test, list_files("test_amp")), 
                             batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ----------------- TEST -----------------

if __name__ == "__main__":
    # the three normalizations: mean and std of training and test after the preprocessing
    for norm in NORMS:
        train_loader, _, test_loader = get_loaders(norm=norm)
        Xtr, Xte = train_loader.dataset.X, test_loader.dataset.X
        print(f"norm={norm}: train mean {Xtr.mean():.3f} std {Xtr.std():.3f}"
              f" | test mean {Xte.mean():.3f} std {Xte.std():.3f}")

    for val in [None, "random", "block"]:
        train_loader, val_loader, test_loader = get_loaders(val=val)
        n_val = len(val_loader.dataset) if val_loader else 0
        print(f"val={val}: train {len(train_loader.dataset)} - val {n_val} - test {len(test_loader.dataset)}")
        if val_loader:
            print(" validation samples per class:", np.bincount(val_loader.dataset.y.numpy()).tolist())
            print(" first validation samples:", val_loader.dataset.files[:5])