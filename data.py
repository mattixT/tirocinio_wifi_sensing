"""

"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import load_sample

DATASET = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/03_DATASET/NTU-Fi_HAR")
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]

class CSIDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def get_loaders(batch_size = 64):
    X_train, y_train = load_split("train_amp")
    X_test, y_test = load_split("test_amp")

    train_loader = DataLoader(CSIDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CSIDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_split(split):
    X, y = [], []
    for label, act in enumerate(ACTIVITIES):
        for f in sorted((DATASET / split / act).glob("*.mat")):
            X.append(load_sample(f))
            y.append(label)
    return np.stack(X), np.array(y, dtype=np.int64)

if __name__ == "__main__":
    train_loader, test_loader = get_loaders()

    print("Samples: train", len(train_loader.dataset), "- test", len(test_loader.dataset))
    print("Batches per epoch: train", len(train_loader), "- test", len(test_loader))

    xb, yb = next(iter(train_loader))
    print("One batch -> X:", xb.shape, "| y:", yb.shape, yb.dtype)
    print("Labels in the first batch:", yb.tolist())