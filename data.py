"""Caricamento del dataset NTU-Fi HAR (solo ampiezza CSI).

Ogni file .mat contiene la variabile 'CSIamp' di forma (342, 2000):
342 = 3 antenne x 114 sottoportanti, 2000 = pacchetti (istanti temporali).
Come in SenseFi, si sottocampiona nel tempo (default ogni 4 pacchetti -> 500)
e si riorganizza il campione in (3, 114, T).
"""
import re
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

CLASSES = ["box", "circle", "clean", "fall", "run", "walk"]
N_ANT, N_SUB, N_PKT = 3, 114, 2000

# Costanti di normalizzazione globali usate da SenseFi per NTU-Fi
SENSEFI_MEAN, SENSEFI_STD = 42.3199, 4.9802


def load_split(root, split, step=4, cache=True):
    """Restituisce X (N, 3, 114, 2000//step) float32 e y (N,) int64.

    step: fattore di sottocampionamento temporale (4 -> T=500 come SenseFi).
    Con cache=True salva un .npz accanto al dataset per velocizzare i run successivi.
    """
    root = Path(root)
    cache_file = root / "cache" / f"{split}_step{step}.npz"
    if cache and cache_file.exists():
        d = np.load(cache_file)
        return d["X"], d["y"], list(d["files"])

    X, y, files = [], [], []
    for label, cls in enumerate(CLASSES):
        for f in sorted((root / split / cls).glob("*.mat"),
                        key=lambda p: int(re.search(r"\d+", p.stem).group())):
            amp = loadmat(f)["CSIamp"][:, ::step].astype(np.float32)
            X.append(amp.reshape(N_ANT, N_SUB, -1))
            y.append(label)
            files.append(f"{cls}/{f.name}")
    X, y = np.stack(X), np.array(y, dtype=np.int64)
    if cache:
        cache_file.parent.mkdir(exist_ok=True)
        np.savez(cache_file, X=X, y=y, files=np.array(files))
    return X, y, files


class Normalizer:
    """Tre strategie, per l'analisi dell'effetto della normalizzazione:
    - 'sensefi': costanti globali di SenseFi
    - 'zscore' : media/deviazione standard globali stimate SOLO sul training set
    - 'subcarrier': z-score per (antenna, sottoportante), stimato sul training set
    - 'none'   : nessuna normalizzazione
    """
    def __init__(self, mode="sensefi"):
        self.mode = mode
        self.mean, self.std = 0.0, 1.0

    def fit(self, X):
        if self.mode == "sensefi":
            self.mean, self.std = SENSEFI_MEAN, SENSEFI_STD
        elif self.mode == "zscore":
            self.mean, self.std = float(X.mean()), float(X.std())
        elif self.mode == "subcarrier":
            self.mean = X.mean(axis=(0, 3), keepdims=True)[0]
            self.std = X.std(axis=(0, 3), keepdims=True)[0] + 1e-6
        elif self.mode != "none":
            raise ValueError(self.mode)
        return self

    def __call__(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


class CSIDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def get_datasets(root, step=4, norm="sensefi", val_ratio=0.2, seed=0):
    """Train/val/test. La validazione e' ricavata dal training set con split
    stratificato: il test set ufficiale non viene mai usato per scegliere
    iperparametri o epoca migliore."""
    Xtr, ytr, _ = load_split(root, "train_amp", step)
    Xte, yte, _ = load_split(root, "test_amp", step)
    idx_tr, idx_va = train_test_split(np.arange(len(ytr)), test_size=val_ratio,
                                      stratify=ytr, random_state=seed)
    normalizer = Normalizer(norm).fit(Xtr[idx_tr])
    return (CSIDataset(normalizer(Xtr[idx_tr]), ytr[idx_tr]),
            CSIDataset(normalizer(Xtr[idx_va]), ytr[idx_va]),
            CSIDataset(normalizer(Xte), yte))
