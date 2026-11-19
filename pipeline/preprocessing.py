"""
Preprocessing of a single CSI sample of NTU-Fi HAR.
load_sample(path, norm) reads one .mat file (shape 342 x 2000) and returns an array (3, 114, 500) of float32:
    1. downsampling: 1 packet every 4 (2000 -> 500 packets)
    2. normalization, chosen with norm:
         "sensefi" (default)  z-score with the global constants of SenseFi: (x - 42.3199) / 4.9802
         "sample"             z-score with the mean and std of the sample itself
         "none"               no normalization (used for the per-subcarrier normalization, which needs
                              the statistics of the training set and is applied later in data.py)
    3. reshape (342, 500) -> (3, 114, 500): 3 antenna pairs x 114 subcarriers x 500 time steps
    4. conversion to float32 (from float64 of loadmat)

This script prints shape, mean and std of one training sample before and after the
preprocessing and saves the comparison figure preprocessing_test.pdf
Usage (from the repository root): python pipeline/preprocessing.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Constants of the official SenseFi code and verified by the phase of exploration of the dataset
MEAN = 42.3199  # global mean
STD = 4.9802    # global std
STEP = 4        # 2000 -> 500 packets (downsampling factor)

NORMS = ["sensefi", "sample", "subcarrier"]   # normalizations compared in the additional analysis

def load_sample(path, norm="sensefi"):
    """Load one .mat file and return the preprocessed sample: array (3, 114, 500) of float32.
    norm: "sensefi", "sample" or "none" (see the module docstring)."""
    csi = loadmat(path)["CSIamp"]
    csi = csi[:, ::STEP]
    if norm == "sensefi":
        csi = (csi - MEAN) / STD                      # same two numbers for every sample
    elif norm == "sample":
        csi = (csi - csi.mean()) / csi.std()          # mean and std of this sample only
    elif norm != "none":
        raise ValueError(f"unknown normalization: {norm}")
    csi = csi.reshape(3, 114, -1)
    return csi.astype(np.float32)


# ----------------- TEST -----------------

if __name__ == "__main__":
    BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
    FILE = BASE / "03_DATASET" / "NTU-Fi_HAR" / "train_amp" / "walk" / "walk0.mat"
    FIGURE = BASE / "05_FIGURE"

    raw = loadmat(FILE)["CSIamp"]
    x = load_sample(FILE)

    # print of shape, mean and std of the original and preprocessed sample
    print("Before -> shape:", raw.shape, f" mean: {raw.mean():.2f} std: {raw.std():.2f}")
    print("After  -> shape:", x.shape, f" mean: {x.mean():.2f} std: {x.std():.2f}")
    print("Numeric type:", x.dtype)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # plot of the original and preprocessed sample
    im1 = axes[0].imshow(raw[:114], aspect="auto", cmap="viridis")      # original
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(x[0], aspect="auto", cmap="viridis")           # preprocessed
    fig.colorbar(im2, ax=axes[1])

    for ax in axes:
        ax.set_ylabel("Sottoportante")
    axes[0].set_xlabel("Pacchetto")
    axes[1].set_xlabel("Istante")

    # saving for the report
    fig.savefig(FIGURE / "preprocessing_test.pdf", dpi=200, bbox_inches="tight")
    plt.show()