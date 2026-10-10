"""
Preprocessing of a single CSI sample of NTU-Fi HAR, identical to the official SenseFi code.

load_sample(path) reads one .mat file (variable 'CSIamp', shape 342 x 2000) and returns an
array (3, 114, 500) of float32:
    1. downsampling: 1 packet every 4 (2000 -> 500 packets, 500 Hz -> 125 Hz, still 4 s)
    2. z-score normalization with the global constants of SenseFi: (x - 42.3199) / 4.9802
    3. reshape (342, 500) -> (3, 114, 500): 3 antenna pairs x 114 subcarriers x 500 time steps
SenseFi normalizes before downsampling: the order of the two steps does not change the values.

Run directly, it prints shape, mean and std of one training sample before and after the
preprocessing and saves the comparison figure 05_FIGURE/preprocessing_test.png.

Usage (from the repository root): python pipeline/preprocessing.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Constants of the official SenseFi code and verified by the phase of exploration of the dataset
MEAN = 42.3199
STD = 4.9802
STEP = 4

def load_sample(path):
    """Load one .mat file and return the preprocessed sample: array (3, 114, 500) of float32."""
    csi = loadmat(path)["CSIamp"]
    csi = csi[:, ::STEP]
    csi = (csi - MEAN) / STD
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
    axes[0].set_title("Prima: ampiezza originale (2000 pacchetti)")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(x[0], aspect="auto", cmap="viridis")           # preprocessed
    axes[1].set_title("Dopo: sottocampionata e normalizzata (500 istanti)")
    fig.colorbar(im2, ax=axes[1])

    for ax in axes:
        ax.set_ylabel("Sottoportante")
    axes[0].set_xlabel("Pacchetto")
    axes[1].set_xlabel("Istante (dopo ::4)")

    # saving for the report
    fig.savefig(FIGURE / "preprocessing_test.png", dpi=200, bbox_inches="tight")
    plt.show()