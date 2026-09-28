"""
Preprocessing of a single CSI sample of the NTU-Fi HAR dataset.
The function load_sample() applies the same preprocessing used by the official SenseFi implementation:
    - packet downsampling: 1 packet every 4
    - z-score normalization: (x - mean) / std
    - reshape to (3 x 114 x 2000)
    - conversion to float32 (PyTorch)
"""

from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

MEAN = 42.3199
STD = 4.9802
STEP = 4

def load_sample(path):
    csi = loadmat(path)["CSIamp"]
    csi = csi[:, ::STEP]
    csi = (csi - MEAN) / STD
    csi = csi.reshape(3, 114, -1)
    return csi.astype(np.float32)

# -------------
# TEST
# -------------
if __name__ == "__main__":
    BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
    FILE = BASE / "03_DATASET" / "NTU-Fi_HAR" / "train_amp" / "walk" / "walk0.mat"
    FIGURE = BASE / "05_FIGURE"

    raw = loadmat(FILE)["CSIamp"]
    x = load_sample(FILE)

    # print of the shape, mean and std of the original and preprocessed sample
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