"""
    This script compares different samples of the same activity 
    by plotting the CSI amplitude of each sample in a subplot
"""

from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat

# building of the paths to the dataset and figure folders
BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
DATASET = BASE / "03_DATASET" / "NTU-Fi_HAR" / "train_amp"
FIGURE = BASE / "05_FIGURE"

ATTIVITA = "box"                      # name of the activity to compare
CAMPIONI = [0, 30, 60, 90, 120, 150]   # which files of the dataset to compare

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)

for ax, n in zip(axes.flat, CAMPIONI):
    csi = loadmat(DATASET / ATTIVITA / f"{ATTIVITA}{n}.mat")["CSIamp"].reshape(3, 114, 2000)
    im = ax.imshow(csi[0], aspect="auto", cmap="viridis", vmin=0, vmax=55)
    ax.set_title(f"{ATTIVITA}{n}.mat")

for ax in axes[1]:
    ax.set_xlabel("Pacchetto (tempo)")
for ax in axes[:, 0]:
    ax.set_ylabel("Sottoportante")

# save the figure
fig.colorbar(im, ax=axes, label="Ampiezza CSI")
fig.suptitle(f"Attività: {ATTIVITA} — 6 campioni diversi")
fig.savefig(FIGURE / f"confronto_{ATTIVITA}.png", dpi=200, bbox_inches="tight")
plt.show()