"""
Figure: six training samples of the same activity (files 0, 30, 60, 90, 120, 150), raw amplitude of
antenna pair 1, same colour scale for all the panels. It shows how much the CSI of one activity
changes from one repetition to another. The activity is fixed by the constant ACTIVITY.

Output: the image confronto_{ACTIVITY}.png

Usage (from the repository root): python exploration/compare_samples.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat

# building of the paths to the dataset and figure folders
BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
DATASET = BASE / "03_DATASET" / "NTU-Fi_HAR" / "train_amp"
FIGURE = BASE / "05_FIGURE"

ACTIVITY = "box"                      # name of the activity to compare
SAMPLES = [0, 30, 60, 90, 120, 150]   # which files of the dataset to compare

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)

# loop over the files and plot the heatmaps of the raw CSI amplitude
for ax, n in zip(axes.flat, SAMPLES):
    csi = loadmat(DATASET / ACTIVITY / f"{ACTIVITY}{n}.mat")["CSIamp"].reshape(3, 114, 2000)
    im = ax.imshow(csi[0], aspect="auto", cmap="viridis", vmin=0, vmax=55)
    ax.set_title(f"{ACTIVITY}{n}.mat")

for ax in axes[1]:
    ax.set_xlabel("Pacchetto (tempo)")
for ax in axes[:, 0]:
    ax.set_ylabel("Sottoportante")

# save the figure
fig.colorbar(im, ax=axes, label="Ampiezza CSI")
fig.suptitle(f"Attività: {ACTIVITY} — 6 campioni diversi")
fig.savefig(FIGURE / f"confronto_{ACTIVITY}.png", dpi=200, bbox_inches="tight")
plt.show()