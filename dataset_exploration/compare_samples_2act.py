"""
Figure: six training samples of two activities (one per row), raw amplitude of
antenna pair (1,1), same colour scale for all the panels. It shows how much the CSI of an activity
changes from one repetition to another and how similar the two activities are.
The activities and the samples are fixed by the constants ACTIVITIES and SAMPLES.

Output: 05_FIGURE/confronto_<activity>_<activity>.pdf (e.g. confronto_walk_run.pdf)
Usage (from the repository root): python dataset_exploration/compare_samples_2act.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat

# building of the paths to the dataset and figure folders
BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
DATASET = BASE / "03_DATASET" / "NTU-Fi_HAR" / "train_amp"
FIGURE = BASE / "05_FIGURE"

ACTIVITIES = ["walk", "run"]         # names of the activities to compare
SAMPLES = [0, 60, 120]   # which files of the dataset to compare

fig, axes = plt.subplots(len(ACTIVITIES), len(SAMPLES), figsize=(15, 7), sharex=True, sharey=True)

# loop over the files and plot the heatmaps of the raw CSI amplitude
for row, act in zip(axes, ACTIVITIES):
    for ax, n in zip(row, SAMPLES):
        csi = loadmat(DATASET / act / f"{act}{n}.mat")["CSIamp"].reshape(3, 114, 2000)
        im = ax.imshow(csi[0], aspect="auto", cmap="viridis", vmin=0, vmax=55)   # pair (1,1)
        ax.set_title(f"{act}, campione {n}")

for ax in axes[1]:
    ax.set_xlabel("Pacchetto")
for ax in axes[:, 0]:
    ax.set_ylabel("Sottoportante")

# save the figure
fig.colorbar(im, ax=axes, label="Ampiezza CSI")
name = "confronto_" + "_".join(ACTIVITIES)
fig.savefig(FIGURE / f"{name}.pdf", dpi=200, bbox_inches="tight")
plt.show()