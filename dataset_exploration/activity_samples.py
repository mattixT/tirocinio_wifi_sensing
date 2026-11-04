"""
Figure: one sample for each of the 6 activities of NTU-Fi HAR (training set),
raw amplitude of antenna pair 1, same colour scale for all the panels.
It shows that the activities leave different "signatures" in the CSI

Output: the figure campioni_per_attivita_<n>.pdf
Usage (from root directory): python dataset_exploration/activity_samples.py [--sample <file number, default 0>]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.io import loadmat

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thesis_figures"))
import thesis_font  # same font (Latin Modern) and text sizes in all the figures of the thesis

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
TRAIN = BASE / "03_DATASET" / "NTU-Fi_HAR" / "train_amp"
FIGURE = BASE / "05_FIGURE"
ACTIVITIES = {"box": "box\n(pugni)", "circle": "circle\n(rotazione delle braccia)",
              "clean": "clean\n(pulire il pavimento)", "fall": "fall\n(caduta)",
              "run": "run\n(corsa)", "walk": "walk\n(camminata)"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One CSI sample for every activity")
    parser.add_argument("--sample", type=int, default=0, help="number of the file (e.g. 4 -> box4.mat)")
    n = parser.parse_args().sample

    fig, axes = plt.subplots(2, 3, figsize=thesis_font.size(1.0, 7.5), sharex=True, sharey=True, layout="constrained")
    for ax, (act, label) in zip(axes.flat, ACTIVITIES.items()):
        csi = loadmat(TRAIN / act / f"{act}{n}.mat")["CSIamp"].reshape(3, 114, 2000)
        # same vmin/vmax for all panels: equal colours = equal amplitudes
        im = ax.imshow(csi[0], aspect="auto", cmap="viridis", vmin=0, vmax=55)
        ax.set_title(label)

    fig.supxlabel("Pacchetto (500 pacchetti = 1 s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Sottoportante")

    fig.colorbar(im, ax=axes, label="Ampiezza CSI")
    fig.suptitle(f"Un campione per attività: file numero {n} dell'insieme di addestramento, coppia di antenne 1")
    fig.savefig(FIGURE / f"campioni_per_attivita_{n}.pdf", dpi=200, bbox_inches="tight")
    plt.show()
