"""
Compare misclassified test files with correctly classified files of the same activity
and with files of the activity they are confused with.
Each figure is a 2 x 3 grid of heatmaps:
    - first row:  two files of the true activity + one file of the confused activity
    - second row: two misclassified files + another file of the confused activity
    
Output: 05_FIGURE/<figure>_errori_vs_corretti.png
Usage (from the repository root): python results_analysis/compare_files.py --figure <walk|fall>
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.io import loadmat

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
TEST = BASE / "03_DATASET" / "NTU-Fi_HAR" / "test_amp"
EXPERIMENTS = BASE / "04_ESPERIMENTI"
FIGURE = BASE / "05_FIGURE"
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]
NAMES = {"cnn5": "CNN-5", "gru": "GRU", "lstm": "LSTM"}

FIGURES = {
    "walk": {"model": "cnn5",           # CNN-5: all its 25 errors are walk -> run
             "files": ["walk/walk170.mat", "walk/walk175.mat", "run/run170.mat",
                       "walk/walk188.mat", "walk/walk199.mat", "run/run188.mat"]},
    "fall": {"model": "gru",            # GRU: 11 errors fall -> clean
             "files": ["fall/fall170.mat", "fall/fall175.mat", "clean/clean170.mat",
                       "fall/fall190.mat", "fall/fall197.mat", "clean/clean190.mat"]},
}

def errors_per_file(model):
    """File -> list of the wrong predictions over the baseline runs of a model; also the number of runs."""
    paths = sorted(EXPERIMENTS.glob(f"{model}_seed*/results.json"))
    wrong = {}
    for path in paths:
        with open(path) as f:
            res = json.load(f)["test"]
        for file, true, pred in zip(res["files"], res["y_true"], res["y_pred"]):
            if true != pred:
                wrong.setdefault(file, []).append(ACTIVITIES[pred])
    return wrong, len(paths)

def title(file, wrong, n_runs):
    """Writes the title of a heatmap panel: "<file> - mai sbagliato" if the file was always correctly classified, 
    otherwise "<file> - sbagliato <n>/<N> -> <most frequent wrong prediction>"."""
    name = file.split("/")[1].removesuffix(".mat")
    if file not in wrong:                           # always correctly classified
        return f"{name} — mai sbagliato"
    preds = wrong[file]
    most = max(set(preds), key=preds.count)          # most frequent wrong prediction
    return f"{name} — sbagliato {len(preds)}/{n_runs} → {most}"


# ----------------- MAIN -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heatmaps of misclassified vs correctly classified test files")
    parser.add_argument("--figure", default="walk", choices=list(FIGURES.keys()))
    name = parser.parse_args().figure
    model, files = FIGURES[name]["model"], FIGURES[name]["files"]

    wrong, n_runs = errors_per_file(model)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)

    for ax, file in zip(axes.flat, files):
        csi = loadmat(TEST / file)["CSIamp"].reshape(3, 114, 2000)    # raw amplitude, not preprocessed
        # same vmin/vmax for all panels so that the panels are comparable
        im = ax.imshow(csi[0], aspect="auto", cmap="viridis", vmin=0, vmax=55)
        ax.set_title(title(file, wrong, n_runs), fontsize=11)

    for ax in axes[1]:
        ax.set_xlabel("Pacchetto (tempo)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Sottoportante")

    fig.colorbar(im, ax=axes, label="Ampiezza CSI grezza")
    fig.suptitle(f"Errori della {NAMES[model]} su {n_runs} run: file corretti e file sbagliati (coppia di antenne 1)")
    fig.savefig(FIGURE / f"{name}_errori_vs_corretti.png", dpi=200, bbox_inches="tight")
    plt.show()
