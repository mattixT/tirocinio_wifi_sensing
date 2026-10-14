"""
This script is used to plot the number of test errors for every file number, one panel per model.
For each model, the x axis lists the file numbers of the test set (the same for every activity); 
the bar is the number of errors on the files with that number, summed over the
6 activities and the 5 runs. 

Output: 05_FIGURE/errori_per_numero_file[_val-<mode>].png

Usage (from the repository root): python results_analysis/errors_by_file.py [--val <none|random|block>]
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
EXPERIMENTS = BASE / "04_ESPERIMENTI"
FIGURE = BASE / "05_FIGURE"
MODELS = {"cnn5": "CNN-5", "gru": "GRU", "lstm": "LSTM"}
COLORS = {"cnn5": "#2a78d6", "gru": "#eb6834", "lstm": "#1baf7a"}    # fixed colour per model
NUMBERS = [16, 17, 18, 19] + list(range(160, 200))                         # test file numbers

def file_number(name):
    """Take the number of a test file from its name, e.g. 'walk/walk188.mat' -> 188."""
    return int(re.search(r"\d+", name.split("/")[1]).group())

def errors_by_number(model, suffix):
    """For each file number, count the number of errors (all activities, all runs)"""
    paths = sorted(EXPERIMENTS.glob(f"{model}{suffix}_seed*/results.json"))
    counts = {n: 0 for n in NUMBERS}
    for path in paths:
        with open(path) as f:
            res = json.load(f)["test"]
        for file, true, pred in zip(res["files"], res["y_true"], res["y_pred"]):
            if true != pred:
                counts[file_number(file)] += 1
    return counts, len(paths)

# ----------------- MAIN -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test errors per file number, one panel per model")
    parser.add_argument("--val", default="none", choices=["none", "random", "block"])
    args = parser.parse_args()
    suffix = "" if args.val == "none" else f"_val-{args.val}"

    # one bar slot per file number, with a small gap between 16-19 and 160-199
    positions = [i if n < 160 else i + 1.5 for i, n in enumerate(NUMBERS)]
    pos = dict(zip(NUMBERS, positions))
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(12, 7), sharex=True)

    for ax, (model, label) in zip(axes, MODELS.items()):
        counts, n_runs = errors_by_number(model, suffix)
        total = sum(counts.values())
        in_block = sum(counts[n] for n in range(190, 200))
        ax.axvspan(pos[180] - 0.5, pos[189] + 0.5, color="0.96", zorder=0)    # block 180-189 
        ax.axvspan(pos[190] - 0.5, pos[199] + 0.5, color="0.90", zorder=0)    # block 190-199
        ax.bar(positions, [counts[n] for n in NUMBERS], width=0.7, color=COLORS[model], zorder=2)
        ax.set_title(f"{label}: {total} errori in {n_runs} run, {in_block} nel blocco 190–199 "
                     f"({100 * in_block / max(total, 1):.0f}%)", fontsize=10, loc="left")
        ax.set_ylabel("Errori")
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

    for low, high in [(180, 189), (190, 199)]:
        axes[0].text((pos[low] + pos[high]) / 2, axes[0].get_ylim()[1] * 0.95, f"blocco {low}–{high}",
                     ha="center", va="top", fontsize=9, color="0.35")
    labels = [str(n) if n in (16, 17, 18, 19) or n % 5 == 0 or n == 199 else "" for n in NUMBERS]
    axes[-1].set_xticks(positions, labels)
    axes[-1].set_xlabel("Numero del file di test")
    variant = "" if args.val == "none" else f" (variante con validation {args.val})"
    fig.suptitle(f"Errori sul test per numero di file, somma su attività e run{variant}")
    fig.tight_layout()
    fig.savefig(FIGURE / f"errori_per_numero_file{suffix}.png", dpi=200, bbox_inches="tight")
    plt.show()
