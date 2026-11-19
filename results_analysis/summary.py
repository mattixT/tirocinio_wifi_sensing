"""
Summary of the runs (5 seeds) of one model, for the baseline or for a validation variant and print:
    - test accuracy and macro-F1 of every run, mean ± sample standard deviation
    - confusion matrix summed over the runs (5 x 264 = 1320 predictions)
    - learning curves (training, test and, if present, validation): mean over the runs with a band
      of ± 1 standard deviation
With --val random or --val block it reads the runs of that variant (folders <model>_val-<mode>_seed<N>)
and also reports the test accuracy of the model at the epoch with the lowest validation loss.
With --norm sample or --norm subcarrier it reads the runs of the normalization analysis
(folders <model>_norm-<mode>_seed<N>).

Output: <model>[_val-<mode>][_norm-<mode>]_<N>runs_confusion_matrix.pdf and ..._<N>runs_curves.pdf
Usage (from the repository root):
    python results_analysis/summary.py --model <cnn5|gru|lstm> [--val <none|random|block>] [--norm <sensefi|sample|subcarrier>]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
EXPERIMENTS = BASE / "04_ESPERIMENTI"
FIGURE = BASE / "05_FIGURE"
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]
NAMES = {"cnn5": "CNN-5", "gru": "GRU", "lstm": "LSTM"}     # names used in the figures
COLORS = {"Addestramento": "C0", "Test": "C1", "Validazione": "C2"}   # same colour for the same set in every figure

def plot_confusion_matrix(cm, name):
    """Draw and save the confusion matrix summed over the runs."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), ACTIVITIES, rotation=45)
    ax.set_yticks(range(6), ACTIVITIES)
    ax.set_xlabel("Classe predetta")
    ax.set_ylabel("Classe reale")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.savefig(FIGURE / f"{name}_confusion_matrix.pdf", dpi=200, bbox_inches="tight")


def plot_band(ax, epochs, values, label, log=False):
    """Mean over the runs ± standard deviation. values is a 2D array (runs, epochs)."""
    if log:
        values = np.log10(values)
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    low, high = mean - std, mean + std
    if log:
        mean, low, high = 10 ** mean, 10 ** low, 10 ** high
    else:
        low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)    # accuracy stays in [0, 1]
    line, = ax.plot(epochs, mean, label=label, color=COLORS[label])
    ax.fill_between(epochs, low, high, color=line.get_color(), alpha=0.2)


def plot_curves(histories, model, name):
    """Loss (log scale) and accuracy curves of training, validation (if present) and test, mean ± std over the runs"""
    epochs = [h["epoch"] for h in histories[0]]

    def get(key):                                   # array (runs, epochs) for one metric
        return np.array([[h[key] for h in history] for history in histories])

    has_val = "val_loss" in histories[0][0]         # check if there is a validation curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # plot constituation: training, validation (if present), test
    plot_band(ax1, epochs, get("loss"), "Addestramento", log=True)
    if has_val:
        plot_band(ax1, epochs, get("val_loss"), "Validazione", log=True)
    plot_band(ax1, epochs, get("test_loss"), "Test", log=True)
    plot_band(ax2, epochs, get("accuracy_exact"), "Addestramento")
    if has_val:
        plot_band(ax2, epochs, get("val_accuracy"), "Validazione")
    plot_band(ax2, epochs, get("test_accuracy"), "Test")

    ax1.set_yscale("log")
    ax1.set_ylabel("Loss (scala logaritmica)")
    ax2.set_ylabel("Accuratezza")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoca")
        ax.grid(alpha=0.3)
        ax.legend()
    variant = " con validation set" if has_val else ""
    fig.savefig(FIGURE / f"{name}_curves.pdf", dpi=200, bbox_inches="tight")


def mean_std(values):
    """'mean ± std' in percent, sample standard deviation (ddof=1)."""
    values = 100 * np.array(values)
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f} %"


# ----------------- MAIN -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate the runs of one model")
    parser.add_argument("--model", default="cnn5", choices=list(NAMES.keys()))
    parser.add_argument("--val", default="none", choices=["none", "random", "block"])
    parser.add_argument("--norm", default="sensefi", choices=["sensefi", "sample", "subcarrier"])
    args = parser.parse_args()
    suffix = "" if args.val == "none" else f"_val-{args.val}"
    if args.norm != "sensefi":
        suffix += f"_norm-{args.norm}"                # e.g. gru_norm-sample
    name = f"{args.model}{suffix}_{{}}runs"         # e.g. gru_val-random_5runs

    acc, f1, cms, histories, best = [], [], [], [], []
    for path in sorted(EXPERIMENTS.glob(f"{args.model}{suffix}_seed*/results.json")):
        with open(path) as f:
            run = json.load(f)
        acc.append(run["test"]["accuracy_exact"])
        f1.append(run["test"]["macro_f1"])
        cms.append(np.array(run["test"]["confusion_matrix"]))
        histories.append(run["history"])
        line = f"{path.parent.name}: accuracy {acc[-1]:.4f} | macro-F1 {f1[-1]:.4f}"
        if run.get("test_best_val"):                 # only in the validation variant
            best.append(run["test_best_val"])
            line += (f" | best val epoch {best[-1]['epoch']:2d} -> test accuracy "
                     f"{best[-1]['accuracy_exact']:.4f}")
        print(line)

    name = name.format(len(acc))
    print(f"\n{args.model}{suffix} over {len(acc)} runs (model after the last epoch):")
    print(f"  accuracy: {mean_std(acc)}")                           
    print(f"  macro-F1: {mean_std(f1)}")

    if best:
        print("\nModel of the epoch with the lowest validation loss:")
        print(f"  test accuracy: {mean_std([b['accuracy_exact'] for b in best])}")
        print(f"  test macro-F1: {mean_std([b['macro_f1'] for b in best])}")
        print(f"  chosen epochs: {[b['epoch'] for b in best]}")
        print(f"  validation accuracy at epoch 30: {mean_std([h[-1]['val_accuracy'] for h in histories])}")

    cm_total = sum(cms)
    print(f"\nConfusion matrix summed over {len(cms)} runs (rows = true, columns = predicted):")
    print(cm_total)
    plot_confusion_matrix(cm_total, name)

    if all("test_loss" in history[0] for history in histories):
        plot_curves(histories, args.model, name)
    else:
        print("\nNo per-epoch test curves in these runs: run train.py to get them.")
    plt.show()