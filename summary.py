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


def plot_confusion_matrix(cm, model, n_runs):
    """Confusion matrix summed over the runs."""
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
    fig.savefig(FIGURE / f"{model}_confusion_matrix_{n_runs}runs.png", dpi=200, bbox_inches="tight")


def plot_band(ax, epochs, values, label, log=False):
    if log:
        values = np.log10(values)
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    low, high = mean - std, mean + std
    if log:
        mean, low, high = 10 ** mean, 10 ** low, 10 ** high
    else:
        low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)    # accuracy stays in [0, 1]
    line, = ax.plot(epochs, mean, label=label)
    ax.fill_between(epochs, low, high, color=line.get_color(), alpha=0.2)


def plot_curves(histories, model):
    """Training and test curves, mean ± std over the runs."""
    epochs = [h["epoch"] for h in histories[0]]

    def get(key):                                   # array (runs, epochs) for one metric
        return np.array([[h[key] for h in history] for history in histories])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    plot_band(ax1, epochs, get("loss"), "Training", log=True)
    plot_band(ax1, epochs, get("test_loss"), "Test", log=True)
    plot_band(ax2, epochs, get("accuracy_exact"), "Training")
    plot_band(ax2, epochs, get("test_accuracy"), "Test")

    ax1.set_yscale("log")
    ax1.set_ylabel("Loss (scala logaritmica)")
    ax2.set_ylabel("Accuracy")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoca")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{NAMES[model]} — media su {len(histories)} run (banda: ± dev. std.)")
    fig.savefig(FIGURE / f"{model}_curves_{len(histories)}runs.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate the runs of one model")
    parser.add_argument("--model", default="cnn5", choices=list(NAMES.keys()))
    model = parser.parse_args().model

    acc, f1, cms, histories = [], [], [], []
    for path in sorted(EXPERIMENTS.glob(f"{model}_seed*/results.json")):
        with open(path) as f:
            run = json.load(f)
        acc.append(run["test"]["accuracy_exact"])
        f1.append(run["test"]["macro_f1"])
        cms.append(np.array(run["test"]["confusion_matrix"]))
        histories.append(run["history"])
        print(f"{path.parent.name}: accuracy {acc[-1]:.4f} | macro-F1 {f1[-1]:.4f}")

    acc, f1 = np.array(acc), np.array(f1)
    print(f"\n{model} over {len(acc)} runs:")
    print(f"  accuracy: {100 * acc.mean():.2f} ± {100 * acc.std(ddof=1):.2f} %")
    print(f"  macro-F1: {100 * f1.mean():.2f} ± {100 * f1.std(ddof=1):.2f} %")

    cm_total = sum(cms)
    print(f"\nConfusion matrix summed over {len(cms)} runs (rows = true, columns = predicted):")
    print(cm_total)
    plot_confusion_matrix(cm_total, model, len(cms))

    if all("test_loss" in history[0] for history in histories):
        plot_curves(histories, model)
    else:
        print("\nNo per-epoch test curves in these runs: rerun train.py to get them.")
    plt.show()