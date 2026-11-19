"""
Figure: test accuracy against the number of parameters of each model.

Output: images/accuratezza_complessita.pdf
Usage (from the repository root): python work_figures/accuratezza_complessita.py
"""
import matplotlib.pyplot as plt

import thesis_font as tf
from results_data import test_accuracy

PARAMETERS = {"cnn5": 476_582, "gru": 78_726, "lstm": 104_838}
LABELS = {"cnn5": (14, 0, "left"), "gru": (0, 0.85, "center"), "lstm": (12, -0.45, "left")}

def main():
    fig, ax = plt.subplots(figsize=tf.size(0.65, 5.5))
    for model, name in tf.MODELS.items():
        acc = test_accuracy(model)
        mean, std = acc.mean(), acc.std(ddof=1)
        x = PARAMETERS[model] / 1000
        ax.errorbar(x, mean, yerr=std, fmt="o", color=tf.COLORS[model], markersize=4, capsize=3, elinewidth=1)
        dx, dy, ha = LABELS[model]
        ax.text(x + dx, mean + dy, f"{name}: {tf.comma(mean)}{tf.percent()}", ha=ha, va="center", fontsize=7)
    ax.set_xlim(0, 550)
    ax.set_ylim(95.5, 100)
    ax.set_xlabel("parametri (migliaia)")
    ax.set_ylabel(f"accuratezza di test ({tf.percent()})")
    ax.grid(color="#e3e3e3", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    tf.save(fig, "accuratezza_complessita")

if __name__ == "__main__":
    main()
