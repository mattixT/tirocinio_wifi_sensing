"""
Figure: test accuracy of the five runs (seeds 0-4) of each model and the value published by SenseFi.

Output: images/variabilita_seed.pdf
Usage (from the repository root): python work_figures/variabilita_seed.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import thesis_font as tf
from results_data import test_accuracy 

SENSEFI = {"cnn5": 98.70, "gru": 97.66, "lstm": 97.14}

def main():
    fig, ax = plt.subplots(figsize=tf.size(0.72, 6.0))
    offsets = np.array([-0.24, -0.18, -0.12, -0.06, 0.0])       # one point per seed, side by side
    for i, model in enumerate(tf.MODELS):
        acc = test_accuracy(model)
        mean, std = acc.mean(), acc.std(ddof=1)                  # sample standard deviation
        ax.scatter(i + offsets, acc, s=10, color=tf.COLORS[model], zorder=3)
        ax.errorbar(i + 0.14, mean, yerr=std, fmt="o", color="black", markersize=3.5,
                    capsize=3, elinewidth=1, capthick=1, zorder=4)
        ax.text(i + 0.21, mean, tf.comma(mean), va="center", fontsize=7, color="0.33")
        ax.scatter(i + 0.36, SENSEFI[model], marker="D", s=22, facecolor="white",
                   edgecolor="black", linewidth=0.9, zorder=4)

    ax.set_xticks(range(len(tf.MODELS)), tf.MODELS.values())
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(95, 100.3)
    ax.set_yticks(range(95, 101))
    ax.set_ylabel(f"accuratezza di test ({tf.percent()})")
    ax.grid(axis="y", color="#d9d9d9", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [Line2D([], [], marker="o", ls="", color=tf.COLORS["cnn5"], markersize=3.5),
               Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor="black", markersize=4.5),
               ax.errorbar([], [], yerr=[], fmt="o", color="black", markersize=3.5, capsize=3)]
    ax.legend(handles, ["singola esecuzione", "valore SenseFi", "media ± dev. std."], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.2)
    tf.save(fig, "variabilita_seed")

if __name__ == "__main__":
    main()
