"""
Figure: for every run, the exact test accuracy (correct samples / 264) and the accuracy computed by
the SenseFi code (mean of the accuracies of the test batches). The differences of more than
3 points are written next to the run.

Output: images/esatta_vs_batch.pdf
Usage (from the repository root): python thesis_figures/esatta_vs_batch.py
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import thesis_font as tf
from results_data import SEEDS, test_accuracy

LABEL_FROM = 3.0                                     # write the difference when it is larger than this


def main():
    fig, axes = plt.subplots(1, 3, figsize=tf.size(1.0, 6.0), sharey=True, layout="constrained")
    for ax, (model, name) in zip(axes, tf.MODELS.items()):
        exact = test_accuracy(model)
        batch = test_accuracy(model, key="accuracy_sensefi")
        color = tf.COLORS[model]
        for s, e, b in zip(SEEDS, exact, batch):
            ax.plot([s, s], [e, b], color="#d0d0d0", lw=2, zorder=1, solid_capstyle="butt")
            if abs(b - e) > LABEL_FROM:
                ax.text(s + 0.16, b, tf.comma(b - e, 1, sign=True), va="center", fontsize=7, color="0.33")
        ax.scatter(SEEDS, exact, s=9, color=color, zorder=3)
        ax.scatter(SEEDS, batch, s=30, facecolor="white", edgecolor=color, lw=1.1, zorder=2)
        ax.set_title(name, loc="left")
        ax.set_xticks(list(SEEDS))
        ax.set_xlim(-0.5, 4.6)
        ax.set_xlabel("seed")
        ax.grid(axis="y", color="#d9d9d9", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(84, 100.6)
    axes[0].set_yticks(range(84, 101, 2))
    axes[0].set_ylabel(f"accuratezza di test ({tf.percent()})")

    handles = [Line2D([], [], marker="o", ls="", color="0.3", markersize=3),
               Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="0.3", markersize=5.5)]
    fig.legend(handles, ["accuratezza esatta", "media sui batch (codice SenseFi)"], ncol=2,
               loc="outside upper center")
    tf.save(fig, "esatta_vs_batch")


if __name__ == "__main__":
    main()
