"""
Figure: test accuracy of the 5 runs of each model with the three normalizations. The grey lines join
the runs with the same seed (same initial weights and same order of the batches), so that every run
can be compared with the corresponding run of the baseline.

Output: images/normalizzazione_seed.pdf
Usage (from the repository root): python thesis_figures/normalizzazione_seed.py
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import thesis_font as tf
from results_data import NORMS, test_accuracy

TICKS = ["SenseFi", "per\ncampione", "per sotto-\nportante"]


def main():
    fig, axes = plt.subplots(1, 3, figsize=tf.size(1.0, 6.2), sharey=True, layout="constrained")
    for ax, (model, name) in zip(axes, tf.MODELS.items()):
        values = [test_accuracy(model, norm) for norm in NORMS]
        for left, right, x in zip(values[:-1], values[1:], range(len(NORMS) - 1)):
            for a, b in zip(left, right):                           # same seed
                ax.plot([x - 0.12, x + 0.88], [a, b], color="#d6d6d6", lw=0.6, zorder=1)
        for x, acc in enumerate(values):
            ax.scatter([x - 0.12] * len(acc), acc, s=8, color=tf.COLORS[model], zorder=3)
            ax.errorbar(x + 0.08, acc.mean(), yerr=acc.std(ddof=1), fmt="o", color="black",
                        markersize=3, capsize=2.5, elinewidth=0.9, zorder=4)
            ax.text(x + 0.15, acc.mean(), tf.comma(acc.mean()), va="center", fontsize=6.5, color="0.33")
        ax.set_title(name, loc="left")
        ax.set_xticks(range(len(NORMS)), TICKS)
        ax.set_xlim(-0.45, 2.6)
        ax.grid(axis="y", color="#e3e3e3", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(87, 100.8)
    axes[0].set_yticks(range(88, 101, 2))
    axes[0].set_ylabel(f"accuratezza di test ({tf.percent()})")
    handles = [Line2D([], [], marker="o", ls="", color="0.35", markersize=3),
               Line2D([], [], marker="o", color="black", markersize=3, lw=0.9),
               Line2D([], [], color="#d6d6d6", lw=0.8)]
    fig.legend(handles, ["singola esecuzione", "media ± dev. std.", "stesso seed"], ncol=3,
               loc="outside upper center")
    tf.save(fig, "normalizzazione_seed")


if __name__ == "__main__":
    main()
