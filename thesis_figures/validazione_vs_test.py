"""
Figure: in the two variants with a validation set, the validation accuracy and the test accuracy of
the model after the last epoch (30), mean over the 5 runs. The dashed line is the test accuracy of
the baseline (without validation set). It shows how much the validation set overestimates the test.

Output: images/validazione_vs_test.pdf
Usage (from the repository root): python thesis_figures/validazione_vs_test.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import thesis_font as tf
from results_data import runs, test_accuracy

VARIANTS = {"val-random": "validazione\ncasuale", "val-block": "validazione\na blocchi"}


def main():
    fig, axes = plt.subplots(1, 3, figsize=tf.size(1.0, 6.0), sharey=True, layout="constrained")
    for ax, (model, name) in zip(axes, tf.MODELS.items()):
        color = tf.COLORS[model]
        for x, variant in enumerate(VARIANTS):
            val = np.mean([100 * run["history"][-1]["val_accuracy"] for run in runs(model, variant)])
            test = test_accuracy(model, variant).mean()
            ax.plot([x, x], [val, test], color="#d0d0d0", lw=2, zorder=1, solid_capstyle="butt")
            ax.scatter(x, val, marker="s", s=18, color=color, zorder=3)
            ax.scatter(x, test, s=12, color=color, zorder=3)
        ax.axhline(test_accuracy(model).mean(), color="0.3", ls="--", lw=0.8, zorder=2)
        ax.set_title(name, loc="left")
        ax.set_xticks(range(len(VARIANTS)), VARIANTS.values())
        ax.set_xlim(-0.5, 1.5)
        ax.grid(axis="y", color="#d9d9d9", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(92.5, 100.5)
    axes[0].set_yticks(range(93, 101))
    axes[0].set_ylabel(f"accuratezza all'epoca 30 ({tf.percent()})")

    handles = [Line2D([], [], marker="s", ls="", color="0.3", markersize=3.5),
               Line2D([], [], marker="o", ls="", color="0.3", markersize=3),
               Line2D([], [], color="0.3", ls="--", lw=0.8)]
    fig.legend(handles, ["accuratezza di validazione", "accuratezza di test",
                         "test della baseline (senza validazione)"], ncol=3, loc="outside upper center")
    tf.save(fig, "validazione_vs_test")


if __name__ == "__main__":
    main()
