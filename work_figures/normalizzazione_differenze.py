"""
Figure: difference of test accuracy from the SenseFi normalization, run by run (same seed), for the
per-sample and the per-subcarrier normalization.

Output: images/normalizzazione_differenze.pdf
Usage (from the repository root): python work_figures/normalizzazione_differenze.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import ttest_rel

import thesis_font as tf
from results_data import mean_difference, test_accuracy

VARIANTS = {"norm-sample": "Normalizzazione per campione", "norm-subcarrier": "Normalizzazione per sottoportante"}
JITTER = np.array([-0.16, -0.08, 0.0, 0.08, 0.16])        

def main():
    fig, axes = plt.subplots(1, 2, figsize=tf.size(1.0, 6.0), sharey=True, layout="constrained")
    for ax, (variant, title) in zip(axes, VARIANTS.items()):
        ax.axhline(0, color="#52514e", lw=0.9, zorder=1)
        for i, model in enumerate(tf.MODELS):
            base, acc = test_accuracy(model), test_accuracy(model, variant)
            diff = acc - base                                   # seed by seed
            ax.scatter(i + JITTER, diff, s=12, color=tf.COLORS[model], edgecolor="white", lw=0.4, zorder=3)
            ax.plot([i - 0.27, i + 0.27], [diff.mean()] * 2, color="black", lw=1.5, zorder=4)
            ax.text(i + 0.31, diff.mean(), tf.comma(mean_difference(acc, base), 2, sign=True), fontsize=7, va="center",
                    fontweight="bold")
            p = ttest_rel(acc, base).pvalue
            ax.text(i, -9.35, f"p = {tf.comma(p, 3 if p < 0.01 else 2)}", fontsize=6.5, color="#52514e", ha="center")
        ax.set_xticks(range(3), tf.MODELS.values())
        ax.set_xlim(-0.5, 2.9)
        ax.set_title(title, loc="left")
        ax.yaxis.grid(True, color="#e3e3e3")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(-9.8, 3.4)
    axes[0].set_yticks(range(-8, 4, 2))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: tf.comma(v, 0, sign=v != 0)))
    axes[0].set_ylabel("differenza rispetto a SenseFi\n(punti percentuali)")
    axes[1].text(2.88, 0.2, "0 = come SenseFi", fontsize=6.5, color="#52514e", va="bottom", ha="right")
    handles = [Line2D([], [], marker="o", ls="", color="#52514e", ms=3.5), Line2D([], [], color="black", lw=1.5)]
    fig.legend(handles, ["singola esecuzione, rispetto alla baseline con lo stesso seed", "differenza media"],
               ncol=2, loc="outside upper center")
    tf.save(fig, "normalizzazione_differenze")

if __name__ == "__main__":
    main()
