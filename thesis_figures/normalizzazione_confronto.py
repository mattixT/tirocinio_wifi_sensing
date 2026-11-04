"""
Figure: test accuracy (mean ± sample standard deviation over 5 runs) of each model with the three
normalizations. On the right: difference from SenseFi and p of the paired t test (same seeds),
in bold when p < 0.05.

Output: images/normalizzazione_confronto.pdf
Usage (from the repository root): python thesis_figures/normalizzazione_confronto.py
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel

import thesis_font as tf
from results_data import NORMS, mean_difference, test_accuracy

STYLE = {"": ("#52514e", "o", 0.25), "norm-sample": ("#a3a199", "s", 0.0),     # colour, marker,
         "norm-subcarrier": ("#6250d6", "D", -0.25)}                          # offset in the row


def p_text(p):
    return tf.comma(p, 3 if p < 0.01 else 2)


def main():
    fig, ax = plt.subplots(figsize=tf.size(1.0, 6.2))
    fig.subplots_adjust(right=0.74)
    models = list(tf.MODELS)
    for r, model in enumerate(models):
        y0 = len(models) - 1 - r                                   # CNN-5 at the top
        if r:
            ax.axhline(y0 + 0.5, color="#e1e0d9", lw=0.6, zorder=0)
        base = test_accuracy(model)
        for norm, (color, marker, offset) in STYLE.items():
            acc = test_accuracy(model, norm)
            y = y0 + offset
            ax.errorbar(acc.mean(), y, xerr=acc.std(ddof=1), fmt="none", ecolor=color, elinewidth=1, capsize=2.5)
            ax.plot(acc.mean(), y, marker=marker, ms=5, color=color, mec="white", mew=0.7, ls="")
            if norm == "":
                label, weight, c = "riferimento", "normal", "#898781"
            else:
                p = ttest_rel(acc, base).pvalue
                label = f"{tf.comma(mean_difference(acc, base), 2, sign=True)}  (p = {p_text(p)})"
                weight, c = ("bold", "black") if p < 0.05 else ("normal", "#52514e")
            ax.text(1.03, y, label, transform=ax.get_yaxis_transform(), fontsize=7, va="center",
                    color=c, fontweight=weight)
    ax.text(1.03, len(models) - 0.45, "differenza rispetto\na SenseFi (punti)", transform=ax.get_yaxis_transform(),
            fontsize=7, color="#52514e", va="bottom")
    ax.set_yticks(range(len(models)), [tf.MODELS[m] for m in models][::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.55, len(models) - 0.45)
    ax.set_xlim(87.5, 101.5)
    ax.set_xticks(range(88, 101, 2))
    ax.xaxis.grid(True, color="#e1e0d9")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel(f"accuratezza di test ({tf.percent()}), media ± deviazione standard su 5 esecuzioni")
    handles = [Line2D([], [], marker=m, ls="", color=c, mec="white", ms=5) for c, m, _ in STYLE.values()]
    labels = [f"{NORMS[''] } (baseline)"] + [NORMS[n] for n in list(STYLE)[1:]]
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.42, 0.9),
               handletextpad=0.3, columnspacing=1.2)
    tf.save(fig, "normalizzazione_confronto")


if __name__ == "__main__":
    main()
