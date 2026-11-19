"""
Output: images/peso_ultimo_batch.pdf
Usage (from the repository root): python work_figures/peso_ultimo_batch.py
"""
import matplotlib.pyplot as plt

import thesis_font as tf

N_TEST, BATCH = 264, 64
LIGHT, LAST = "#c9dbf3", "#eb6834"

def main():
    sizes = [BATCH] * (N_TEST // BATCH) + [N_TEST % BATCH]          # 64, 64, 64, 64, 8
    share = [100 * s / N_TEST for s in sizes]                         # share of the test samples
    weight = [100 / len(sizes)] * len(sizes)                          # weight in the mean over batches
    pct = tf.percent()

    fig, ax = plt.subplots(figsize=tf.size(1.0, 3.0))
    for y, values in ((1, share), (0, weight)):
        left = 0
        for i, v in enumerate(values):
            last = i == len(values) - 1
            ax.barh(y, v, left=left, height=0.62, color=LAST if last else LIGHT, edgecolor="white", lw=1.2)
            if v > 10:                                               
                ax.text(left + v / 2, y, f"batch {i + 1}\n{tf.comma(v, 1)}{pct}", ha="center", va="center",
                        fontsize=7, color="white" if last else "0.15")
            left += v

    ax.annotate(f"ultimo batch: {sizes[-1]} campioni ({tf.comma(share[-1], 1)}{pct})",
                xy=(100 - share[-1] / 2, 1.32), xytext=(99, 1.55), ha="right", va="bottom", fontsize=7,
                color=LAST, arrowprops=dict(arrowstyle="->", color=LAST, lw=0.6))
    ax.text(100, -0.4, f"ultimo batch: {tf.comma(weight[-1], 0)}{pct}", ha="right", va="top", fontsize=7, color=LAST)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.75, 1.85)
    ax.set_yticks([1, 0], ["Quota dei campioni\ndi test", "Peso nella media\nsui batch (SenseFi)"])
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    tf.save(fig, "peso_ultimo_batch")

if __name__ == "__main__":
    main()
