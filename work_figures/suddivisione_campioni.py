"""
Figure: split of the 200 samples of every activity of NTU-Fi HAR into training and test set. 

Output: images/suddivisione_campioni.pdf
Usage (from the repository root): python work_figures/suddivisione_campioni.py
"""
import matplotlib.pyplot as plt

import thesis_font as tf

TRAIN, TEST = "#2a78d6", "#eb6834"
TEST_BLOCKS = [(16, 4), (160, 40)]                  # (first file, number of files)

def main():
    n_test = sum(n for _, n in TEST_BLOCKS)
    fig, ax = plt.subplots(figsize=tf.size(1.0, 2.3))
    ax.broken_barh([(0, 200)], (0, 1), color=TRAIN)
    ax.broken_barh(TEST_BLOCKS, (0, 1), color=TEST)

    ax.annotate("test (16–19)", xy=(18, 1.02), xytext=(18, 1.32), ha="center", va="bottom",
                color=TEST, fontsize=7, arrowprops=dict(arrowstyle="->", color=TEST, lw=0.6))
    ax.text(100, 1.32, tf.bold(f"addestramento: {200 - n_test} campioni"), color=TRAIN, ha="center", va="bottom")
    ax.text(180, 1.32, tf.bold(f"test: {n_test} campioni"), color=TEST, ha="center", va="bottom")

    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1.6)
    ax.set_xticks([0, 18, 160, 199], ["0", "16–19", "160", "199"])
    ax.set_yticks([])
    ax.set_xlabel("numero del campione (uguale per tutte le attività)")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    tf.save(fig, "suddivisione_campioni")

if __name__ == "__main__":
    main()
