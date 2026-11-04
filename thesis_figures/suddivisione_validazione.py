"""
Figure: the three ways of splitting the 200 samples of an activity. Baseline (SenseFi): training and
test only. Random validation: 20% of the training files, drawn at random (different for every activity:
the figure shows the files of box). Block validation: files 140-159, the same for every activity.
The validation files are read from the results of the runs (val_files).

Output: images/suddivisione_validazione.pdf
Usage (from the repository root): python thesis_figures/suddivisione_validazione.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import thesis_font as tf
from results_data import file_number, runs

TRAIN, VAL, TEST = "#2a78d6", "#1baf7a", "#eb6834"
TEST_BLOCKS = [(16, 4), (160, 40)]
EXAMPLE = "box"                                      # activity shown for the random validation


def validation_numbers(variant):
    """File numbers of the validation set of EXAMPLE (the same for every seed and model)."""
    files = runs("cnn5", variant)[0]["val_files"]
    return sorted(file_number(f) for f in files if f.startswith(f"{EXAMPLE}/"))


def main():
    rows = [("Baseline (SenseFi)", []),
            (f"Validazione casuale\n(esempio: {EXAMPLE})", validation_numbers("val-random")),
            ("Validazione a blocchi", validation_numbers("val-block"))]
    fig, ax = plt.subplots(figsize=tf.size(1.0, 4.3))
    for i, (_, val) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.broken_barh([(0, 200)], (y - 0.32, 0.64), color=TRAIN)
        ax.broken_barh([(n, 1) for n in val], (y - 0.32, 0.64), color=VAL)
        ax.broken_barh(TEST_BLOCKS, (y - 0.32, 0.64), color=TEST)

    ax.set_xlim(0, 200)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_yticks(range(len(rows)), [name for name, _ in rows][::-1])
    ax.set_xticks([0, 18, 140, 160, 199], ["0", "16–19", "140", "160", "199"])
    ax.set_xlabel("numero del campione (per ciascuna attività)")
    ax.tick_params(axis="y", length=0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    handles = [Patch(color=c) for c in (TRAIN, VAL, TEST)]
    fig.legend(handles, ["addestramento", "validazione", "test"], ncol=3, loc="lower center",
               bbox_to_anchor=(0.6, 0.98), handlelength=2.5, columnspacing=1.6)
    tf.save(fig, "suddivisione_validazione")


if __name__ == "__main__":
    main()
