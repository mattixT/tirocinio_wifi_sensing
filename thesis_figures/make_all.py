"""
Generates all the figures of the thesis, with the same font (Latin Modern) and the final sizes, and
puts them in 07_RELAZIONE/images (the existing files with the same name are replaced):
    1. the figures of this folder (schemes of chapter 2 and result figures), saved directly in images;
    2. the figures of the analysis scripts of the repository (dataset_exploration, pipeline,
       results_analysis): they are saved in 05_FIGURE as usual and then copied into images.

Usage (from the repository root):
    python thesis_figures/make_all.py              all the figures
    python thesis_figures/make_all.py --skip-slow  without the two slow ones (dataset_analysis.py reads
                                                   all the 1200 files, evaluate.py reloads a model)
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import thesis_font

REPO = Path(__file__).resolve().parents[1]
FIGURE = thesis_font.BASE / "05_FIGURE"

# figures of this folder: module name = name of the PDF
THESIS_FIGURES = [
    "schema_multicammino", "schema_ofdm_csi_reale", "schema_neurone", "funzioni_attivazione",
    "schema_convoluzione", "schema_cnn", "schema_rnn",                                  # chapter 2
    "suddivisione_campioni", "suddivisione_validazione", "peso_ultimo_batch",          # chapters 3-4
    "variabilita_seed", "esatta_vs_batch", "validazione_vs_test", "accuratezza_complessita",
    "normalizzazione_confronto", "normalizzazione_seed", "normalizzazione_differenze",  # chapter 5
]

# figures of the repository scripts: (script, arguments, PDFs it produces in 05_FIGURE, slow)
MODELS = ["cnn5", "gru", "lstm"]
VALIDATION = {"none": "", "random": "_val-random", "block": "_val-block"}
REPO_FIGURES = (
    [("dataset_exploration/activity_samples.py", ["--sample", n], [f"campioni_per_attivita_{n}"], False)
     for n in ["0", "15", "24"]]
    + [("dataset_exploration/compare_samples.py", ["--activity", a], [f"confronto_{a}"], False)
       for a in ["box", "circle", "clean", "fall", "run", "walk"]]
    + [("dataset_exploration/dataset_analysis.py", [], ["variazione_temporale_per_attivita"], True),
       ("pipeline/preprocessing.py", [], ["preprocessing_test"], False),
       ("pipeline/evaluate.py", ["--run", "cnn5_seed0", "--device", "cpu"], ["cnn5_seed0_training_curves"], True)]
    + [("results_analysis/summary.py", ["--model", m, "--val", v],
        [f"{m}{s}_5runs_confusion_matrix", f"{m}{s}_5runs_curves"], False)
       for m in MODELS for v, s in VALIDATION.items()]
    + [("results_analysis/errors_by_file.py", ["--val", v], [f"errori_per_numero_file{s}"], False)
       for v, s in VALIDATION.items()]
    + [("results_analysis/compare_files.py", ["--figure", f], [f"{f}_errori_vs_corretti"], False)
       for f in ["walk", "fall"]]
)


def run_repo_script(script, arguments, outputs):
    """Run one analysis script (without opening windows) and copy its PDFs into images."""
    env = dict(os.environ, MPLBACKEND="Agg")         # plt.show() does not open windows
    result = subprocess.run([sys.executable, script, *arguments], cwd=REPO, env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        errors = [line for line in lines if "Error" in line][:2] or lines[-3:]
        last_lines = "\n    ".join(errors)
        print(f"ERROR in {script} {' '.join(arguments)}:\n    {last_lines}")
        return False
    for name in outputs:
        source = FIGURE / f"{name}.pdf"
        shutil.copy2(source, thesis_font.IMAGES / source.name)
        print(f"copied {source.name} -> images  [fonts: {', '.join(thesis_font.pdf_fonts(source))}]")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all the figures of the thesis")
    parser.add_argument("--skip-slow", action="store_true", help="skip dataset_analysis.py and evaluate.py")
    args = parser.parse_args()

    for name in THESIS_FIGURES:
        importlib.import_module(name).main()

    failed = []
    for script, arguments, outputs, slow in REPO_FIGURES:
        if slow and args.skip_slow:
            continue
        print(f"running {script} {' '.join(arguments)}")
        if not run_repo_script(script, arguments, outputs):
            failed.append(f"{script} {' '.join(arguments)}")
    if failed:
        print("\nnot generated (see the errors above):\n  " + "\n  ".join(failed))
    else:
        print(f"\nall the figures are in {thesis_font.IMAGES}")
