"""
Generates all the figures used in the thesis.

Usage (from the repository root):
    python work_figures/make_all.py              all the figures
    python work_figures/make_all.py --skip-slow  without dataset_analysis.py (it reads all the
                                                   1200 samples of the dataset)
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import thesis_font

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIGURE = thesis_font.BASE / "05_FIGURE"

THESIS_FIGURES = [
    "schema_multicammino", "schema_ofdm_csi_reale", "schema_neurone", "funzioni_attivazione",
    "schema_convoluzione", "schema_cnn", "schema_rnn",                                    # chapter 2
    "suddivisione_campioni", "peso_ultimo_batch",                                        # chapter 3
    "suddivisione_validazione",                                                          # chapter 4
    "variabilita_seed", "esatta_vs_batch", "validazione_vs_test", "accuratezza_complessita",
    "normalizzazione_confronto",                                                         # chapter 5
]

TIKZ_FIGURES = ["schema_mimo_router",                                                    # chapter 2
                "schema_pipeline"]                                                       # chapter 4

MODELS = ["cnn5", "gru", "lstm"]
REPO_FIGURES = [
    # chapter 3
    ("dataset_exploration/activity_samples.py", ["--sample", "0"], ["campioni_per_attivita_0"], False),
    ("dataset_exploration/compare_samples_2act.py", [], ["confronto_walk_run"], False),
    ("dataset_exploration/dataset_analysis.py", [], ["variazione_temporale_per_attivita"], True),
    ("pipeline/preprocessing.py", [], ["preprocessing_test"], False),
    # chapter 5
    *[("results_analysis/summary.py", ["--model", m],
       [f"{m}_5runs_confusion_matrix", f"{m}_5runs_curves"], False) for m in MODELS],
    *[("results_analysis/summary.py", ["--model", "cnn5", "--val", v],
       [f"cnn5_val-{v}_5runs_curves"], False) for v in ["random", "block"]],
    ("results_analysis/errors_by_file.py", [], ["errori_per_numero_file"], False),
    *[("results_analysis/compare_files.py", ["--figure", f], [f"{f}_errori_vs_corretti"], False)
      for f in ["walk", "fall"]],
]

def run_tikz(name):
    if not thesis_font.USE_LATEX:
        print(f"ERROR in {name}.tex: pdflatex not found, the PDF in images is not changed")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                 "-output-directory", tmp, str(HERE / f"{name}.tex")],
                                capture_output=True, text=True)
        if result.returncode != 0:
            errors = [line for line in result.stdout.splitlines() if line.startswith("!")][:2]
            print(f"ERROR in {name}.tex:\n    " + "\n    ".join(errors or ["see the .log of pdflatex"]))
            return False
        target = thesis_font.IMAGES / f"{name}.pdf"
        shutil.copy2(Path(tmp) / f"{name}.pdf", target)
    print(f"saved {target}  [fonts: {', '.join(thesis_font.pdf_fonts(target))}]")
    return True

def run_repo_script(script, arguments, outputs):
    """Run one analysis script (without opening windows) and copy its PDFs into images."""
    env = dict(os.environ, MPLBACKEND="Agg")         
    result = subprocess.run([sys.executable, script, *arguments], cwd=REPO, env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        errors = [line for line in lines if "Error" in line][:2] or lines[-3:]
        print(f"ERROR in {script} {' '.join(arguments)}:\n    " + "\n    ".join(errors))
        return False
    for name in outputs:
        source = FIGURE / f"{name}.pdf"
        shutil.copy2(source, thesis_font.IMAGES / source.name)
        print(f"copied {source.name} -> images  [fonts: {', '.join(thesis_font.pdf_fonts(source))}]")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the figures of the thesis")
    parser.add_argument("--skip-slow", action="store_true", help="skip dataset_analysis.py")
    args = parser.parse_args()

    failed = []
    for name in THESIS_FIGURES:
        try:
            importlib.import_module(name).main()
        except Exception as error:               
            print(f"ERROR in {name}.py: {type(error).__name__}: {error}")
            failed.append(f"{name}.py")

    for name in TIKZ_FIGURES:
        if not run_tikz(name):
            failed.append(f"{name}.tex")

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
