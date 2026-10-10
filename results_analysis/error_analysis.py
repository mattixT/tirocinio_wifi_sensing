"""
Test files misclassified by one model over its runs (5 seeds).
For every file misclassified at least once it prints in how many runs it is wrong and the predicted
classes, from the most to the least frequent error (text output only, nothing is saved).
It is the starting point of the error analysis: the figures are made by errors_by_file.py and
compare_files.py, the statistics (binomial test on the blocks of files) by stats_report.py.

Usage (from the repository root): python analysis/error_analysis.py --model <cnn5|gru|lstm> [--val <none|random|block>]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

EXPERIMENTS = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/04_ESPERIMENTI")
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test files misclassified by one model over its runs")
    parser.add_argument("--model", default="cnn5", choices=["cnn5", "gru", "lstm"])
    parser.add_argument("--val", default="none", choices=["none", "random", "block"])
    args = parser.parse_args()
    suffix = "" if args.val == "none" else f"_val-{args.val}"

    runs = sorted(EXPERIMENTS.glob(f"{args.model}{suffix}_seed*/results.json"))
    wrong = Counter()        # file -> number of runs in which it is misclassified
    predicted_as = {}        # file -> predicted classes in those runs

    for path in runs:
        with open(path) as f:
            res = json.load(f)["test"]
        # three lists in the same order: file name, true label, predicted label
        for file, t, p in zip(res["files"], res["y_true"], res["y_pred"]):
            if t != p:
                wrong[file] += 1
                predicted_as.setdefault(file, []).append(ACTIVITIES[p])

    print(f"{args.model}{suffix}: {sum(wrong.values())} errors in {len(runs)} runs, "
          f"on {len(wrong)} different test files\n")
    # most_common() returns a list of files sorted from the most to the least frequently misclassified
    for file, n in wrong.most_common():
        print(f"{file:<22} wrong in {n}/{len(runs)} runs -> predicted: {', '.join(predicted_as[file])}")