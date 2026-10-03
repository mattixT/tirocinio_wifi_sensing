"""
Error analysis on the test set for one model, over all its runs (5 seeds).
This script prints, for each test file misclassified at least once, how many runs it is
wrong and which classes were predicted (useful to understand which activities are confused).

Usage: python error_analysis.py --model <model_name>
"""

import argparse
import json
from collections import Counter
from pathlib import Path

EXPERIMENTS = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/04_ESPERIMENTI")
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn5")
    MODEL = parser.parse_args().model

    runs = sorted(EXPERIMENTS.glob(f"{MODEL}_seed*/results.json"))
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

    print(f"{sum(wrong.values())} errors in {len(runs)} runs, "
        f"on {len(wrong)} different test files\n")
    # most_common() returns a list of files sorted from the most to the least frequently misclassified
    for file, n in wrong.most_common():
        print(f"{file:<22} wrong in {n}/{len(runs)} runs -> predicted: {', '.join(predicted_as[file])}")