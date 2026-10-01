import json
import argparse
from collections import Counter
from pathlib import Path

EXPERIMENTS = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/04_ESPERIMENTI")
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="cnn5")
MODEL = parser.parse_args().model

runs = sorted(EXPERIMENTS.glob(f"{MODEL}_seed*/results.json"))
wrong = Counter()        # file -> number of runs in which it is misclassified
predicted_as = {}        # file -> predicted classes in those runs

for path in runs:
    with open(path) as f:
        res = json.load(f)["test"]
    for file, t, p in zip(res["files"], res["y_true"], res["y_pred"]):
        if t != p:
            wrong[file] += 1
            predicted_as.setdefault(file, []).append(ACTIVITIES[p])

print(f"{sum(wrong.values())} errors in {len(runs)} runs, "
      f"on {len(wrong)} different test files\n")
for file, n in wrong.most_common():
    print(f"{file:<22} wrong in {n}/{len(runs)} runs -> predicted: {', '.join(predicted_as[file])}")