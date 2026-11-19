"""
Results of the experiments (folders of 04_ESPERIMENTI, one results.json per run) used by the
figures of chapters 4 and 5. A variant is "" (baseline), "val-random", "val-block",
"norm-sample" or "norm-subcarrier".
"""
import json
import re

import numpy as np

from thesis_font import EXPERIMENTS

SEEDS = range(5)
NORMS = {"": "SenseFi", "norm-sample": "per campione", "norm-subcarrier": "per sottoportante"}

def runs(model, variant=""):
    prefix = f"{model}_{variant}_seed" if variant else f"{model}_seed"
    paths = sorted(EXPERIMENTS.glob(f"{prefix}*/results.json"),
                   key=lambda p: int(re.search(r"seed(\d+)$", p.parent.name).group(1)))
    if not paths:
        raise SystemExit(f"no runs {prefix}* in {EXPERIMENTS}")
    return [json.loads(p.read_text()) for p in paths]

def test_accuracy(model, variant="", key="accuracy_exact"):
    return np.array([100 * run["test"][key] for run in runs(model, variant)])

def mean_difference(variant, baseline):
    return round(variant.mean(), 2) - round(baseline.mean(), 2)

def file_number(name):
    return int(re.search(r"\d+", name.split("/")[1]).group())
