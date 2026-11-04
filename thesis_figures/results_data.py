"""
Results of the experiments (folders of 04_ESPERIMENTI, one results.json per run) used by the
figures of chapters 4 and 5. A variant is "" (baseline), "val-random", "val-block",
"norm-sample" or "norm-subcarrier"; the runs are returned in the order of the seed (0-4).
"""
import json
import re

import numpy as np

from thesis_font import EXPERIMENTS

SEEDS = range(5)
NORMS = {"": "SenseFi", "norm-sample": "per campione", "norm-subcarrier": "per sottoportante"}


def runs(model, variant=""):
    """The results.json of the 5 runs of a model, as dictionaries, seed 0 first."""
    prefix = f"{model}_{variant}_seed" if variant else f"{model}_seed"
    paths = sorted(EXPERIMENTS.glob(f"{prefix}*/results.json"),
                   key=lambda p: int(re.search(r"seed(\d+)$", p.parent.name).group(1)))
    if not paths:
        raise SystemExit(f"no runs {prefix}* in {EXPERIMENTS}")
    return [json.loads(p.read_text()) for p in paths]


def test_accuracy(model, variant="", key="accuracy_exact"):
    """Test accuracy (%) of the 5 runs. key="accuracy_sensefi" gives the mean over the batches."""
    return np.array([100 * run["test"][key] for run in runs(model, variant)])


def mean_difference(variant, baseline):
    """Difference between the mean accuracies, computed on the means rounded to 2 decimals as in the
    tables of the thesis (so that the figures show the same numbers as the text)."""
    return round(variant.mean(), 2) - round(baseline.mean(), 2)


def file_number(name):
    """'walk/walk188.mat' -> 188."""
    return int(re.search(r"\d+", name.split("/")[1]).group())
