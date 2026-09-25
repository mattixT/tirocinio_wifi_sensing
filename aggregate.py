"""Raccoglie i metrics.json di tutti i run in una tabella (media +- dev. std sui seed).
    python aggregate.py            -> stampa la tabella e salva 04_ESPERIMENTI/summary.csv
"""
import json, csv
from collections import defaultdict
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[2] / "04_ESPERIMENTI"
rows, groups = [], defaultdict(list)
for m in sorted(EXP.glob("*/metrics.json")):
    cfg = json.load(open(m.parent / "config.json")); met = json.load(open(m))
    key = (cfg.get("tag", ""), cfg["model"], cfg["norm"], cfg["T"])
    groups[key].append((met["test_acc"], met["test_macro_f1"], cfg["params"], met["train_time_s"]))
for (tag, model, norm, T), v in sorted(groups.items()):
    a = np.array(v)
    rows.append({"tag": tag, "model": model, "norm": norm, "T": T, "n_seed": len(v), "params": int(a[0, 2]),
                 "acc_mean": a[:, 0].mean(), "acc_std": a[:, 0].std(), "f1_mean": a[:, 1].mean(),
                 "f1_std": a[:, 1].std(), "time_s_mean": a[:, 3].mean()})
if rows:
    with open(EXP / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['tag']:>10} {r['model']:>5} {r['norm']:>10} T={r['T']:<4} n={r['n_seed']} "
              f"acc {100*r['acc_mean']:.2f}+-{100*r['acc_std']:.2f}  F1 {100*r['f1_mean']:.2f}+-{100*r['f1_std']:.2f}")
