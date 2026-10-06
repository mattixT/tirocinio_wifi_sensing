"""
All the numbers reported in the thesis, recomputed from the results.json files of the runs.
Nothing is trained here: the script only reads 04_ESPERIMENTI and prints (and saves) a report.

Sections of the report:
    1. accuracy per seed, mean ± std, 95% confidence interval, macro-F1 (every model and variant)
    2. comparison between models (baseline): Welch t-test and Mann-Whitney test
    3. exact accuracy vs SenseFi accuracy (mean over batches)
    4. precision, recall and F1 per class (summed confusion matrices, baseline)
    5. learning curves (baseline): test accuracy at some epochs, epoch with 100% training
       accuracy, final loss, and the optimistic "best epoch chosen on test" (NOT used as a result)
    6. misclassified test files: distinct files, blocks 180-199 and 190-199, binomial test
    7. validation variants: comparison with the baseline, validation vs test, epoch chosen on validation

Usage: python statistics.py            (report printed and saved in 04_ESPERIMENTI/statistics.txt)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

EXPERIMENTS = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/04_ESPERIMENTI")
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]
MODELS = ["cnn5", "gru", "lstm"]
VARIANTS = {"base": "", "random": "_val-random", "block": "_val-block"}     # variant -> folder suffix
TEST_PER_CLASS = 44                                                          # test files of every activity

lines = []                                     # the report, also saved to a text file


def out(text=""):
    """Print one line of the report and keep it for the text file."""
    print(text)
    lines.append(text)


def load_runs(folder, model, variant):
    """results.json of all the seeds of one model and variant, sorted by seed."""
    paths = sorted(folder.glob(f"{model}{VARIANTS[variant]}_seed*/results.json"))
    return [json.load(open(p)) for p in paths]


def pct(values):
    """Accuracies in [0, 1] -> numpy array in percent."""
    return 100 * np.array(values)


def mean_std_ci(values):
    """'mean ± sample std [95% CI]' of a list of values (already in percent)."""
    values = np.array(values)
    mean, std = values.mean(), values.std(ddof=1)
    half = stats.t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))
    return f"{mean:.2f} ± {std:.2f}  [CI 95%: {mean - half:.2f}; {mean + half:.2f}]"


def file_number(name):
    """'walk/walk188.mat' -> 188"""
    return int(re.search(r"\d+", name.split("/")[1]).group())


def misclassified(runs, key="test"):
    """Counts of (file, predicted class) over the runs, for the misclassified test files."""
    counts = {}
    for run in runs:
        res = run[key]
        for file, true, pred in zip(res["files"], res["y_true"], res["y_pred"]):
            if true != pred:
                counts[(file, ACTIVITIES[pred])] = counts.get((file, ACTIVITIES[pred]), 0) + 1
    return counts


def block_test(files, low, high):
    """How many distinct files are in the block low-high, and binomial test against chance.
    Under chance, a misclassified file is in the block with probability (block size) / 44."""
    k = sum(low <= file_number(f) <= high for f in files)
    p0 = (high - low + 1) / TEST_PER_CLASS
    p = stats.binomtest(k, len(files), p0, alternative="greater").pvalue if files else float("nan")
    return f"{k}/{len(files)} files in {low}-{high} (chance {100 * p0:.0f}%), binomial p = {p:.1e}"


# ----------------- MAIN -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute all the statistics of the thesis")
    parser.add_argument("--dir", default=str(EXPERIMENTS), help="folder with the runs")
    folder = Path(parser.parse_args().dir)

    runs = {(m, v): load_runs(folder, m, v) for m in MODELS for v in VARIANTS}
    acc = {k: pct([r["test"]["accuracy_exact"] for r in rs]) for k, rs in runs.items() if rs}

    # 1. accuracy, CI, macro-F1
    out("=== 1. Test accuracy (%) per seed, mean ± std, 95% CI, macro-F1 ===")
    for (m, v), values in acc.items():
        f1 = pct([r["test"]["macro_f1"] for r in runs[(m, v)]])
        out(f"{m:5s} {v:6s} seeds {np.round(values, 2).tolist()}")
        out(f"             accuracy {mean_std_ci(values)} | macro-F1 {f1.mean():.2f} ± {f1.std(ddof=1):.2f}")

    # 2. comparison between models (baseline)
    out("\n=== 2. Comparison between models (baseline) ===")
    for a, b in [("gru", "cnn5"), ("gru", "lstm"), ("cnn5", "lstm")]:
        x, y = acc[(a, "base")], acc[(b, "base")]
        welch = stats.ttest_ind(x, y, equal_var=False)
        mw = stats.mannwhitneyu(x, y)
        out(f"{a} vs {b}: Welch t = {welch.statistic:.2f}, p = {welch.pvalue:.3f} | Mann-Whitney p = {mw.pvalue:.3f}")

    # 3. exact accuracy vs SenseFi accuracy
    out("\n=== 3. Exact accuracy vs SenseFi accuracy (mean over batches), baseline ===")
    for m in MODELS:
        sensefi = pct([r["test"]["accuracy_sensefi"] for r in runs[(m, "base")]])
        out(f"{m:5s} exact   {np.round(acc[(m, 'base')], 2).tolist()} mean {acc[(m, 'base')].mean():.2f}")
        out(f"      SenseFi {np.round(sensefi, 2).tolist()} mean {sensefi.mean():.2f}")
    out(f"last test batch: {runs[('cnn5', 'base')][0]['test']['files'][-8:]}")

    # 4. per-class metrics from the summed confusion matrices
    out("\n=== 4. Precision / recall / F1 per class (%), summed confusion matrices, baseline ===")
    for m in MODELS:
        cm = sum(np.array(r["test"]["confusion_matrix"]) for r in runs[(m, "base")])
        recall = np.diag(cm) / cm.sum(axis=1)
        precision = np.diag(cm) / cm.sum(axis=0)
        f1 = 2 * precision * recall / (precision + recall)
        out(f"{m}: {cm.sum()} predictions, {cm.sum() - np.trace(cm)} errors")
        for i, act in enumerate(ACTIVITIES):
            out(f"   {act:6s} P {100 * precision[i]:6.2f}  R {100 * recall[i]:6.2f}  F1 {100 * f1[i]:6.2f}")

    # 5. learning curves
    out("\n=== 5. Learning curves (baseline, mean over the runs) ===")
    for m in MODELS:
        hist = [r["history"] for r in runs[(m, "base")]]
        get = lambda key: np.array([[h[key] for h in hs] for hs in hist])     # (runs, epochs)
        test_acc = 100 * get("test_accuracy")
        at = ", ".join(f"ep{e} {test_acc[:, e - 1].mean():.2f}" for e in (5, 10, 20, 30))
        first = [next((h["epoch"] for h in hs if h["accuracy_exact"] == 1.0), None) for hs in hist]
        geo = lambda key: 10 ** np.log10(get(key)[:, -1]).mean()             # geometric mean at epoch 30
        best = test_acc.max(axis=1)
        out(f"{m}: test accuracy {at}")
        out(f"      first epoch with 100% training accuracy per seed: {first}")
        out(f"      loss at epoch 30 (geometric mean): train {geo('loss'):.1e}, test {geo('test_loss'):.3f}")
        out(f"      best epoch chosen ON TEST (optimistic, not a result): mean {best.mean():.2f}, "
            f"epochs {(test_acc.argmax(axis=1) + 1).tolist()}")

    # 6. misclassified files
    out("\n=== 6. Misclassified test files (baseline) ===")
    union = set()
    for m in MODELS:
        counts = misclassified(runs[(m, "base")])
        files = {f for f, _ in counts}
        union |= files
        out(f"{m}: {sum(counts.values())} errors on {len(files)} files")
        out(f"      {block_test(files, 190, 199)}")
        out(f"      {block_test(files, 180, 199)}")
        detail = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        out("      " + "; ".join(f"{f.split('/')[1][:-4]}->{p} {n}/5" for (f, p), n in detail))
    out(f"all models: {len(union)} distinct files")
    out(f"      {block_test(union, 190, 199)}")
    out(f"      {block_test(union, 180, 199)}")

    # 7. validation variants
    out("\n=== 7. Validation variants ===")
    for m in MODELS:
        for v in ("random", "block"):
            rs = runs[(m, v)]
            if not rs:
                continue
            x, y = acc[(m, "base")], acc[(m, v)]
            best = pct([r["test_best_val"]["accuracy_exact"] for r in rs])
            hist = [r["history"] for r in rs]
            val30 = pct([hs[-1]["val_accuracy"] for hs in hist])
            gap = np.mean([np.mean([100 * (h["val_accuracy"] - h["test_accuracy"]) for h in hs[14:]]) for hs in hist])
            geo = lambda key: 10 ** np.mean([np.log10(hs[-1][key]) for hs in hist])
            files = {f for f, _ in misclassified(rs)}
            n_val = len(rs[0]["val_files"])
            out(f"{m:5s} {v:6s}: train {936 - n_val} / val {n_val} | test {y.mean():.2f} ± {y.std(ddof=1):.2f} "
                f"vs baseline {x.mean():.2f}: paired p = {stats.ttest_rel(x, y).pvalue:.3f}, "
                f"Welch p = {stats.ttest_ind(x, y, equal_var=False).pvalue:.3f}")
            out(f"             validation accuracy at epoch 30 {val30.mean():.2f} ± {val30.std(ddof=1):.2f} | "
                f"gap val-test epochs 15-30 {gap:.2f} points | "
                f"loss at epoch 30 (geom.) val {geo('val_loss'):.4f}, test {geo('test_loss'):.4f} "
                f"(test/val {geo('test_loss') / geo('val_loss'):.1f}x)")
            out(f"             epoch chosen on validation {[r['test_best_val']['epoch'] for r in rs]}: "
                f"test {best.mean():.2f} ± {best.std(ddof=1):.2f} vs epoch 30 {y.mean():.2f}, "
                f"paired p = {stats.ttest_rel(y, best).pvalue:.3f}")
            out(f"             misclassified test files: {block_test(files, 180, 199)}")

    report = folder / "stats_report.txt"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nReport saved in {report}")
