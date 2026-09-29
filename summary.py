import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FIGURE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/05_FIGURE")
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]
EXPERIMENTS = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/04_ESPERIMENTI")
MODEL = "cnn5"

acc, f1, cms = [], [], []
for path in sorted(EXPERIMENTS.glob(f"{MODEL}_seed*/results.json")):
    with open(path) as f:
        res = json.load(f)["test"]
    acc.append(res["accuracy_exact"])
    f1.append(res["macro_f1"])
    cms.append(np.array(res["confusion_matrix"]))
    print(f"{path.parent.name}: accuracy {res['accuracy_exact']:.4f} | macro-F1 {res['macro_f1']:.4f}")

acc, f1 = np.array(acc), np.array(f1)
print(f"\n{MODEL} over {len(acc)} runs:")
print(f"  accuracy: {100 * acc.mean():.2f} ± {100 * acc.std(ddof=1):.2f} %")
print(f"  macro-F1: {100 * f1.mean():.2f} ± {100 * f1.std(ddof=1):.2f} %")
cm_total = sum(cms)
print(f"\nConfusion matrix summed over {len(cms)} runs (rows = true, columns = predicted):")
print(cm_total)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_total, cmap="Blues")
ax.set_xticks(range(6), ACTIVITIES, rotation=45)
ax.set_yticks(range(6), ACTIVITIES)
ax.set_xlabel("Classe predetta")
ax.set_ylabel("Classe reale")
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm_total[i, j], ha="center", va="center",
                color="white" if cm_total[i, j] > cm_total.max() / 2 else "black")
fig.colorbar(im, ax=ax)
fig.savefig(FIGURE / f"{MODEL}_confusion_matrix_{len(cms)}runs.png", dpi=200, bbox_inches="tight")
plt.show()