"""
This script measures some statistics of the training set of NTU-Fi HAR dataset.
The three statistics allow:
    - find out if all the files have the same shape (342, 2000)
    - calculate global mean and the standard deviation
    - measure the variation during time of the signal for each activity (dynamic component)
Only the training set is used, to avoid data leakage, since the test set is used for evaluation of the model.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
TRAIN = BASE / "03_DATASET" / "NTU-Fi_HAR" / "train_amp"
FIGURE = BASE / "05_FIGURE"
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]

# containers for the results
shapes = set()
total, total_sq, n_values = 0.0, 0.0, 0
variation = {act: [] for act in ACTIVITIES}

for act in ACTIVITIES:
    files = sorted((TRAIN / act).glob("*.mat"))

    for f in files:
        csi = loadmat(f)["CSIamp"]
        shapes.add(csi.shape)
        total += csi.sum()
        total_sq += (csi ** 2).sum()
        n_values += csi.size
        variation[act].append(csi.std(axis=1).mean())

    print(f"{act}: {len(files)} files loaded")

mean = total / n_values
std = np.sqrt(total_sq / n_values - mean ** 2)

print("\nFound shapes:", shapes)
print(f"Total number of values: {n_values:,}")
print(f"Global mean: {mean:.4f} (SenseFI: 42.3199)")
print(f"Global standard deviation: {std:.4f} (SenseFi: 4.9802)")

print("\nMean temporal variation per activity:")
for act in ACTIVITIES:
    v = np.array(variation[act])
    print(f"{act:>7}: {v.mean():.4f} ± {v.std():.4f}")

plt.figure(figsize=(8, 4))
plt.boxplot([variation[a] for a in ACTIVITIES])
plt.xticks(range(1, len(ACTIVITIES) + 1), ACTIVITIES)
plt.ylabel("Deviazione standard nel tempo\n(media sulle sottoportanti)")
plt.title("Intensità della variazione temporale per attività - training set")
plt.grid(axis="y", alpha=0.3)
plt.savefig(FIGURE / "variazione_temporale_per_attivita.png", dpi=200, bbox_inches="tight")
plt.show()




