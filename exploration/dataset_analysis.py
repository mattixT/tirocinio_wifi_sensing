"""
Statistics of the NTU-Fi HAR dataset (1200 files):
    1. shape check: every file must be (342, 2000)
    2. global mean and standard deviation over 1200 files, compared with the normalization constants of SenseFi
    3. temporal variation per activity (train files only): for every file, the standard deviation over time of each of
       the 342 rows, averaged over the rows; printed as mean ± std per activity and drawn as a boxplot

Output: the figure variazione_temporale_per_attivita.png

Usage (from the repository root): python exploration/dataset_analysis.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
DATASET = BASE / "03_DATASET" / "NTU-Fi_HAR"
FIGURE = BASE / "05_FIGURE"
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]

# containers for the results
shapes = set()
total, total_sq, n_values = 0.0, 0.0, 0
variation = {act: [] for act in ACTIVITIES}

for split in ["train_amp", "test_amp"]:
    for act in ACTIVITIES:
        for f in sorted((DATASET / split / act).glob("*.mat")):
            csi = loadmat(f)["CSIamp"]
            shapes.add(csi.shape)
            total += csi.sum()
            total_sq += (csi ** 2).sum()
            n_values += csi.size
            if split == "train_amp":
                # temporal variation: training files only (936 files)
                variation[act].append(csi.std(axis=1).mean())

# global mean and standard deviation over the whole dataset (train + test: 936 + 264 = 1200 files)
# this reproduces the normalization constants of SenseFi (mean = 42.3199, std = 4.9802)
mean = total / n_values
std = np.sqrt(total_sq / n_values - mean ** 2)

# print the results
print("\nFound shapes:", shapes)
print(f"Total number of values: {n_values:,}")
print(f"Global mean: {mean:.4f} (SenseFi: 42.3199)")
print(f"Global standard deviation: {std:.4f} (SenseFi: 4.9802)")

# print the mean and std of the temporal variation per activity
print("\nMean temporal variation per activity:")
for act in ACTIVITIES:
    v = np.array(variation[act])
    print(f"{act:>7}: {v.mean():.4f} ± {v.std():.4f}")

# plot the boxplot of the temporal variation per activity
plt.figure(figsize=(8, 4))
plt.boxplot([variation[a] for a in ACTIVITIES])
plt.xticks(range(1, len(ACTIVITIES) + 1), ACTIVITIES)
plt.ylabel("Deviazione standard nel tempo\n(media sulle sottoportanti)")
plt.title("Intensità della variazione temporale per attività - training set")
plt.grid(axis="y", alpha=0.3)
plt.savefig(FIGURE / "variazione_temporale_per_attivita.png", dpi=200, bbox_inches="tight")
plt.show()
