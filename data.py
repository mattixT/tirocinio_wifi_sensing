"""

"""

from pathlib import Path
import numpy as np
from preprocessing import load_sample

DATASET = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/03_DATASET/NTU-Fi_HAR")
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]

def load_split(split):
    X, y = [], []
    for label, act in enumerate(ACTIVITIES):
        for f in sorted((DATASET / split / act).glob("*.mat")):
            X.append(load_sample(f))
            y.append(label)
    return np.stack(X), np.array(y, dtype=np.int64)

if __name__ == "__main__":
    X_train, y_train = load_split("train_amp")
    X_test, y_test = load_split("test_amp")

    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)
    print("Samples per class (train):", np.bincount(y_train))
    print("Samples per class (test):", np.bincount(y_test))
    print(f"Memory used by X_train: {X_train.nbytes / 1024**2:.2f} MB")
    print("First sample label:", y_train[0], "=" , ACTIVITIES[y_train[0]])
    print("Last sample label:", y_train[-1], "=" , ACTIVITIES[y_train[-1]])