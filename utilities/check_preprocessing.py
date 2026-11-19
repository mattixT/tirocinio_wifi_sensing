"""
Check if the preprocessing of this repository (pipeline/preprocessing.py) gives exactly the same data as the official SenseFi code.
All the 1200 samples of NTU-Fi HAR are preprocessed in both ways and compared.

Usage (from the repository root): python utilities/check_preprocessing.py
"""
from pathlib import Path
import sys

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))   # folder of preprocessing.py
from preprocessing import load_sample

DATASET = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/03_DATASET/NTU-Fi_HAR")

def sensefi_preprocessing(path):
    """The same operations of the SenseFi code."""
    x = sio.loadmat(path)["CSIamp"]
    x = (x - 42.3199) / 4.9802          # normalize
    x = x[:, ::4]                       # sampling: 2000 -> 500
    x = x.reshape(3, 114, 500)
    x = torch.FloatTensor(x)            # convert to float32 tensor
    return x.numpy()                    # back to numpy array for comparison with load_sample

files = sorted(DATASET.glob("*_amp/*/*.mat"))
max_diff = max(np.abs(load_sample(f) - sensefi_preprocessing(f)).max() for f in files)
print(f"{len(files)} samples compared, maximum absolute difference: {max_diff}")