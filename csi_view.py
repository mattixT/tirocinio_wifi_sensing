"""
    This script is used to visualize CSI amplitude of a single sample of the dataset NTU-Fi_HAR
    The figure make it easier to understand the structure (matrix) of the CSI data and how it changes over time in response to human activities.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
DATASET = BASE / "03_DATASET" / "NTU-Fi_HAR"    # path to the dataset NTU-Fi_HAR
file = DATASET / "train_amp" / "fall" / "fall90.mat"

csi = loadmat(file)["CSIamp"] 
csi = csi.reshape(3, 114, 2000)   # 3 pairs of antennas, 114 subcarriers, 2000 packages

plt.figure(figsize=(10, 4))
plt.imshow(csi[0], aspect="auto", cmap="viridis", vmin=0, vmax=55)
plt.xlabel("Pacchetto (tempo)")
plt.ylabel("Sottoportante")
plt.colorbar(label="Ampiezza CSI")
plt.title(f"{file.name} - coppia di antenne 1")
plt.show()