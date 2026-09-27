"""
    This script is used to visualize CSI amplitude of a single sample of the dataset NTU-Fi_HAR
    The figure make it easier to undestand the structure (matrix) of the CSI data and how it changes over time in response to human activities.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat

DATASET = Path("../../03_DATASET/NTU-Fi_HAR")   # path to the dataset NTU-Fi_HAR
file = DATASET / "train_amp" / "run" / "run0.mat"

csi = loadmat(file)["CSIamp"] 
csi = csi.reshape(3, 114, 2000)   # 3 antennas, 114 subcarriers, 2000 samples

plt.figure(figsize=(10, 4))
plt.imshow(csi[0], aspect="auto", cmap="viridis")
plt.xlabel("Pacchetto")
plt.ylabel("Sottoportante")
plt.colorbar(label="Ampiezza CSI")
plt.title(f"{file.name} - coppia di antenne 1")
plt.show()