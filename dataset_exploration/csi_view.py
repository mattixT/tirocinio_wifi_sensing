"""
Heatmap of the raw CSI amplitude of a single sample (antenna pair 1): subcarriers on the vertical
axis, packets (time) on the horizontal axis. It is a quick way to look at the structure of a sample:
horizontal bands are the static part of the channel, vertical traces are the movement of the person.
The file is chosen with the constant FILE; the figure is only shown, not saved.

Usage (from the repository root): python dataset_exploration/csi_view.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
DATASET = BASE / "03_DATASET" / "NTU-Fi_HAR"    # path to the dataset NTU-Fi_HAR
FILE = DATASET / "train_amp" / "box" / "box134.mat"

csi = loadmat(FILE)["CSIamp"] 
csi = csi.reshape(3, 114, 2000)   # 3 antenna pairs, 114 subcarriers, 2000 packets

# building the figure
plt.figure(figsize=(10, 4))
plt.imshow(csi[0], aspect="auto", cmap="viridis", vmin=0, vmax=55)
plt.xlabel("Pacchetto (tempo)")
plt.ylabel("Sottoportante")
plt.colorbar(label="Ampiezza CSI")
plt.title(f"{FILE.name} - coppia di antenne 1")
plt.show()