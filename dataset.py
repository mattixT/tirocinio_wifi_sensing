from pathlib import Path
from scipy.io import loadmat

# Inserisci qui il percorso della tua cartella NTU-Fi_HAR
dataset_dir = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/03_DATASET/NTU-Fi_HAR")

for split in ["train_amp", "test_amp"]:
    print(f"\n{split}")

    split_dir = dataset_dir / split

    for activity_dir in sorted(split_dir.iterdir()):
        if not activity_dir.is_dir():
            continue

        files = sorted(activity_dir.glob("*.mat"))
        print(f"{activity_dir.name}: {len(files)} file")

# Apre il primo campione disponibile
first_file = next((dataset_dir / "train_amp").rglob("*.mat"))
mat = loadmat(first_file)

print("\nPrimo file:", first_file)
print("Variabili presenti:", mat.keys())

csi = mat["CSIamp"]

print("Forma originale:", csi.shape)
print("Tipo numerico:", csi.dtype)
print("Valore minimo:", csi.min())
print("Valore massimo:", csi.max())

# Stessa riduzione temporale effettuata da SenseFi
csi_reduced = csi[:, ::4]
csi_reshaped = csi_reduced.reshape(3, 114, 500)

print("Forma usata da SenseFi:", csi_reshaped.shape)