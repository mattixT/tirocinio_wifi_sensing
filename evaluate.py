import json
from pathlib import Path

import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data import load_split, CSIDataset, ACTIVITIES
from models import NTU_Fi_LeNet
from train import get_device

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
FIGURE = BASE / "05_FIGURE"


@torch.no_grad()
def predict(model, loader, device):
    """Return true labels and predicted labels for all samples of the loader."""
    model.eval()
    y_true, y_pred = [], []
    for inputs, labels in loader:
        outputs = model(inputs.to(device))
        y_pred.append(torch.argmax(outputs, dim=1).cpu())   # back to CPU for numpy/sklearn
        y_true.append(labels)
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate one trained run on the test set")
    parser.add_argument("--run", default="cnn5_seed0", help="folder name in 04_ESPERIMENTI")
    args = parser.parse_args()
    RUN_DIR = BASE / "04_ESPERIMENTI" / args.run

    device = get_device()

    # 1. model with the trained weights
    model = NTU_Fi_LeNet(num_classes=6).to(device)
    model.load_state_dict(torch.load(RUN_DIR / "model.pt", map_location=device, weights_only=True))

    # 2. test set only (the training set is not needed here)
    X_test, y_test = load_split("test_amp")
    test_loader = DataLoader(CSIDataset(X_test, y_test), batch_size=64, shuffle=False)
    y_true, y_pred = predict(model, test_loader, device)

    # 3. metrics
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}\n")
    print(classification_report(y_true, y_pred, target_names=ACTIVITIES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(cm)

    print("\nErrors:")
    for i in np.where(y_true != y_pred)[0]:
        print(f"  test sample {i}: true {ACTIVITIES[y_true[i]]} -> predicted {ACTIVITIES[y_pred[i]]}")

    # 4. confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), ACTIVITIES, rotation=45)
    ax.set_yticks(range(6), ACTIVITIES)
    ax.set_xlabel("Classe predetta")
    ax.set_ylabel("Classe reale")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.savefig(FIGURE / f"{args.run}_confusion_matrix.png", dpi=200, bbox_inches="tight")
    
    # 5. training curves (from the history saved by train.py)
    with open(RUN_DIR / "results.json") as f:
        history = json.load(f)["history"]
    epochs = [h["epoch"] for h in history]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["loss"] for h in history], marker="o")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss (scala logaritmica)")
    ax2.plot(epochs, [h["accuracy"] for h in history], marker="o")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Accuracy di training")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig2.savefig(FIGURE / f"{args.run}_training_curves.png", dpi=200, bbox_inches="tight")
    plt.show()