import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data import ACTIVITIES, CSIDataset, load_split
from models import MODELS
from train import get_device

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
EXPERIMENTS = BASE / "04_ESPERIMENTI"
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


def plot_training_curves(history, run_name):
    """Plot training loss (log scale) and training accuracy per epoch."""
    epochs = [h["epoch"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["loss"] for h in history], marker="o")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss (scala logaritmica)")
    ax2.plot(epochs, [h["accuracy"] for h in history], marker="o")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Accuracy di training")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.suptitle(f"Andamento del training — {run_name}")
    fig.savefig(FIGURE / f"{run_name}_training_curves.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate one trained run on the test set")
    parser.add_argument("--run", default="cnn5_seed0", help="folder name in 04_ESPERIMENTI")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    run_dir = EXPERIMENTS / args.run

    device = get_device() if args.device == "auto" else torch.device(args.device)

    # 1. results saved by train.py (architecture name, history, test metrics)
    with open(run_dir / "results.json") as f:
        results = json.load(f)

    # 2. model with the trained weights
    model = MODELS[results["model"]](num_classes=6).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))

    # 3. predictions on the test set
    X_test, y_test = load_split("test_amp")
    test_loader = DataLoader(CSIDataset(X_test, y_test), batch_size=64, shuffle=False)
    y_true, y_pred = predict(model, test_loader, device)

    # 4. consistency check with the metrics saved during training
    accuracy = accuracy_score(y_true, y_pred)
    saved = results["test"]["accuracy_exact"]
    print(f"Run: {args.run} ({results['model']}, seed {results['seed']})")
    print(f"Accuracy now: {accuracy:.4f} | saved by train.py: {saved:.4f}"
          f" -> {'OK' if abs(accuracy - saved) < 1e-6 else 'DIFFERENT'}")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}\n")

    # 5. per-class metrics
    print(classification_report(y_true, y_pred, target_names=ACTIVITIES, digits=4))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_true, y_pred))

    # 6. training curves
    plot_training_curves(results["history"], args.run)
    plt.show()