"""
Check of one saved run: it reloads the weights of a run, predicts the test
set again and compares the exact accuracy with the one saved by the training script. 
It then prints the per-class report (precision, recall, F1) and the confusion
matrix, and draws the training curves of the run.

Usage (from the repository root):
    python pipeline/evaluate.py --run <folder in 04_ESPERIMENTI> [--device <auto|cpu|mps|cuda>]
    e.g. python pipeline/evaluate.py --run gru_seed0 --device cpu
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from data import ACTIVITIES, get_loaders
from models import MODELS
from train import get_device

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
EXPERIMENTS = BASE / "04_ESPERIMENTI"
FIGURE = BASE / "05_FIGURE"


@torch.no_grad()
def predict(model, loader, device):
    """True and predicted labels of all the samples of the loader (numpy arrays)."""
    model.eval()
    y_true, y_pred = [], []
    for inputs, labels in loader:
        outputs = model(inputs.to(device))                  # (B, 6) logits
        y_pred.append(torch.argmax(outputs, dim=1).cpu())   # position of the highest logit = predicted class
        y_true.append(labels)                               # true class from the folder name
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def plot_training_curves(history, run_name):
    """Training loss and training accuracy per epoch of one run."""
    epochs = [h["epoch"] for h in history]                              # history is a list of 30 dicts, one per epoch

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["loss"] for h in history], marker="o")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss (scala logaritmica)")
    ax2.plot(epochs, [h["accuracy"] for h in history], marker="o")      # training accuracy, mean over batches
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Accuracy di training")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.suptitle(f"Andamento del training — {run_name}")
    fig.savefig(FIGURE / f"{run_name}_training_curves.pdf", dpi=200, bbox_inches="tight")


# ----------------- MAIN -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate one trained run on the test set")
    parser.add_argument("--run", default="cnn5_seed0", help="folder name in 04_ESPERIMENTI")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    run_dir = EXPERIMENTS / args.run

    device = get_device() if args.device == "auto" else torch.device(args.device)

    # results saved by train.py (architecture name, history, test metrics)
    with open(run_dir / "results.json") as f:
        results = json.load(f)

    # model with the trained weights
    model = MODELS[results["model"]](num_classes=6).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))

    # predictions on the test set, preprocessed as in the run (validation split and normalization)
    val = results.get("val", "none")
    _, _, test_loader = get_loaders(batch_size=64, val=None if val == "none" else val,
                                    norm=results.get("norm", "sensefi"))
    y_true, y_pred = predict(model, test_loader, device)

    # consistency check with the metrics saved during training
    accuracy = accuracy_score(y_true, y_pred)
    saved = results["test"]["accuracy_exact"]
    print(f"Run: {args.run} ({results['model']}, seed {results['seed']})")
    print(f"Accuracy now: {accuracy:.4f} | saved by train.py: {saved:.4f}"
          f" -> {'OK' if abs(accuracy - saved) < 1e-6 else 'DIFFERENT'}")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}\n")

    # per-class metrics
    print(classification_report(y_true, y_pred, target_names=ACTIVITIES, digits=4))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_true, y_pred))

    # training curves
    plot_training_curves(results["history"], args.run)
    plt.show()