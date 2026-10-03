"""
Training and test of one monedl on NTU-Fi HAR, following the official SenseFi protocol.
Additions to SenseFi:
    - fixed seed, exact accuracy, macro-F1 and confusion matrix
    - test loss and accuracy at every epoch, only to draw the curves

Usage: python train.py --model >model_name> --seed <n.seed> --device <{gpu,cpu}>
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

from data import get_loaders, list_files
from models import MODELS

EXPERIMENTS = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/04_ESPERIMENTI")
NUM_EPOCHS = 30          
LEARNING_RATE = 1e-3     # SenseFi


def get_device():
    """Use the GPU if available (NVIDIA 'cuda' or Apple 'mps'), otherwise the CPU.
    Note: for GRU/LSTM the CPU is faster, because their steps are sequential"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    """Fix all random number generators, so that the same seed gives the same run"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    """Loss and exact accuracy on a loader, used only to draw the curves"""

    rng_state = torch.get_rng_state()
    model.eval()
    total_loss, correct = 0.0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, labels).item() * inputs.size(0)
        correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
    torch.set_rng_state(rng_state)
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def train(model, train_loader, test_loader, num_epochs, learning_rate, criterion, device):
    """Training loop as in SenseFi. Returns training and test metrics of every epoch"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_accuracy, correct = 0.0, 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()                  # 1. reset the gradients
            outputs = model(inputs)                # 2. forward: 6 scores per sample
            loss = criterion(outputs, labels)      # 3. cross-entropy loss
            loss.backward()                        # 4. gradients for every weight
            optimizer.step()                       # 5. Adam updates the weights

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1)
            epoch_accuracy += (predict_y == labels).sum().item() / labels.size(0)   # SenseFi
            correct += (predict_y == labels).sum().item()                            # exact

        epoch_loss /= len(train_loader.dataset)    # mean loss per sample
        epoch_accuracy /= len(train_loader)        # mean accuracy over batches (SenseFi)
        exact_accuracy = correct / len(train_loader.dataset)

        test_loss, test_accuracy = evaluate_epoch(model, test_loader, criterion, device)

        history.append({"epoch": epoch + 1,
                        "loss": epoch_loss, "accuracy": epoch_accuracy, "accuracy_exact": exact_accuracy,
                        "test_loss": test_loss, "test_accuracy": test_accuracy})
        print(f"Epoch:{epoch + 1}, Accuracy:{epoch_accuracy:.4f}, Loss:{epoch_loss:.9f}"
              f" | test accuracy:{test_accuracy:.4f}, test loss:{test_loss:.5f}")

    return history


@torch.no_grad()
def test(model, loader, criterion, device):
    """Test as in SenseFi, plus exact accuracy, macro-F1 and confusion matrix."""
    model.eval()
    test_acc, test_loss, correct = 0.0, 0.0, 0
    y_true, y_pred = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        predict_y = torch.argmax(outputs, dim=1)

        test_acc += (predict_y == labels).sum().item() / labels.size(0)   # SenseFi
        test_loss += loss.item() * inputs.size(0)
        correct += (predict_y == labels).sum().item()                      # exact
        y_true += labels.cpu().tolist()
        y_pred += predict_y.cpu().tolist()

    test_acc /= len(loader)
    test_loss /= len(loader.dataset)
    exact_acc = correct / len(loader.dataset)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\nTest accuracy (SenseFi, mean over batches): {test_acc:.4f}, loss: {test_loss:.5f}")
    print(f"Test accuracy (exact, {correct}/{len(loader.dataset)}): {exact_acc:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}")
    return {"accuracy_sensefi": test_acc, "accuracy_exact": exact_acc, "macro_f1": macro_f1,
            "loss": test_loss, "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "y_true": y_true, "y_pred": y_pred}

# ----------------- MAIN -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test one model on NTU-Fi HAR (SenseFi protocol)")
    parser.add_argument("--seed", type=int, default=0, help="random seed of the run")
    parser.add_argument("--model", default="cnn5", choices=list(MODELS.keys()), help="cnn5, gru or lstm")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device} | model: {args.model} | seed: {args.seed}")

    train_loader, test_loader = get_loaders(batch_size=64)
    model = MODELS[args.model](num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    history = train(model, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, criterion, device)
    print(f"Training time: {time.time() - start:.0f} s")

    results = test(model, test_loader, criterion, device)
    results["files"] = list_files("test_amp")     # file name of every test sample, same order

    out_dir = EXPERIMENTS / f"{args.model}_seed{args.seed}"          # one folder per run
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "results.json", "w") as f:
        json.dump({"model": args.model, "seed": args.seed, "device": str(device),
                   "epochs": NUM_EPOCHS, "learning_rate": LEARNING_RATE,
                   "history": history, "test": results}, f, indent=2)
    print("Saved in", out_dir)