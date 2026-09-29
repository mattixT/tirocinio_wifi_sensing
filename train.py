"""
"""

import json
import time
from pathlib import Path
import argparse
import random
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

import torch
import torch.nn as nn

from data import get_loaders, list_files
from models import NTU_Fi_LeNet

EXPERIMENTS = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/04_ESPERIMENTI")
NUM_EPOCHS = 30          # SenseFi, NTU-Fi HAR
LEARNING_RATE = 1e-3     # SenseFi


def get_device():
    """Use the GPU if available (NVIDIA 'cuda' or Apple 'mps'), otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed):
    """Fix all random number generators, so that the same seed gives the same run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(model, loader, num_epochs, learning_rate, criterion, device):
    """Training loop as in SenseFi. Returns loss and accuracy of every epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_accuracy = 0.0, 0.0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()                  # 1. reset the gradients
            outputs = model(inputs)                # 2. forward: 6 scores per sample
            loss = criterion(outputs, labels)      # 3. cross-entropy loss
            loss.backward()                        # 4. gradients for every weight
            optimizer.step()                       # 5. Adam updates the weights

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1)
            epoch_accuracy += (predict_y == labels).sum().item() / labels.size(0)

        epoch_loss /= len(loader.dataset)          # mean loss per sample
        epoch_accuracy /= len(loader)              # mean accuracy over batches (SenseFi)
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_accuracy})
        print(f"Epoch:{epoch + 1}, Accuracy:{epoch_accuracy:.4f}, Loss:{epoch_loss:.9f}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test the CNN-5 on NTU-Fi HAR (SenseFi protocol)")
    parser.add_argument("--seed", type=int, default=0, help="random seed of the run")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | seed: {args.seed}")

    train_loader, test_loader = get_loaders(batch_size=64)
    model = NTU_Fi_LeNet(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    history = train(model, train_loader, NUM_EPOCHS, LEARNING_RATE, criterion, device)
    print(f"Training time: {time.time() - start:.0f} s")

    results = test(model, test_loader, criterion, device)
    results["files"] = list_files("test_amp")     # file name of every test sample, same order

    out_dir = EXPERIMENTS / f"cnn5_seed{args.seed}"          # one folder per run
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "results.json", "w") as f:
        json.dump({"model": "NTU_Fi_LeNet", "seed": args.seed, "epochs": NUM_EPOCHS,
                   "learning_rate": LEARNING_RATE, "history": history, "test": results}, f, indent=2)
    print("Saved in", out_dir)