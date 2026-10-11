"""
Training and test of one model on NTU-Fi HAR with the SenseFi protocol:
cross-entropy loss, Adam (lr 1e-3), batch 64, 30 epochs, no validation, test once after the last epoch.

Additions to SenseFi (none of them changes the training):
    - fixed seed (Python, NumPy, PyTorch), set before the model is created
    - exact accuracy (correct / total) next to the SenseFi one (mean of the batch accuracies),
      macro-F1, confusion matrix and the prediction of every test sample
    - test loss and accuracy at every epoch, ONLY to draw the curves (never used for any choice);
      the state of the random generator is saved and restored around this evaluation
    - optional validation set (--val random | block): validation curves, and test also of the
      model at the epoch with the lowest validation loss

Output: one folder per run, 04_ESPERIMENTI/<model>[_val-<mode>]_seed<N>/, containing
    model.pt            weights after the last epoch (main result, as in SenseFi)
    model_best_val.pt   weights at the epoch with the lowest validation loss (only with --val)
    results.json        configuration, per-epoch history and test results (fields in README.md)

Usage (from the repository root):
    python pipeline/train.py --model <cnn5|gru|lstm> --seed <n> [--device <auto|cpu|mps|cuda>] [--val <none|random|block>]
"""

import argparse
import json                         # .json files are written to memorize configuration, history and test results
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score      # final metrics on the test set

from data import get_loaders        # DataLoaders for training, validation and test
from models import MODELS           # name of the model to train and test

EXPERIMENTS = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI/04_ESPERIMENTI")
NUM_EPOCHS = 30                     # SenseFi
LEARNING_RATE = 1e-3                # SenseFi


def get_device():
    """Use the GPU if available (NVIDIA 'cuda' or Apple 'mps'), otherwise the CPU.
    Note: for GRU/LSTM the CPU is faster, because their time steps are sequential."""
    if torch.cuda.is_available():               # not on a MAC
        return torch.device("cuda")
    if torch.backends.mps.is_available():       # Apple GPU
        return torch.device("mps")
    return torch.device("cpu")                  # otherwise the processor


def set_seed(seed):
    """Fix the random generators of Python, NumPy and PyTorch, so that the same seed gives the same run.
    It is called before the model is created: the seed decides both the initial weights and the
    order of the batches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    """Mean loss and exact accuracy of the model on a loader, used only to draw the curves."""

    rng_state = torch.get_rng_state()
    model.eval()
    total_loss, correct = 0.0, 0

    for inputs, labels in loader:                                           # one batch at a time
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)                                             # (B, 6) logits
        total_loss += criterion(outputs, labels).item() * inputs.size(0)
        correct += (torch.argmax(outputs, dim=1) == labels).sum().item()    # number of correct predictions

    torch.set_rng_state(rng_state)                         # restore the RNG state, so that the next epoch is not affected
    return total_loss / len(loader.dataset), correct / len(loader.dataset)      # mean loss, exact accuracy


def train(model, train_loader, test_loader, num_epochs, learning_rate, criterion, device, val_loader=None):
    """Training loop. Every epoch records training loss and accuracy (SenseFi and exact),
    test loss and accuracy and, if val_loader is given, validation loss and accuracy."""

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []                                                        # list of dicts, one per epoch, with training/test/validation loss and accuracy
    best = {"epoch": None, "val_loss": float("inf"), "state": None}     # best validation epoch (if val_loader is given)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_accuracy, correct = 0.0, 0.0, 0

        for inputs, labels in train_loader:        # 15 batches in a random order every epoch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()                  # reset the gradients
            outputs = model(inputs)                # forward: (B, 6) logits
            loss = criterion(outputs, labels)      # cross-entropy loss
            loss.backward()                        # backpropagation: gradients for every weight
            optimizer.step()                       

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1)
            epoch_accuracy += (predict_y == labels).sum().item() / labels.size(0)    
            correct += (predict_y == labels).sum().item()    # count of correct predictions

        epoch_loss /= len(train_loader.dataset)    # mean loss per sample
        epoch_accuracy /= len(train_loader)        # mean accuracy over batches 
        exact_accuracy = correct / len(train_loader.dataset)    # exact: correct / 936 (total training samples)

        test_loss, test_accuracy = evaluate_epoch(model, test_loader, criterion, device)
        record = {"epoch": epoch + 1,
                  "loss": epoch_loss, "accuracy": epoch_accuracy, "accuracy_exact": exact_accuracy,
                  "test_loss": test_loss, "test_accuracy": test_accuracy}
        message = f"Epoch:{epoch + 1}, Accuracy:{epoch_accuracy:.4f}, Loss:{epoch_loss:.9f}"

        if val_loader is not None:                                     # only in the validation variants
            val_loss, val_accuracy = evaluate_epoch(model, val_loader, criterion, device)
            record.update({"val_loss": val_loss, "val_accuracy": val_accuracy})
            message += f" | val accuracy:{val_accuracy:.4f}, val loss:{val_loss:.5f}"
            if val_loss < best["val_loss"]:        # new best epoch
                best = {"epoch": epoch + 1, "val_loss": val_loss,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                
        history.append(record)
        print(message + f" | test accuracy:{test_accuracy:.4f}, test loss:{test_loss:.5f}")

    return history, best


@torch.no_grad()
def test(model, loader, criterion, device):
    """Final test as in SenseFi (accuracy = mean over batches) plus exact accuracy, macro-F1,
    confusion matrix and the true/predicted label of every sample. Returns the dict saved in results.json."""
    model.eval()
    test_acc, test_loss, correct = 0.0, 0.0, 0
    y_true, y_pred = [], []                     # true and predicted class of every test sample

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)                     # (B, 6) logits
        loss = criterion(outputs, labels)
        predict_y = torch.argmax(outputs, dim=1)    # predicted class of every sample

        test_acc += (predict_y == labels).sum().item() / labels.size(0)   # accuracy of this batch
        test_loss += loss.item() * inputs.size(0)
        correct += (predict_y == labels).sum().item()                      # exact: count of correct
        y_true += labels.cpu().tolist()
        y_pred += predict_y.cpu().tolist()

    test_acc /= len(loader)                         # mean of the 5 batch accuracies 
    test_loss /= len(loader.dataset)                # mean loss per sample
    exact_acc = correct / len(loader.dataset)       # exact: correct / 264 (total test samples)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\nTest accuracy (SenseFi, mean over batches): {test_acc:.4f}, loss: {test_loss:.5f}")
    print(f"Test accuracy (exact, {correct}/{len(loader.dataset)}): {exact_acc:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}")
    return {"accuracy_sensefi": test_acc, "accuracy_exact": exact_acc, "macro_f1": macro_f1,
            "loss": test_loss, "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "y_true": y_true, "y_pred": y_pred}     # utils for next error analysis

# ----------------- MAIN -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test one model on NTU-Fi HAR (SenseFi protocol)")
    parser.add_argument("--seed", type=int, default=0, help="random seed of the run")
    parser.add_argument("--model", default="cnn5", choices=list(MODELS.keys()), help="cnn5, gru or lstm")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--val", default="none", choices=["none", "random", "block"],
                        help="none = SenseFi (no validation); random = stratified 20%% of the training set; " \
                        "block = files 140-159 of every activity")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device} | model: {args.model} | seed: {args.seed} | validation: {args.val}")

    train_loader, val_loader, test_loader = get_loaders(batch_size=64, val=None if args.val == "none" else args.val)
    model = MODELS[args.model](num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    history, best = train(model, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, criterion, 
                          device, val_loader=val_loader)        # 30
    print(f"Training time: {time.time() - start:.0f} s")

    # model after the last epoch
    print("\n[model after the last epoch]")
    results = test(model, test_loader, criterion, device)
    results["files"] = test_loader.dataset.files

    suffix = "" if args.val == "none" else f"_val-{args.val}"
    out_dir = EXPERIMENTS / f"{args.model}{suffix}_seed{args.seed}"     # one folder per run
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")

    # only with a validation set: model of the epoch with the lowest validation loss
    best_results = None
    if val_loader is not None:
        print(f"\n[model of the best validation epoch: {best['epoch']}]")
        model.load_state_dict(best["state"])
        best_results = test(model, test_loader, criterion, device)
        best_results["epoch"] = best["epoch"]
        best_results["val_loss"] = best["val_loss"]
        torch.save(best["state"], out_dir / "model_best_val.pt")

    # writing of the configuration, history and test results in a JSON file
    with open(out_dir / "results.json", "w") as f:
        json.dump({"model": args.model, "seed": args.seed, "device": str(device),
                   "epochs": NUM_EPOCHS, "learning_rate": LEARNING_RATE, "val": args.val,
                   "val_files": val_loader.dataset.files if val_loader else None,
                   "history": history, "test": results, "test_best_val": best_results}, f, indent=2)
    print("Saved in", out_dir)