"""Addestramento e valutazione di un modello HAR su NTU-Fi HAR.

Esempio:
    python train.py --model cnn5 --seed 0
    python train.py --model gru --norm zscore --step 8 --seed 1

Ogni run salva in 04_ESPERIMENTI/<nome_run>/:
    config.json, history.csv, metrics.json, curves.png,
    confusion_matrix.png, best_model.pt
"""
import argparse, json, csv, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_datasets, CLASSES, N_PKT
from models import build_model, count_params

BASE = Path(__file__).resolve().parents[2]          # .../TIROCINIO_TESI
DEFAULT_DATA = BASE / "03_DATASET" / "NTU-Fi_HAR"
DEFAULT_OUT = BASE / "04_ESPERIMENTI"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def pick_device(pref):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, preds, labels = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(criterion(out, y).item() * len(y))
        preds.append(out.argmax(1).cpu()); labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return (sum(losses) / len(labels), accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro"), preds, labels)


def plot_curves(hist, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, lab in [("train_loss", "train"), ("val_loss", "validation")]:
        ax[0].plot(ep, [h[k] for h in hist], label=lab)
    for k, lab in [("train_acc", "train"), ("val_acc", "validation")]:
        ax[1].plot(ep, [h[k] for h in hist], label=lab)
    ax[0].set(xlabel="Epoca", ylabel="Loss"); ax[1].set(xlabel="Epoca", ylabel="Accuracy")
    for a in ax: a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_cm(cm, path, title):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="Classe predetta", ylabel="Classe reale", title=title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "black", fontsize=9)
    fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cnn5", choices=["cnn5", "gru", "lstm"])
    p.add_argument("--norm", default="sensefi", choices=["sensefi", "zscore", "subcarrier", "none"])
    p.add_argument("--step", type=int, default=4, help="sottocampionamento temporale (4 -> T=500)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--tag", default="", help="etichetta per raggruppare esperimenti")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    T = N_PKT // args.step
    run = f"{args.tag + '_' if args.tag else ''}{args.model}_{args.norm}_T{T}_s{args.seed}"
    out = Path(args.out) / run; out.mkdir(parents=True, exist_ok=True)

    tr, va, te = get_datasets(args.data, args.step, args.norm, seed=args.seed)
    g = torch.Generator().manual_seed(args.seed)
    tr_dl = DataLoader(tr, args.batch, shuffle=True, generator=g)
    va_dl, te_dl = DataLoader(va, 256), DataLoader(te, 256)

    model = build_model(args.model, len(CLASSES), T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    cfg = vars(args) | {"T": T, "device": str(device), "params": count_params(model),
                         "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print(f"[{run}] device={device} params={cfg['params']:,} train/val/test={len(tr)}/{len(va)}/{len(te)}")

    hist, best_f1, t0 = [], -1, time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot, correct, n = 0.0, 0, 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); out_ = model(x); loss = crit(out_, y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y); correct += (out_.argmax(1) == y).sum().item(); n += len(y)
        vl, vacc, vf1, _, _ = evaluate(model, va_dl, device, crit)
        hist.append({"epoch": ep, "train_loss": tot / n, "train_acc": correct / n,
                     "val_loss": vl, "val_acc": vacc, "val_f1": vf1})
        # selezione del modello sulla validazione (macro-F1, a parita' vince la loss minore)
        if vf1 > best_f1 or (vf1 == best_f1 and vl < best_vl):
            best_f1, best_vl, best_ep = vf1, vl, ep
            torch.save(model.state_dict(), out / "best_model.pt")
        print(f"ep {ep:3d} | train loss {tot/n:.4f} acc {correct/n:.3f} | val loss {vl:.4f} acc {vacc:.3f} f1 {vf1:.3f}")

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
    plot_curves(hist, out / "curves.png")

    # valutazione finale, una sola volta, sul test set con il modello scelto in validazione
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device, weights_only=True))
    _, tacc, tf1, preds, labels = evaluate(model, te_dl, device, crit)
    cm = confusion_matrix(labels, preds)
    plot_cm(cm, out / "confusion_matrix.png", f"{args.model.upper()} - test acc {tacc:.3f}")
    metrics = {"test_acc": tacc, "test_macro_f1": tf1, "best_epoch": best_ep, "best_val_f1": best_f1,
               "train_time_s": round(time.time() - t0, 1), "confusion_matrix": cm.tolist(),
               "report": classification_report(labels, preds, target_names=CLASSES, output_dict=True, digits=4)}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(f"TEST acc {tacc:.4f} macro-F1 {tf1:.4f} (epoca scelta {best_ep}) -> {out}")


if __name__ == "__main__":
    main()
