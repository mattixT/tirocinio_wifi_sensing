"""Modelli di riferimento, ripresi da SenseFi (NTU-Fi) e resi indipendenti
dalla lunghezza temporale T per l'analisi sulla lunghezza del segnale.
Input: (B, 3, 114, T)."""
import torch
import torch.nn as nn


class CNN5(nn.Module):
    """CNN-5 di SenseFi (derivata da LeNet-5): 3 conv + 2 fully connected."""
    def __init__(self, num_classes=6, T=500):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, (15, 23), stride=9), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=(1, 3)), nn.ReLU(True),
            nn.Conv2d(64, 96, (7, 3), stride=(1, 3)), nn.ReLU(True),
        )
        with torch.no_grad():
            n = self.encoder(torch.zeros(1, 3, 114, T)).numel()
        self.fc = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.fc(self.encoder(x).flatten(1))


class _RecurrentNet(nn.Module):
    """Il campione e' trattato come sequenza di T vettori da 342 (3x114) valori;
    la classificazione usa l'ultimo stato nascosto."""
    rnn_cls = None

    def __init__(self, num_classes=6, T=500, hidden=64):
        super().__init__()
        self.rnn = self.rnn_cls(342, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.flatten(1, 2).transpose(1, 2)      # (B, T, 342)
        _, h = self.rnn(x)
        if isinstance(h, tuple):                  # LSTM restituisce (h, c)
            h = h[0]
        return self.fc(h[-1])


class GRU(_RecurrentNet):
    rnn_cls = nn.GRU


class LSTM(_RecurrentNet):
    rnn_cls = nn.LSTM


MODELS = {"cnn5": CNN5, "gru": GRU, "lstm": LSTM}


def build_model(name, num_classes=6, T=500):
    return MODELS[name](num_classes=num_classes, T=T)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
