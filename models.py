"""
Models for NTU-Fi HAR,
"""

import torch
import torch.nn as nn

# CNN model for NTU-Fi HAR dataset (based on  SenseFi approach)
class NTU_Fi_LeNet(nn.Module):

    def __init__(self, num_classes):
        super(NTU_Fi_LeNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, (15, 23), stride=9),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=(1, 3)),
            nn.ReLU(True),
            nn.Conv2d(64, 96, (7, 3), stride=(1, 3)),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 96 * 4 * 6)
        out = self.fc(x)
        return out

# GRU model for NTU-Fi HAR dataset (based on  SenseFi approach)
class NTU_Fi_GRU(nn.Module):

    def __init__(self, num_classes):
        super(NTU_Fi_GRU, self).__init__()
        self.gru = nn.GRU(342, 64, num_layers = 1)  # 342 inputs per step, memory of 64 values
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 342, 500)
        x = x.permute(2, 0, 1)
        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])
        return outputs

# LSTM model for NTU-Fi HAR dataset (based on  SenseFi approach)
class NTU_Fi_LSTM(nn.Module):

    def __init__(self, num_classes):
        super(NTU_Fi_LSTM, self).__init__()
        self.lstm = nn.LSTM(342, 64, num_layers = 1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 342, 500)
        x = x.permute(2, 0, 1)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs

# available models that can be used to train and test on NTU-Fi HAR dataset (train.py -- model <model_name>)
MODELS = {"cnn5": NTU_Fi_LeNet, "gru": NTU_Fi_GRU, "lstm": NTU_Fi_LSTM}

# ----------------- TEST -----------------

if __name__ == "__main__":
    model = NTU_Fi_LeNet(num_classes=6)
    x = torch.zeros(2, 3, 114, 500)
    print("Input:", tuple(x.shape))
    for layer in model.encoder:
        x = layer(x)
        print(f"{layer.__class__.__name__:>7} -> {tuple(x.shape)}")
    x = x.view(-1, 96 * 4 * 6)
    print("Flatten ->", tuple(x.shape))
    for layer in model.fc:
        x = layer(x)
        print(f"{layer.__class__.__name__:>7} -> {tuple(x.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    print("\nAll models (fake batch of 2 samples):")
    x = torch.zeros(2, 3, 114, 500)
    for name, model_class in MODELS.items():
        m = model_class(num_classes=6)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:>5}: output {tuple(m(x).shape)} | parameters {n:,}")