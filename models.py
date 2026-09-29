"""
Models for NTU-Fi HAR,
"""

import torch
import torch.nn as nn

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