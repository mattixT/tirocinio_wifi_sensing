"""
The three models for NTU-Fi HAR, reproduced without changes (same layers and same number
of parameters as the official code). All of them take a batch (B, 3, 114, 500) and return 6 logits.
    
    - NTU_Fi_LeNet (CNN-5): 3 convolutional + 2 fully connected layers -> 476,582 parameters
    - NTU_Fi_GRU:  1 GRU layer (342 inputs, hidden size 64) + linear 64 -> 78,726 parameters
    - NTU_Fi_LSTM: 1 LSTM layer (342 inputs, hidden size 64) + linear 64 -> 104,838 parameters

MODELS maps the name used on the command line (cnn5, gru, lstm) to the class.
The script prints the output shape of every CNN-5 layer and the parameters of the three models.

Usage (from the repository root): python pipeline/models.py
"""

import torch
import torch.nn as nn

class NTU_Fi_LeNet(nn.Module):
    """CNN-5: 3 convolutional layers followed by 2 fully connected layers."""
    def __init__(self, num_classes):
        super(NTU_Fi_LeNet, self).__init__()
        self.encoder = nn.Sequential(
            # 32 filters of 3 x 15 x 23, step 9
            nn.Conv2d(3, 32, (15, 23), stride=9),
            nn.ReLU(True),
            # 64 filters of 3 x 3 on the 32 maps, step 3
            nn.Conv2d(32, 64, 3, stride=(1, 3)),
            nn.ReLU(True),
            # 96 filters of 7 x 3
            nn.Conv2d(64, 96, (7, 3), stride=(1, 3)),
            nn.ReLU(True),
        )
        # classifier
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        """x: (B, 3, 114, 500) -> logits (B, 6)."""
        x = self.encoder(x)
        x = x.view(-1, 96 * 4 * 6)
        out = self.fc(x)
        return out

class NTU_Fi_GRU(nn.Module):
    """GRU: one recurrent layer (342 inputs per time step, hidden size 64) and a
    linear layer on the hidden state of the last time step."""
    def __init__(self, num_classes):
        super(NTU_Fi_GRU, self).__init__()
        # 342 inputs per step, memory of 64 values and 3 blocks (update gate, reset gate, candidate)
        self.gru = nn.GRU(342, 64, num_layers = 1) 
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """x: (B, 3, 114, 500), read as a sequence of 500 steps of 342 values -> logits (B, 6)."""
        x = x.view(-1, 342, 500)
        x = x.permute(2, 0, 1)
        _, ht = self.gru(x)         # _, ht = hidden state at every step, hidden state after the last step
        outputs = self.fc(ht[-1])   # ht[-1] is the last layer
        return outputs

class NTU_Fi_LSTM(nn.Module):
    """LSTM: same structure as NTU_Fi_GRU, with an LSTM cell instead of a GRU cell."""
    def __init__(self, num_classes):
        super(NTU_Fi_LSTM, self).__init__()
        # same input and hidden size as the GRU, but 4 blocks (forget, input, output gates, candidate)
        self.lstm = nn.LSTM(342, 64, num_layers = 1)
        self.fc = nn.Linear(64, num_classes)    # 64 -> 6 logits

    def forward(self, x):
        """x: (B, 3, 114, 500), read as a sequence of 500 steps of 342 values -> logits (B, 6)."""
        x = x.view(-1, 342, 500)
        x = x.permute(2, 0, 1)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])   # (B, 64) -> (B, 6) logits
        return outputs

# available models that can be used to train and test on NTU-Fi HAR dataset (train.py --model <model_name>)
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