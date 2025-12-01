import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ======================================================================
# CovCNN: simple CNN classifier for SPD matrices
# ======================================================================

class CovCNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, n_classes)

        # 🔥 enforce float32 ALWAYS
        self.float()

    def forward(self, C):
        # C: (B, d, d)
        C = C.float()                    # <<< FIX #1
        x = C.unsqueeze(1)               # (B,1,d,d)
        h = self.net(x.float())          # <<< FIX #2
        h = h.view(h.size(0), -1)        # (B,64)
        logits = self.fc(h.float())      # <<< FIX #3
        return logits


# ======================================================================
# Wrapper classifier like TSLR / MDM / LDA
# ======================================================================

class CovCNNClassifier:
    def __init__(self, n_channels: int, n_classes: int, lr=1e-3, epochs=20, device="cpu"):
        self.device = device
        self.model = CovCNN(n_channels, n_classes).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.n_channels = n_channels
        self.n_classes = n_classes

    def fit(self, X, y):
        """
        X : (N,d,d) numpy or torch
        y : (N,)
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        else:
            X = X.float()

        y = torch.tensor(y, dtype=torch.long)

        X = X.to(self.device)
        y = y.to(self.device)

        N = X.shape[0]
        batch = min(128, N)

        self.model.train()

        for ep in range(self.epochs):
            idx = torch.randperm(N)[:batch]
            Cb = X[idx].float()               # <<< FIX #4
            yb = y[idx]

            logits = self.model(Cb)           # <<< FIX #5
            loss = F.cross_entropy(logits, yb)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        else:
            X = X.float()

        X = X.to(self.device)
        logits = self.model(X.float())         # <<< FIX #6
        return logits.argmax(dim=1).cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean()
