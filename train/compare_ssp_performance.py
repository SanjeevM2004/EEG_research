"""
compare_ablation_balanced_focal.py
---------------------------------
Ablation study:
  (1) Linear probe on pooled RAW features
  (2) Linear probe on pooled MAE embeddings
  (3) MLP classifier on RAW features
  (4) MLP classifier on MAE embeddings

Fixes:
  - Balanced training set (equal # per class)
  - Focal loss for classifiers
  - Stratified validation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from data_construction.EEGFeatureDataset import EEGFeatureDataset
from models.mae import TransformerMAE


# -----------------------------
# Focal loss
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# -----------------------------
# Simple MLP classifier
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden=128, num_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def mae_encode_batch(mae: TransformerMAE, feats: torch.Tensor) -> torch.Tensor:
    mae.eval()
    return mae.encode(feats)


# -----------------------------
# Pool per-channel reps → (B,2*D)
# -----------------------------
def pool_rep(feats: torch.Tensor):
    mean = feats.mean(dim=1)
    std = feats.std(dim=1)
    return torch.cat([mean, std], dim=-1)


# -----------------------------
# Train linear probe
# -----------------------------
def run_linear_probe(train_loader, val_loader, device, rep, mae=None):
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    with torch.no_grad():
        for _, xb, yb in tqdm(train_loader, desc=f"[{rep.upper()} Linear] train reps"):
            xb = xb.to(device)
            if rep == "mae":
                xb = mae_encode_batch(mae, xb)
            pooled = pool_rep(xb).cpu().numpy()
            X_train_list.append(pooled)
            y_train_list.append(yb.cpu().numpy())

        for _, xb, yb in tqdm(val_loader, desc=f"[{rep.upper()} Linear] val reps"):
            xb = xb.to(device)
            if rep == "mae":
                xb = mae_encode_batch(mae, xb)
            pooled = pool_rep(xb).cpu().numpy()
            X_val_list.append(pooled)
            y_val_list.append(yb.cpu().numpy())

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)

    clf = LogisticRegression(
        solver="saga", max_iter=2000, n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)

    acc = accuracy_score(y_val, preds)
    bacc = balanced_accuracy_score(y_val, preds)
    f1m = f1_score(y_val, preds, average="macro")

    print(f"[{rep.upper()}-Linear] acc={acc:.4f} | bacc={bacc:.4f} | f1m={f1m:.4f}")
    return acc, bacc, f1m, preds, y_val


# -----------------------------
# Train MLP classifier
# -----------------------------
def run_mlp(train_loader, val_loader, device, rep, mae=None, epochs=10):
    num_classes = 5
    sample_signals, sample_feats, _ = train_loader.dataset[0]
    C, d_in = sample_feats.shape
    D_for_model = d_in if rep == "raw" else 128
    input_dim = 2 * D_for_model

    model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)

    # Class weights
    train_labels = [y for _, _, y in train_loader.dataset]
    counts = torch.bincount(torch.tensor(train_labels))
    alpha = (counts.sum() / (len(counts) * torch.clamp(counts.float(), min=1))).to(device)

    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for _, xb, yb in tqdm(train_loader, desc=f"[{rep.upper()} MLP] Epoch {epoch}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            if rep == "mae":
                xb = mae_encode_batch(mae, xb)
            pooled = pool_rep(xb)

            opt.zero_grad(set_to_none=True)
            logits = model(pooled)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()

        print(f"Epoch {epoch}: train loss={run_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for _, xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            if rep == "mae":
                xb = mae_encode_batch(mae, xb)
            pooled = pool_rep(xb)
            preds = model(pooled).argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    print(f"[{rep.upper()}-MLP] acc={acc:.4f} | bacc={bacc:.4f} | f1m={f1m:.4f}")
    return acc, bacc, f1m, preds, labels


# -----------------------------
# Confusion matrix
# -----------------------------
def plot_confusion(model_name, preds, labels, class_names, save_dir="./results"):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"cm_{model_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved confusion matrix: {save_path}")


# -----------------------------
# Main ablation
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = EEGFeatureDataset(
        root_dir="./EEG_data/Physionet/",
        fs=160, tmin=-0.5, tmax=4.0,
        cache_path="./EEG_data/dataset_cache.pt",
        rebuild=False
    )

    # Stratified split
    labels = np.array(dataset.labels)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.arange(len(labels)), labels))

    # ✅ Balanced training set
    train_labels = labels[train_idx]
    by_class = defaultdict(list)
    for i, y in zip(train_idx, train_labels):
        by_class[y].append(i)

    min_count = min(len(idxs) for idxs in by_class.values())
    balanced_idx = []
    for c, idxs in by_class.items():
        chosen = np.random.choice(idxs, min_count, replace=False)
        balanced_idx.extend(chosen.tolist())

    train_ds = Subset(dataset, balanced_idx)
    val_ds = Subset(dataset, val_idx.tolist())
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Load MAE encoder
    sample_signals, sample_feats, _ = dataset[0]
    C, d_in = sample_feats.shape
    mae = TransformerMAE(d_in=d_in, n_channels=C,
                         d_model=128, nhead=4, num_layers=4,
                         dim_feedforward=256, dropout=0.1).to(device)
    mae.load_state_dict(torch.load("./models_saved/mae_eeg.pt", map_location=device))
    print("✅ Loaded MAE checkpoint.")

    results = {}

    # Linear probes
    results["raw_linear"] = run_linear_probe(train_loader, val_loader, device, "raw", mae=None)
    results["mae_linear"] = run_linear_probe(train_loader, val_loader, device, "mae", mae=mae)

    # MLP classifiers
    results["raw_mlp"] = run_mlp(train_loader, val_loader, device, "raw", mae=None, epochs=10)
    results["mae_mlp"] = run_mlp(train_loader, val_loader, device, "mae", mae=mae, epochs=10)

    # Plot comparison
    os.makedirs("./results", exist_ok=True)
    methods = list(results.keys())
    accs = [results[m][0] for m in methods]
    baccs = [results[m][1] for m in methods]
    f1ms = [results[m][2] for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, accs, width, label="Acc")
    plt.bar(x, baccs, width, label="BAcc")
    plt.bar(x + width, f1ms, width, label="F1m")
    plt.xticks(x, methods, rotation=30)
    plt.ylabel("Score")
    plt.title("Ablation Study (Balanced Training + Focal Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./results/ablation_balanced_focal.png", dpi=300)
    print("✅ Saved ./results/ablation_balanced_focal.png")

    # Confusion matrices
    ACTION_LABELS = ["rest", "left_fist", "right_fist", "both_fists", "feet"]
    for m in results.keys():
        preds, labels = results[m][3], results[m][4]
        plot_confusion(m, preds, labels, ACTION_LABELS)

    print("\n=== Final Results ===")
    for m, (a, b, f, _, _) in results.items():
        print(f"{m:10s} -> acc={a:.4f}, bacc={b:.4f}, f1m={f:.4f}")


if __name__ == "__main__":
    main()
