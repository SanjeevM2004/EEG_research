"""
SPDNet-Log + Subject Fisher (Mini-DCR) + MLP Action Head
LOSO evaluation on BCI-IV 2a (bci_active4.pt)
"""

from __future__ import annotations
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from time import time

# ============================================================
# CONFIG
# ============================================================

CACHE_PATH = "./EEG_data/bci_active4.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STEPS = 1000
BATCH_SIZE = 256
LR = 1e-3

LAMBDA_CE = 1.0
LAMBDA_FISHER = 1.0
LAMBDA_SUBJ = 0.05

LAMBDA_WITHIN = 1.0
LAMBDA_BETWEEN = 1.0

SEED = 50


# ============================================================
# HELPERS
# ============================================================

def symmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.transpose(-1, -2))


def logm_spd(C: torch.Tensor, eps_eig=1e-5, jitter=1e-4):
    C = symmetrize(C.float())
    B, d, _ = C.shape
    C = C + jitter * torch.eye(d, device=C.device).unsqueeze(0)
    w, V = torch.linalg.eigh(C.double())
    w = torch.clamp(w, min=eps_eig)
    S = V @ torch.diag_embed(torch.log(w)) @ V.transpose(-1, -2)
    return symmetrize(S.float())


def tangent_upper_vec(S: torch.Tensor) -> torch.Tensor:
    B, d, _ = S.shape
    idx = torch.triu_indices(d, d, device=S.device)
    return S[:, idx[0], idx[1]]


# ============================================================
# SUBJECT FISHER LOSS (Mini-DCR)
# ============================================================

def subject_alignment_loss(
    feat: torch.Tensor,
    subjects: torch.Tensor,
    lambda_within: float = 1.0,
    lambda_between: float = 1.0,
):
    B, D = feat.shape
    unique_subj = torch.unique(subjects)

    subj_means = []
    subj_sizes = []
    within = feat.new_tensor(0.0)

    # Within-subject
    for s in unique_subj:
        idx = (subjects == s)
        n_s = idx.sum()
        if n_s <= 1:
            continue
        f_s = feat[idx]
        mu_s = f_s.mean(0, keepdim=True)
        within += ((f_s - mu_s) ** 2).sum() / (n_s * D)

        subj_means.append(mu_s)
        subj_sizes.append(n_s)

    if len(subj_means) <= 1:
        return feat.new_tensor(0.0)

    subj_means = torch.cat(subj_means, dim=0)
    subj_sizes = torch.tensor(subj_sizes, device=feat.device, dtype=feat.dtype)
    mu_global = subj_means.mean(0, keepdim=True)

    between = (((subj_means - mu_global) ** 2).sum(dim=1) * subj_sizes).sum()
    between = between / (subj_sizes.sum() * D)

    return lambda_within * within - lambda_between * between


# ============================================================
# SPDLinear
# ============================================================

class SPDLinear(nn.Module):
    def __init__(self, d_in, d_out, eps=1e-4):
        super().__init__()
        self.d_in, self.d_out, self.eps = d_in, d_out, eps
        self.W = nn.Parameter(torch.randn(d_in, d_out) * (1 / np.sqrt(d_in)))

    def forward(self, C: torch.Tensor):
        C = symmetrize(C.float())
        B, d_in, _ = C.shape
        C = C + self.eps * torch.eye(d_in, device=C.device)

        W = self.W.float()
        CW = C @ W
        C_out = CW.transpose(1, 2) @ W
        C_out = symmetrize(C_out)
        C_out = C_out + self.eps * torch.eye(self.d_out, device=C.device)
        return C_out


# ============================================================
# SPDNet-LOG + MLP HEAD
# ============================================================

class SPDNetLog(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()

        d = n_channels
        dims = [d, d * 2, d * 2, d]

        self.spd_layers = nn.ModuleList([
            SPDLinear(dims[0], dims[1]),
            SPDLinear(dims[1], dims[2]),
            SPDLinear(dims[2], dims[3]),
        ])

        self.d_last = dims[-1]
        self.vec_dim = self.d_last * (self.d_last + 1) // 2

        # ⭐ NEW: MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(self.vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, C: torch.Tensor):
        C_curr = symmetrize(C.float())
        for layer in self.spd_layers:
            C_curr = layer(C_curr)

        S_last = logm_spd(C_curr)
        feat = tangent_upper_vec(S_last)

        logits = self.mlp(feat)
        return logits, S_last


# ============================================================
# TRAINING
# ============================================================

def prepare_covs(covs_np):
    C = torch.from_numpy(covs_np).float()
    C = symmetrize(C)
    tr = C.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    return C / (tr + 1e-6)


def train_one_fold(X_train_np, y_train_np, subj_train_np, n_classes):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    C_train = prepare_covs(X_train_np).to(DEVICE)
    y = torch.tensor(y_train_np, device=DEVICE)
    subj = torch.tensor(subj_train_np, device=DEVICE)

    N, d, _ = C_train.shape

    model = SPDNetLog(n_channels=d, n_classes=n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    idx_all = torch.arange(N, device=DEVICE)

    model.train()
    best_loss = float("inf")
    best_state = None

    for step in range(1, STEPS + 1):
        idx = idx_all[torch.randint(0, N, (min(BATCH_SIZE, N),), device=DEVICE)]
        C_b = C_train[idx]
        y_b = y[idx]
        subj_b = subj[idx]

        logits, S_last = model(C_b)
        ce = F.cross_entropy(logits, y_b)

        feat_b = tangent_upper_vec(S_last)
        subj_loss = subject_alignment_loss(feat_b, subj_b)

        loss = LAMBDA_CE * ce + LAMBDA_SUBJ * subj_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {"model": model.state_dict()}

        if step == 1 or step % 100 == 0:
            print(f"[Train] {step}/{STEPS} | Loss={loss.item():.4f} | CE={ce.item():.4f} | Subj={subj_loss.item():.4f}")

    model.load_state_dict(best_state["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, X_test_np, y_test_np):
    C_test = prepare_covs(X_test_np).to(DEVICE)
    y_test = torch.tensor(y_test_np, device=DEVICE)

    logits, _ = model(C_test)
    preds = torch.argmax(logits, dim=-1)
    return (preds == y_test).float().mean().item()


def run_loso_experiment(covs, labels, subjects):
    S_ids = np.unique(subjects)
    n_classes = int(labels.max() + 1)

    accs = []
    for sid in S_ids:
        train_mask = subjects != sid
        test_mask = subjects == sid

        model = train_one_fold(
            covs[train_mask], labels[train_mask], subjects[train_mask], n_classes
        )

        acc = evaluate_model(model, covs[test_mask], labels[test_mask])
        accs.append(acc)
        print(f"  Subject {sid}: {100 * acc:.2f}%")

    accs = np.array(accs)
    print("\nLOSO Mean =", 100 * accs.mean(), "±", 100 * accs.std())


# ============================================================
# MAIN
# ============================================================

def encode_subjects(subj_arr):
    unique = np.unique(subj_arr)
    mapping = {s: i for i, s in enumerate(unique)}
    return np.array([mapping[s] for s in subj_arr], dtype=np.int32)


def main():
    data = torch.load(CACHE_PATH, map_location="cpu")
    covs = np.stack([c.cpu().numpy() for c in data["ra_covs"]]).astype(np.float32)
    labels = np.array(data["labels"]).astype(int)

    subjects_raw = np.array(data["subj"])
    subjects = encode_subjects(subjects_raw)

    run_loso_experiment(covs, labels, subjects)


if __name__ == "__main__":
    main()
