"""
RiFuNet + Fisher Loss + CE + Subject Loss (no DANN)
TSLR-style Tangent Features at Batch Log-Euclidean Mean
LOSO evaluation on BCI-IV 2a (bci_active4.pt)

Backbone:
    - RiFuNet: congruence-based SPD U-Net with mild compression
      (for 22 ch: 22 -> 16 -> 12 -> 16 -> 22)

Losses:
    * Fisher loss on tangent features (within / between action)
    * Cross-entropy for action labels
    * Subject feature loss (within / between subjects)
    * Riemannian reconstruction loss between input and output SPD

Run:
    python -m train.rifunet_fisher_ce_subject
"""

from __future__ import annotations
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# ============================================================
# CONFIG
# ============================================================

CACHE_PATH = "./EEG_data/bci_active4.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STEPS = 1000
BATCH_SIZE = 256
LR = 1e-3

LAMBDA_FISHER = 1.0      # action Fisher loss
LAMBDA_CE = 1.0          # cross-entropy
LAMBDA_SUBJ = 0.05        # subject loss weight (global)
LAMBDA_RECON = 1e-3      # Riemannian reconstruction

LAMBDA_WITHIN = 1.0      # within-class/within-subject
LAMBDA_BETWEEN = 1.0     # between-class/between-subject

SEED = 50


# ============================================================
# SPD HELPERS
# ============================================================

def symmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.transpose(-1, -2))


def logm_spd(C: torch.Tensor, eps_eig: float = 1e-5, jitter: float = 1e-4) -> torch.Tensor:
    """
    Matrix logarithm for SPD matrices.
    - Enforces symmetry
    - Adds jitter*I before eigendecomp
    - Uses double precision for eigendecomp
    """
    C = symmetrize(C.float())
    B, d, _ = C.shape
    eye = torch.eye(d, device=C.device, dtype=C.dtype).unsqueeze(0)
    C = C + jitter * eye
    w, V = torch.linalg.eigh(C.double())
    w = torch.clamp(w, min=eps_eig)
    logw = torch.log(w)
    S = V @ torch.diag_embed(logw) @ V.transpose(-1, -2)
    return symmetrize(S.float())


def expm_sym(S: torch.Tensor, eps_eig: float = 1e-5) -> torch.Tensor:
    """
    Matrix exponential for symmetric matrices.
    """
    S = symmetrize(S.float())
    w, V = torch.linalg.eigh(S.double())
    expw = torch.exp(w)
    expw = torch.clamp(expw, min=eps_eig)
    C = V @ torch.diag_embed(expw) @ V.transpose(-1, -2)
    return symmetrize(C.float())


def invsqrtm_spd(C: torch.Tensor, eps_eig: float = 1e-5) -> torch.Tensor:
    """
    Inverse square-root of SPD matrix.
    """
    C = symmetrize(C.float())
    w, V = torch.linalg.eigh(C.double())
    w = torch.clamp(w, min=eps_eig)
    invsqrtw = torch.rsqrt(w)
    C_isqrt = V @ torch.diag_embed(invsqrtw) @ V.transpose(-1, -2)
    return symmetrize(C_isqrt.float())


def tangent_upper_vec(S: torch.Tensor) -> torch.Tensor:
    """
    Vectorize upper-triangular part of symmetric matrices.
    """
    B, d, _ = S.shape
    idx = torch.triu_indices(d, d, device=S.device)
    return S[:, idx[0], idx[1]]  # (B, d(d+1)/2)


def riemannian_distance2(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
    """
    Squared AIRM distance between pairs of SPD matrices.
    """
    C1 = symmetrize(C1)
    C2 = symmetrize(C2)
    C1_isqrt = invsqrtm_spd(C1)
    C_tilde = C1_isqrt @ C2 @ C1_isqrt
    S = logm_spd(C_tilde)
    return (S ** 2).sum(dim=(-1, -2))


# ============================================================
# SPD LINEAR
# ============================================================

class SPDLinear(nn.Module):
    """
    Congruence transform layer:
        C_out = W^T C_in W
    """

    def __init__(self, d_in: int, d_out: int, eps: float = 1e-4):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.eps = eps
        init = torch.randn(d_in, d_out) * (1.0 / np.sqrt(d_in))
        self.W = nn.Parameter(init)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        C = symmetrize(C.float())
        B, d_in, _ = C.shape
        W = self.W.float()
        eye_in = torch.eye(d_in, device=C.device, dtype=C.dtype).unsqueeze(0)
        C = C + self.eps * eye_in      # keep SPD
        CW = C @ W                     # (B, d_in, d_out)
        C_out = CW.transpose(1, 2) @ W # (B, d_out, d_out)
        C_out = symmetrize(C_out)
        eye_out = torch.eye(self.d_out, device=C_out.device, dtype=C_out.dtype).unsqueeze(0)
        C_out = C_out + self.eps * eye_out
        return C_out


def merge_spd(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
    """
    Geometric "average" of two SPD matrices via log-Euclidean midpoint.
    """
    S1 = logm_spd(C1)
    S2 = logm_spd(C2)
    S = 0.5 * (S1 + S2)
    return expm_sym(S)


# ============================================================
# RiFuNet (mild compression U-Net)
# ============================================================

class RiFuNet(nn.Module):
    """
    For d=22: 22 -> 16 -> 12 -> 16 -> 22
    In general:
        d1 = min(16, d)
        d2 = min(12, d1)
    """

    def __init__(self, n_channels: int, d_mid1: int | None = None, d_mid2: int | None = None):
        super().__init__()
        d = n_channels
        if d_mid1 is None:
            d_mid1 = min(16, d)
        if d_mid2 is None:
            d_mid2 = min(12, d_mid1)

        self.enc1 = SPDLinear(d, d_mid1)
        self.enc2 = SPDLinear(d_mid1, d_mid2)
        self.bottleneck = SPDLinear(d_mid2, d_mid2)
        self.dec2 = SPDLinear(d_mid2, d_mid1)
        self.dec1 = SPDLinear(d_mid1, d)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        C = symmetrize(C.float())
        C0 = C
        C1 = self.enc1(C0)
        C2 = self.enc2(C1)
        C3 = self.bottleneck(C2)
        U2 = self.dec2(C3)
        U2m = merge_spd(U2, C1)   # skip connection
        U1 = self.dec1(U2m)
        C_out = merge_spd(U1, C0) # final skip
        return C_out.float()


# ============================================================
# LOSSES
# ============================================================

def fisher_feature_loss(
    feat: torch.Tensor,
    y_action: torch.Tensor,
    lambda_within: float = 1.0,
    lambda_between: float = 1.0,
) -> torch.Tensor:
    """
    Fisher-style loss on action classes in feature space.
    """
    B, D = feat.shape
    mu_global = feat.mean(0, keepdim=True)
    within = 0.0
    between = 0.0
    count = 0

    classes = torch.unique(y_action)
    for c in classes:
        idx = (y_action == c)
        if idx.sum() <= 1:
            continue
        f = feat[idx]
        mu_c = f.mean(0, keepdim=True)
        n_c = f.shape[0]
        within += ((f - mu_c) ** 2).sum()
        between += n_c * ((mu_c - mu_global) ** 2).sum()
        count += n_c

    if count == 0:
        return torch.tensor(0.0, device=feat.device)

    within = within / (count * D)
    between = between / (count * D)
    return lambda_within * within - lambda_between * between


def subject_feature_loss(
    feat: torch.Tensor,
    subjects: torch.Tensor,
    lambda_within: float = 1.0,
    lambda_between: float = 1.0,
) -> torch.Tensor:
    """
    Subject alignment loss (Mini-DCR style variant)

    - μ_s   = mean feature per subject
    - μ_g   = mean of {μ_s} over subjects (NOT mean of all features)
    - within:  sum_s sum_{i in s} ||f_i - μ_s||^2
    - between: sum_s n_s ||μ_s - μ_g||^2

    Returns: λ_within * within - λ_between * between (normalized).
    """
    B, D = feat.shape
    subj_ids = torch.unique(subjects)

    mu_list = []
    n_list = []

    within = 0.0
    between = 0.0

    # Compute subject means and within-subject spread
    for s in subj_ids:
        idx = (subjects == s)
        n_s = idx.sum().item()
        if n_s <= 1:
            continue
        f_s = feat[idx]                        # (n_s, D)
        mu_s = f_s.mean(0, keepdim=True)       # (1, D)
        mu_list.append(mu_s)
        n_list.append(n_s)
        within += ((f_s - mu_s) ** 2).sum()

    if len(mu_list) == 0:
        return torch.tensor(0.0, device=feat.device)

    mu_stack = torch.cat(mu_list, dim=0)       # (#subjects, D)
    n_stack = torch.tensor(n_list, device=feat.device, dtype=torch.float32)

    # Global mean over subject means
    mu_global = mu_stack.mean(0)              # (D,)

    # Between-subject spread
    for mu_s, n_s in zip(mu_stack, n_stack):
        between += n_s * ((mu_s - mu_global) ** 2).sum()

    within = within / (B * D)
    between = between / (B * D)

    return lambda_within * within - lambda_between * between


def riemannian_reconstruction_loss(
    C_in: torch.Tensor,
    C_out: torch.Tensor,
    lambda_recon: float = 1e-3
) -> torch.Tensor:
    """
    Reconstruction loss in AIRM on SPD manifold.
    """
    d2 = riemannian_distance2(C_in, C_out)
    return lambda_recon * d2.mean()


# ============================================================
# RiFuNet CLASSIFIER
# ============================================================

class RiFuNetClassifier(nn.Module):
    """
    C_in -> RiFuNet -> C_out -> batch log-Euclidean mean alignment
         -> tangent log -> upper-tri vec -> Linear head
    """

    def __init__(self, n_channels: int, n_actions: int):
        super().__init__()
        self.backbone = RiFuNet(n_channels)
        d = n_channels
        self.feat_dim = d * (d + 1) // 2
        self.action_head = nn.Linear(self.feat_dim, n_actions)

    def forward(self, C: torch.Tensor):
        C_in = symmetrize(C.float())
        C_out = self.backbone(C_in)

        # Log of output covariances
        S_all = logm_spd(C_out)                 # (B, d, d)

        # Batch log-Euclidean mean as reference
        S_mean = S_all.mean(dim=0, keepdim=True)  # (1, d, d)
        C_ref = expm_sym(S_mean)                  # (1, d, d)
        C_ref_isqrt = invsqrtm_spd(C_ref)         # (1, d, d)
        C_ref_isqrt = C_ref_isqrt.expand_as(C_out)

        # Align to reference and take tangent log
        C_align = C_ref_isqrt @ C_out @ C_ref_isqrt
        S_tan = logm_spd(C_align)               # (B, d, d)

        # Upper-tri vectorization
        feat = tangent_upper_vec(S_tan)         # (B, feat_dim)

        logits_action = self.action_head(feat)
        return logits_action, C_in, C_out, feat


# ============================================================
# TRAINING + LOSO
# ============================================================

def prepare_covs(covs_np: np.ndarray) -> torch.Tensor:
    """
    Convert numpy covariances to normalized torch tensor.
    """
    C = torch.from_numpy(covs_np).float()
    C = symmetrize(C)
    tr = C.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    C = C / (tr + 1e-6)
    return C


def train_one_fold(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    subj_train_np: np.ndarray,
    n_actions: int,
) -> RiFuNetClassifier:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    C_train = prepare_covs(X_train_np).to(DEVICE)
    y = torch.tensor(y_train_np, dtype=torch.long, device=DEVICE)
    subjects_train = torch.tensor(subj_train_np, dtype=torch.long, device=DEVICE)

    N, d, _ = C_train.shape

    model = RiFuNetClassifier(n_channels=d, n_actions=n_actions).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    idx_all = torch.arange(N, device=DEVICE)

    best_loss = float("inf")
    best_state = None

    model.train()
    for step in range(1, STEPS + 1):
        batch_size = min(BATCH_SIZE, N)
        batch_idx = idx_all[torch.randint(0, N, (batch_size,), device=DEVICE)]
        C_b = C_train[batch_idx]
        y_b = y[batch_idx]
        subj_b = subjects_train[batch_idx]

        logits, C_in_b, C_out_b, feat_b = model(C_b)

        ce_action = F.cross_entropy(logits, y_b)

        fisher_loss = fisher_feature_loss(
            feat_b, y_b,
            lambda_within=LAMBDA_WITHIN,
            lambda_between=LAMBDA_BETWEEN,
        )

        subj_loss = subject_feature_loss(
            feat_b, subj_b,
            lambda_within=LAMBDA_WITHIN,
            lambda_between=LAMBDA_BETWEEN,
        )

        recon_loss = riemannian_reconstruction_loss(
            C_in_b, C_out_b,
            lambda_recon=LAMBDA_RECON,
        )

        total_loss = (
            LAMBDA_CE * ce_action +
            LAMBDA_FISHER * fisher_loss +
            LAMBDA_SUBJ * subj_loss +
            recon_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {
                "model": model.state_dict(),
                "loss": best_loss,
                "step": step,
            }

        if step == 1 or step % 100 == 0:
            print(
                f"[Fold train] step {step}/{STEPS} | "
                f"loss={total_loss.item():.6f} "
                f"(ce={ce_action.item():.6f}, "
                f"fisher={fisher_loss.item():.6f}, "
                f"subj={subj_loss.item():.6f}, "
                f"recon={recon_loss.item():.6f}) | best={best_loss:.6f}"
            )

    if best_state is not None:
        print(
            f"\n*** Restoring best model "
            f"(step={best_state['step']}, loss={best_state['loss']:.6f}) ***\n"
        )
        model.load_state_dict(best_state["model"])

    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model: RiFuNetClassifier, X_test_np: np.ndarray, y_test_np: np.ndarray) -> float:
    C_test = prepare_covs(X_test_np).to(DEVICE)
    y_test = torch.tensor(y_test_np, dtype=torch.long, device=DEVICE)
    logits, _, _, _ = model(C_test)
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == y_test).float().mean().item()
    return acc


def run_loso_experiment(covs: np.ndarray, labels: np.ndarray, subjects: np.ndarray):
    S_ids = np.unique(subjects)
    n_classes = int(labels.max() + 1)

    print(
        f"N={covs.shape[0]}, channels={covs.shape[1]}, "
        f"classes={n_classes}, subjects={len(S_ids)}\n"
    )

    accs = []
    t0 = time()

    for sid in S_ids:
        print("=" * 70)
        print(f"🧠 LOSO Fold – Test Subject: {sid}")
        print("=" * 70)

        train_mask = (subjects != sid)
        test_mask = (subjects == sid)

        X_train = covs[train_mask]
        y_train = labels[train_mask]
        subj_train = subjects[train_mask]

        X_test = covs[test_mask]
        y_test = labels[test_mask]

        model = train_one_fold(
            X_train, y_train, subj_train,
            n_actions=n_classes,
        )

        acc = evaluate_model(model, X_test, y_test)
        accs.append(acc)
        print(f"  ✅ Subject {sid}: Test Accuracy = {100 * acc:5.2f}%\n")

    elapsed = time() - t0
    accs = np.array(accs)

    print("\n" + "=" * 70)
    print("🧾 RiFuNet + Fisher + CE + Subject Loss – LOSO SUMMARY")
    print("=" * 70)
    for sid, a in zip(S_ids, accs):
        print(f"    {sid:>6} : {100 * a:5.2f}%")
    print("-" * 70)
    print(f"→ LOSO Mean Accuracy = {100 * accs.mean():.2f}% ± {100 * accs.std():.2f}%")
    print(f"⏱️ Total Time: {elapsed:.2f}s")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print(" RiFuNet + Fisher Loss + CE + Subject Loss")
    print(f" Device: {DEVICE}")
    print("=" * 70)

    data = torch.load(CACHE_PATH, map_location="cpu")
    covs = np.stack([c.cpu().numpy() for c in data["ra_covs"]]).astype(np.float32)
    labels = np.asarray(data["labels"]).astype(int)
    subjects = np.asarray(data["subj"])
    # Convert subject strings (like 'A01T') into integer IDs
    unique_subj = np.unique(subjects)
    subj_map = {s: i for i, s in enumerate(unique_subj)}
    subjects_int = np.array([subj_map[s] for s in subjects], dtype=np.int64)
    run_loso_experiment(covs, labels, subjects_int)


if __name__ == "__main__":
    main()

#======================================================================
#🧾 RiFuNet + Fisher + CE (no DANN) – LOSO SUMMARY
#======================================================================
#      A01T : 67.36%
#      A02T : 27.08%
#      A03T : 80.21%
#      A04T : 44.79%
#      A05T : 45.49%
#      A06T : 41.67%
#      A07T : 50.00%
#      A08T : 76.74%
#      A09T : 62.85%
#----------------------------------------------------------------------
#→ LOSO Mean Accuracy = 55.13% ± 16.66%
#⏱️ Total Time: 760.34s
#======================================================================