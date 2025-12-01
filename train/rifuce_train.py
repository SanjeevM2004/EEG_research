"""
RiFuNet + Fisher Loss + TSLR-style Classifier + DANN
LOSO evaluation on BCI-IV 2a (bci_active4.pt)

- Backbone: RiFuNet (congruence-based SPD U-Net)
- Losses:
    * Fisher loss on tangent features (within / between action)
    * Cross-entropy for action labels
    * DANN cross-entropy for subject labels (via GRL)
    * Riemannian reconstruction loss between input and output SPD

- Features: TSLR-style tangent features at batch log-Euclidean mean
            (upper-tri of log(C_ref^{-1/2} C_out C_ref^{-1/2}))

Run:
    python -m train.rifunet_fisher_ce_dann
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils import clip_grad_norm_
from time import time

# ============================================================
# CONFIG
# ============================================================

CACHE_PATH = "./EEG_data/bci_active4.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RiFu / Loss hyperparams
STEPS = 1000         # training steps per LOSO fold
BATCH_SIZE = 256
LR = 1e-3

LAMBDA_FISHER = 1.0
LAMBDA_CE = 1.0
LAMBDA_DANN = 0.1
LAMBDA_RECON = 1e-3

LAMBDA_WITHIN = 1.0
LAMBDA_BETWEEN = 1.0

SEED = 42  # for reproducibility


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

    # add jitter to push away from singular
    C = C + jitter * eye

    # eigendecomp in double for stability
    w, V = torch.linalg.eigh(C.double())
    w = torch.clamp(w, min=eps_eig)
    logw = torch.log(w)
    S = V @ torch.diag_embed(logw) @ V.transpose(-1, -2)
    S = symmetrize(S)
    return S.float()


def expm_sym(S: torch.Tensor, eps_eig: float = 1e-5) -> torch.Tensor:
    """
    Matrix exponential for symmetric matrices.
    Returns SPD, eigenvalues clamped to >= eps_eig.
    """
    S = symmetrize(S.float())
    w, V = torch.linalg.eigh(S.double())
    expw = torch.exp(w)
    expw = torch.clamp(expw, min=eps_eig)
    C = V @ torch.diag_embed(expw) @ V.transpose(-1, -2)
    C = symmetrize(C)
    return C.float()


def sqrtm_spd(C: torch.Tensor, eps_eig: float = 1e-5) -> torch.Tensor:
    """
    SPD matrix square root.
    """
    C = symmetrize(C.float())
    w, V = torch.linalg.eigh(C.double())
    w = torch.clamp(w, min=eps_eig)
    sqrtw = torch.sqrt(w)
    C_sqrt = V @ torch.diag_embed(sqrtw) @ V.transpose(-1, -2)
    return symmetrize(C_sqrt.float())


def invsqrtm_spd(C: torch.Tensor, eps_eig: float = 1e-5) -> torch.Tensor:
    """
    SPD matrix inverse square root.
    """
    C = symmetrize(C.float())
    w, V = torch.linalg.eigh(C.double())
    w = torch.clamp(w, min=eps_eig)
    invsqrtw = torch.rsqrt(w)  # 1 / sqrt(w)
    C_isqrt = V @ torch.diag_embed(invsqrtw) @ V.transpose(-1, -2)
    return symmetrize(C_isqrt.float())


def tangent_upper_vec(S: torch.Tensor) -> torch.Tensor:
    """
    TSLR-style tangent features:
      - S: (B, d, d), symmetric (usually logm of aligned C_out)
      - returns: (B, d*(d+1)/2) upper-tri including diag
    """
    B, d, _ = S.shape
    idx = torch.triu_indices(d, d, device=S.device)
    feat = S[:, idx[0], idx[1]]  # (B, p)
    return feat


def riemannian_distance2(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
    """
    Squared AIRM distance between two batches of SPD matrices:
        d^2(C1, C2) = || log( C1^{-1/2} C2 C1^{-1/2} ) ||_F^2
    C1, C2: (B, d, d)
    Returns: (B,)
    """
    C1 = symmetrize(C1)
    C2 = symmetrize(C2)
    C1_isqrt = invsqrtm_spd(C1)
    # C_tilde = C1^{-1/2} C2 C1^{-1/2}
    C_tilde = C1_isqrt @ C2 @ C1_isqrt
    S = logm_spd(C_tilde)
    d2 = (S ** 2).sum(dim=(-1, -2))
    return d2


# ============================================================
# SPD LINEAR (congruence transform: W^T C W)
# ============================================================

class SPDLinear(nn.Module):
    """
    SPD -> SPD via congruence transform:
        C_out = W^T C_in W

    If C_in is SPD and W has non-zero columns, C_out is SPD (up to eps).
    Dimension changes from d_in -> d_out.
    """

    def __init__(self, d_in: int, d_out: int, eps: float = 1e-4):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.eps = eps

        # W: (d_in, d_out)
        init = torch.randn(d_in, d_out) * (1.0 / np.sqrt(d_in))
        self.W = nn.Parameter(init)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        C: (B, d_in, d_in)
        Returns: (B, d_out, d_out) SPD
        """
        C = symmetrize(C.float())         # (B, d_in, d_in)
        B, d_in, _ = C.shape
        W = self.W.float()                # (d_in, d_out)
        d_out = self.d_out

        # small jitter on input SPD to help eigens later
        eye_in = torch.eye(d_in, device=C.device, dtype=C.dtype).unsqueeze(0)
        C = C + self.eps * eye_in

        # CW = C @ W  → (B, d_in, d_out)
        CW = torch.matmul(C, W)

        # C_out = W^T C W  → (B, d_out, d_out)
        C_out = torch.matmul(CW.transpose(1, 2), W)   # (B, d_out, d_out)

        C_out = symmetrize(C_out)

        # Ensure strict SPD by bumping diagonal
        eye_out = torch.eye(d_out, device=C_out.device, dtype=C_out.dtype).unsqueeze(0)
        C_out = C_out + self.eps * eye_out

        return C_out


def merge_spd(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
    """
    U-Net style skip merge for SPD:
      - go to log domain (any SPD mean can be done in log-Euclidean way)
      - average
      - come back with expm
    """
    S1 = logm_spd(C1)
    S2 = logm_spd(C2)
    S = 0.5 * (S1 + S2)
    return expm_sym(S)


# ============================================================
# RiFuNet: U-Net in SPD Space via congruence transforms
# ============================================================

class RiFuNet(nn.Module):
    """
    Riemannian Fisher U-Net (congruence-based)
    -----------------------------------------
    - Input:  SPD covariances C_in (B, d, d)
    - Encoder path:
        C1 = SPDLinear(d -> d1)
        C2 = SPDLinear(d1 -> d2)
    - Bottleneck:
        C3 = SPDLinear(d2 -> d2)
    - Decoder path:
        U2 = SPDLinear(d2 -> d1); merge with C1
        U1 = SPDLinear(d1 -> d);  merge with C_in
    - Output: SPD C_out
    """

    def __init__(self, n_channels: int, d_mid1: int = None, d_mid2: int = None):
        super().__init__()
        d = n_channels
        if d_mid1 is None:
            d_mid1 = max(4, d // 2)          # first compression
        if d_mid2 is None:
            d_mid2 = max(2, d_mid1 // 2)     # deeper compression

        self.d_in = d
        self.d_mid1 = d_mid1
        self.d_mid2 = d_mid2

        # Encoder (congruence-based)
        self.enc1 = SPDLinear(d, d_mid1)
        self.enc2 = SPDLinear(d_mid1, d_mid2)

        # Bottleneck
        self.bottleneck = SPDLinear(d_mid2, d_mid2)

        # Decoder
        self.dec2 = SPDLinear(d_mid2, d_mid1)
        self.dec1 = SPDLinear(d_mid1, d)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        C: (B, d, d) SPD
        Returns:
            C_out: (B, d, d) SPD
        """
        C = symmetrize(C.float())
        B, d, d2 = C.shape
        assert d == d2, "Covariance must be square."

        # ---------- Encoder ----------
        C0 = C
        C1 = self.enc1(C0)   # (B, d_mid1, d_mid1)
        C2 = self.enc2(C1)   # (B, d_mid2, d_mid2)

        # ---------- Bottleneck ----------
        C3 = self.bottleneck(C2)  # (B, d_mid2, d_mid2)

        # ---------- Decoder ----------
        U2 = self.dec2(C3)               # (B, d_mid1, d_mid1)
        U2m = merge_spd(U2, C1)          # skip with encoder level 1

        U1 = self.dec1(U2m)              # (B, d, d)
        # Final merge with input for stability
        C_out = merge_spd(U1, C0)        # (B, d, d) SPD

        return C_out.float()


# ============================================================
# Fisher loss on tangent features + Riemannian reconstruction
# ============================================================

def fisher_feature_loss(
    feat: torch.Tensor,
    y_action: torch.Tensor,
    lambda_within: float = 1.0,
    lambda_between: float = 1.0,
) -> torch.Tensor:
    """
    Fisher-style loss on feature vectors:

        feat: (B, D)   - tangent features
        y_action: (B,) - class labels

        within  = average intra-class variance
        between = average distance of class means from global mean

        loss = λ_w * within - λ_b * between
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
        f = feat[idx]                     # (n_c, D)
        mu_c = f.mean(0, keepdim=True)
        n_c = f.shape[0]

        within += ((f - mu_c) ** 2).sum()
        between += n_c * ((mu_c - mu_global) ** 2).sum()
        count += n_c

    if count == 0:
        return torch.tensor(0.0, device=feat.device)

    within = within / count
    between = between / count

    loss = lambda_within * within - lambda_between * between
    return loss


def riemannian_reconstruction_loss(
    C_in: torch.Tensor,
    C_out: torch.Tensor,
    lambda_recon: float = 1e-3
) -> torch.Tensor:
    """
    Riemannian reconstruction loss between input and output SPD matrices.
    Uses squared AIRM distance, averaged over the batch:
        loss = λ_r * E[ d^2(C_in, C_out) ]
    """
    d2 = riemannian_distance2(C_in, C_out)  # (B,)
    return lambda_recon * d2.mean()


# ============================================================
# GRADIENT REVERSAL LAYER (for DANN)
# ============================================================

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradReverse.apply(x, self.alpha)


# ============================================================
# RiFuNet + TSLR + DANN CLASSIFIER
# ============================================================

class RiFuNetDANNClassifier(nn.Module):
    """
    End-to-end classifier:

        C_in (SPD) → RiFuNet (SPD U-Net) → C_out (SPD)
        → align at batch log-Euclidean mean:
              C_ref = exp( mean( log(C_out) ) )
              S_tan = log( C_ref^{-1/2} C_out C_ref^{-1/2} )
        → upper-tri vec feat (TSLR-style)
        → action head (softmax over actions)
        → GRL → subject head (softmax over subjects)

    Forward returns:
        logits_action, logits_subject, C_in, C_out, feat_tan
    """

    def __init__(self, n_channels: int, n_actions: int, n_subjects: int, grl_alpha: float = 1.0):
        super().__init__()
        self.backbone = RiFuNet(n_channels)
        self.n_actions = n_actions
        self.n_subjects = n_subjects

        d = n_channels
        self.d = d
        self.feat_dim = d * (d + 1) // 2  # upper-tri of (d x d)

        self.action_head = nn.Linear(self.feat_dim, n_actions)
        self.subject_head = nn.Linear(self.feat_dim, n_subjects)

        self.grl = GradientReversal(alpha=grl_alpha)

    def forward(self, C: torch.Tensor):
        """
        C: (B, d, d)
        Returns:
            logits_action:  (B, n_actions)
            logits_subject: (B, n_subjects)
            C_in:           (B, d, d)
            C_out:          (B, d, d)
            feat_tan:       (B, feat_dim)
        """
        C_in = symmetrize(C.float())
        C_out = self.backbone(C_in)           # SPD U-Net

        # -------- Tangent-space features at batch log-Euclidean mean --------
        # 1) Log of C_out
        S_all = logm_spd(C_out)              # (B, d, d)
        # 2) Log-Euclidean mean
        S_mean = S_all.mean(dim=0, keepdim=True)     # (1, d, d)
        C_ref = expm_sym(S_mean)                     # (1, d, d)
        # 3) Align to C_ref
        C_ref_isqrt = invsqrtm_spd(C_ref)            # (1, d, d)
        C_ref_isqrt = C_ref_isqrt.expand_as(C_out)   # (B, d, d)
        C_align = C_ref_isqrt @ C_out @ C_ref_isqrt  # (B, d, d)
        # 4) Log in tangent
        S_tan = logm_spd(C_align)                    # (B, d, d)
        feat = tangent_upper_vec(S_tan)              # (B, feat_dim)

        # Action logits
        logits_action = self.action_head(feat)       # (B, n_actions)

        # DANN subject logits with GRL
        feat_rev = self.grl(feat)
        logits_subject = self.subject_head(feat_rev)  # (B, n_subjects)

        return logits_action, logits_subject, C_in, C_out, feat


# ============================================================
# TRAINING + LOSO EVAL
# ============================================================

def prepare_covs(covs_np: np.ndarray) -> torch.Tensor:
    """
    Input: covs_np (N, d, d) in numpy
    Output: torch.Tensor (N, d, d) float32, symmetrized & trace-normalized
    """
    C = torch.from_numpy(covs_np).float()
    C = symmetrize(C)
    tr = C.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    C = C / (tr + 1e-6)
    return C


def train_one_fold(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    s_train_np: np.ndarray,
    n_actions: int,
    n_subjects: int,
) -> RiFuNetDANNClassifier:
    """
    Train RiFuNetDANNClassifier on one LOSO fold (train split only).
    Returns the trained model (best-loss weights already loaded).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    C_train = prepare_covs(X_train_np).to(DEVICE)
    ya = torch.tensor(y_train_np, dtype=torch.long, device=DEVICE)
    ys = torch.tensor(s_train_np, dtype=torch.long, device=DEVICE)

    N, d, _ = C_train.shape

    model = RiFuNetDANNClassifier(
        n_channels=d,
        n_actions=n_actions,
        n_subjects=n_subjects,
        grl_alpha=1.0
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    idx_all = torch.arange(N, device=DEVICE)

    model.train()
    best_loss = float("inf")
    best_state = None

    for step in range(1, STEPS + 1):
        batch_size = min(BATCH_SIZE, N)
        batch_idx = idx_all[torch.randint(0, N, (batch_size,), device=DEVICE)]

        C_b = C_train[batch_idx]
        ya_b = ya[batch_idx]
        ys_b = ys[batch_idx]

        logits_action, logits_subject, C_in_b, C_out_b, feat_b = model(C_b)

        # ---- Loss components ----
        fisher_loss = fisher_feature_loss(
            feat_b,
            ya_b,
            lambda_within=LAMBDA_WITHIN,
            lambda_between=LAMBDA_BETWEEN,
        )

        recon_loss = riemannian_reconstruction_loss(
            C_in_b,
            C_out_b,
            lambda_recon=LAMBDA_RECON,
        )

        ce_action = F.cross_entropy(logits_action, ya_b)
        ce_subject = F.cross_entropy(logits_subject, ys_b)

        total_loss = (
            LAMBDA_FISHER * fisher_loss +
            LAMBDA_CE * ce_action +
            LAMBDA_DANN * ce_subject +
            recon_loss
        )

        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {
                "model": model.state_dict(),
                "loss": best_loss,
                "step": step
            }

        if step == 1 or step % 100 == 0:
            print(
                f"[Fold train] step {step}/{STEPS} | "
                f"loss={total_loss.item():.6f} "
                f"(fisher={fisher_loss.item():.6f}, "
                f"ce_a={ce_action.item():.6f}, "
                f"ce_s={ce_subject.item():.6f}, "
                f"recon={recon_loss.item():.6f}) | best={best_loss:.6f}"
            )

    # Restore best weights
    if best_state is not None:
        print(f"\n*** Restoring best model (step={best_state['step']}, loss={best_state['loss']:.6f}) ***\n")
        model.load_state_dict(best_state["model"])

    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model: RiFuNetDANNClassifier, X_test_np: np.ndarray, y_test_np: np.ndarray) -> float:
    """
    Evaluate action classification accuracy on test set.
    """
    C_test = prepare_covs(X_test_np).to(DEVICE)
    y_test = torch.tensor(y_test_np, dtype=torch.long, device=DEVICE)

    logits_action, _, _, _, _ = model(C_test)
    preds = torch.argmax(logits_action, dim=-1)
    acc = (preds == y_test).float().mean().item()
    return acc


def run_loso_experiment(covs: np.ndarray, labels: np.ndarray, subjects: np.ndarray):
    """
    Perform LOSO cross-subject evaluation with RiFuNetDANNClassifier.
    """
    S_ids = np.unique(subjects)
    sid_to_int = {sid: i for i, sid in enumerate(S_ids)}
    s_int = np.vectorize(sid_to_int.get)(subjects)

    S = len(S_ids)
    n_classes = int(labels.max() + 1)

    print(f"N={covs.shape[0]}, channels={covs.shape[1]}, classes={n_classes}, subjects={S}\n")

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
        s_train = s_int[train_mask]

        X_test = covs[test_mask]
        y_test = labels[test_mask]
        s_test = s_int[test_mask]  # not used, but here if needed later

        model = train_one_fold(
            X_train, y_train, s_train,
            n_actions=n_classes,
            n_subjects=S
        )

        acc = evaluate_model(model, X_test, y_test)
        accs.append(acc)

        print(f"  ✅ Subject {sid}: Test Accuracy = {100 * acc:5.2f}%\n")

    elapsed = time() - t0
    accs = np.array(accs)

    print("\n" + "=" * 70)
    print("🧾 RiFuNet + Fisher + CE + DANN – LOSO SUMMARY")
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
    print(" RiFuNet + Fisher Loss + CE + DANN (TSLR-style Tangent Features)")
    print(f" Device: {DEVICE}")
    print("=" * 70)

    data = torch.load(CACHE_PATH, map_location="cpu")
    covs = np.stack([c.cpu().numpy() for c in data["ra_covs"]]).astype(np.float32)
    labels = np.asarray(data["labels"]).astype(int)
    subjects = np.asarray(data["subj"])

    run_loso_experiment(covs, labels, subjects)


if __name__ == "__main__":
    main()

#======================================================================
#🧾 RiFuNet + Fisher + CE + DANN – LOSO SUMMARY
#======================================================================
#      A01T : 69.44%
#      A02T : 28.47%
#      A03T : 78.12%
#      A04T : 44.10%
#      A05T : 43.06%
#      A06T : 39.93%
#      A07T : 52.08%
#      A08T : 77.78%
#      A09T : 65.97%
#----------------------------------------------------------------------
#→ LOSO Mean Accuracy = 55.44% ± 16.94%
#⏱️ Total Time: 764.40s
#======================================================================