from __future__ import annotations
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # More stable init than pure random
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
      - go to log domain
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
    - Output: SPD C_out, plus log-domain S_in, S_out for loss.
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

    def forward(self, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        C: (B, d, d) SPD
        Returns:
            C_out: (B, d, d) SPD
            S_out: (B, d, d) log-domain of output
            S_in : (B, d, d) log-domain of input
        """
        C = symmetrize(C.float())
        B, d, d2 = C.shape
        assert d == d2, "Covariance must be square."

        # Log of input (for reconstruction + Fisher)
        S_in = logm_spd(C)

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

        S_out = logm_spd(C_out)

        return C_out.float(), S_out.float(), S_in.float()


# ============================================================
# Fisher + Reconstruction Loss
# ============================================================

def fisher_rifunet_loss(
    S_in: torch.Tensor,
    S_out: torch.Tensor,
    y_action: torch.Tensor,
    y_subject: torch.Tensor,
    lambda_within: float = 1.0,
    lambda_between: float = 1.0,
    lambda_subj_var: float = 0.1,
    lambda_recon: float = 1e-3,
) -> torch.Tensor:
    """
    Fisher-style loss in log-domain + reconstruction:

        Loss =
            λ_w * within_action
          - λ_b * between_action
          - λ_s * within_subject_variance
          + λ_r * ||S_out - S_in||^2
    """
    B = S_out.shape[0]

    feat = S_out.reshape(B, -1)
    feat_in = S_in.reshape(B, -1)

    mu_global = feat.mean(0, keepdim=True)

    # -------- within & between action --------
    within = 0.0
    between = 0.0
    count = 0

    for a in torch.unique(y_action):
        idx = (y_action == a)
        if idx.sum() <= 1:
            continue
        f = feat[idx]
        mu = f.mean(0, keepdim=True)
        n = f.shape[0]

        within += ((f - mu) ** 2).sum()
        between += n * ((mu - mu_global) ** 2).sum()
        count += n

    within = within / max(count, 1)
    between = between / max(count, 1)

    # -------- maximize subject variance --------
    subj_var = 0.0
    scount = 0
    for s in torch.unique(y_subject):
        idx = (y_subject == s)
        if idx.sum() <= 1:
            continue
        f = feat[idx]
        mu = f.mean(0, keepdim=True)
        subj_var += ((f - mu) ** 2).sum()
        scount += f.shape[0]

    subj_var = subj_var / max(scount, 1)

    # -------- reconstruction term --------
    recon = F.mse_loss(feat, feat_in)

    loss = (
        lambda_within * within
        - lambda_between * between
        - lambda_subj_var * subj_var
        + lambda_recon * recon
    )

    return loss


# ============================================================
# RiFuNetPreAligner (DCR-style wrapper, same interface)
# ============================================================

class RiFuNetPreAligner:
    """
    Riemannian Fisher U-Net Pre-Aligner (congruence-based)

    Usage:
        pre = RiFuNetPreAligner(n_channels=d, n_actions=4, n_subjects=9, device="cuda")
        pre.fit(covs, y_action, y_subject)
        covs_out = pre.transform(covs_new)
    """

    def __init__(
        self,
        n_channels: int,
        n_actions: int,
        n_subjects: int,
        base_ch: int = 16,          # kept for compatibility (unused now)
        steps: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        lambda_within: float = 1.0,
        lambda_between: float = 1.0,
        lambda_subj_var: float = 0.1,
        lambda_recon: float = 1e-3,
        device: str = "cpu",
    ):
        self.n_channels = n_channels
        self.n_actions = n_actions
        self.n_subjects = n_subjects
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_within = lambda_within
        self.lambda_between = lambda_between
        self.lambda_subj_var = lambda_subj_var
        self.lambda_recon = lambda_recon
        self.device = device

        # congruence-based U-Net
        self.model = RiFuNet(n_channels=n_channels).to(device)
        self._fitted = False

    # ------------------ TRAIN ------------------
    def fit(self, covs, y_action, y_subject, verbose: bool = True):

        # FORCE FLOAT32
        if isinstance(covs, np.ndarray):
            C = torch.from_numpy(covs).float()
        else:
            C = covs.float()

        # normalize by trace
        C = symmetrize(C)
        tr = C.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        C = C / (tr + 1e-6)
        C = C.to(self.device)

        ya = torch.tensor(y_action, dtype=torch.long, device=self.device)
        ys = torch.tensor(y_subject, dtype=torch.long, device=self.device)

        N = C.shape[0]
        idx_all = torch.arange(N, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()

        # =====================================================
        # ⭐ BEST LOSS TRACKING
        # =====================================================
        best_loss = float("inf")
        best_state = None

        for step in range(1, self.steps + 1):
            batch_size = min(self.batch_size, N)
            batch_idx = idx_all[torch.randint(0, N, (batch_size,), device=self.device)]

            C_b = C[batch_idx]
            ya_b = ya[batch_idx]
            ys_b = ys[batch_idx]

            C_out, S_out, S_in = self.model(C_b)

            loss = fisher_rifunet_loss(
                S_in,
                S_out,
                ya_b,
                ys_b,
                lambda_within=self.lambda_within,
                lambda_between=self.lambda_between,
                lambda_subj_var=self.lambda_subj_var,
                lambda_recon=self.lambda_recon,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # =====================================================
            # ⭐ SAVE BEST MODEL PARAMETERS
            # =====================================================
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {
                    "model": self.model.state_dict(),
                    "step": step,
                    "loss": best_loss
                }

            # =====================================================
            # LOG
            # =====================================================
            if verbose and (step % 100 == 0 or step == 1):
                print(f"[RiFuNet step {step}/{self.steps}] Loss = {loss.item():.6f} | Best = {best_loss:.6f}")

        # =====================================================
        # ⭐ RESTORE BEST MODEL
        # =====================================================
        print(f"\n*** Restoring best model (step={best_state['step']}, loss={best_state['loss']:.6f}) ***\n")
        self.model.load_state_dict(best_state["model"])

        self._fitted = True
        self.model.eval()


    # ------------------ TRANSFORM ------------------
    @torch.no_grad()
    def transform(self, covs):
        """
        Apply trained pre-aligner.
        Input:
            covs: (N, d, d) SPD (np.ndarray or torch.Tensor)
        Output:
            covs_out: (N, d, d) SPD (torch.Tensor on self.device)
        """
        if not self._fitted:
            raise RuntimeError("RiFuNetPreAligner not fitted. Call fit() first.")

        if isinstance(covs, np.ndarray):
            C = torch.from_numpy(covs).float()
        else:
            C = covs.float()

        # match the normalization used in fit
        C = symmetrize(C)
        tr = C.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        C = C / (tr + 1e-6)

        C_out, _, _ = self.model(C.to(self.device))
        return C_out


#======================================================================
#🧾 COMPARISON SUMMARY (Mean ± Std)
#======================================================================
#TS-SVM-RBF   | RA:  54.98% ± 16.06% | RiFu:  55.17% ± 15.46%
#TSLR         | RA:  52.89% ± 14.48% | RiFu:  54.67% ± 15.60%
#MDM          | RA:  52.43% ± 15.66% | RiFu:  52.89% ± 15.41%
#TSA-LDA      | RA:  53.51% ± 15.29% | RiFu:  53.36% ± 15.50%
#======================================================================
#Runtime → RA: 56.39s | RiFu+RA: 295.96s
#======================================================================

#======================================================================
#🧾 COMPARISON SUMMARY (Mean ± Std)
#======================================================================
#TS-SVM-RBF   | RA:  54.98% ± 16.06% | RiFu:  54.71% ± 16.18%
#TSLR         | RA:  52.89% ± 14.48% | RiFu:  54.78% ± 15.43%
#MDM          | RA:  52.43% ± 15.66% | RiFu:  52.51% ± 16.21%
#TSA-LDA      | RA:  53.51% ± 15.29% | RiFu:  54.24% ± 15.81%
#======================================================================
#Runtime → RA: 62.56s | RiFu+RA: 1276.57s
#======================================================================