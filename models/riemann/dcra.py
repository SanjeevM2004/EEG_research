# models/riemann/dcr_prealign.py
# ---------------------------------------------------------------
# DCRPreAligner: Discriminative Class Rotation Preprocessor
# ---------------------------------------------------------------
# Input : RA covariances (N, d, d)
# Output: Rotated covariances (N, d, d) via C_out = Rᵀ C_in R
# Loss  : L = γ * W(R)/(B(R)+eps) + α(λ-1)² + (β_t/d) * ||R - I||_F² + γ_c * center_loss
# Notes : R is parameterized geodesically as R = exp(A - Aᵀ) (no retraction needed)
# ---------------------------------------------------------------

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)

# ----------------------------- SPD utils -----------------------------
def _sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))

def _eigh_spd(C: torch.Tensor):
    w, V = torch.linalg.eigh(_sym(C))
    return torch.clamp(w, min=1e-10), V

def logm_spd(C: torch.Tensor) -> torch.Tensor:
    w, V = _eigh_spd(C)
    return V @ torch.diag_embed(torch.log(w)) @ V.transpose(-1, -2)

def invsqrtm_spd(C: torch.Tensor) -> torch.Tensor:
    w, V = _eigh_spd(C)
    w_inv_sqrt = torch.rsqrt(w)
    return V @ torch.diag_embed(w_inv_sqrt) @ V.transpose(-1, -2)

def offdiag(M: torch.Tensor) -> torch.Tensor:
    d = M.shape[-1]
    mask = torch.ones(d, d, device=M.device, dtype=M.dtype) - torch.eye(d, device=M.device, dtype=M.dtype)
    return M * mask

def diag_vec(M: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(M, dim1=-2, dim2=-1)

# ------------------------- DCR Pre-Aligner ---------------------------
class DCRPreAligner(nn.Module):
    """
    Learn R ∈ SO(d) that minimizes inverse Fisher ratio with stabilizers.
    Then apply rotation: C_out = Rᵀ C_in R.
    """

    def __init__(self,
                 steps: int = 600,
                 lr: float = 1e-3,
                 alpha: float = 1e-5,        # (λ-1)^2 weight
                 beta: float = 3e-4,         # ||R - I||_F^2 weight (normalized by d)
                 gamma: float = 1.0,         # Fisher inverse weight
                 center_gamma: float = 1e-4, # centering loss weight
                 eps: float = 1e-12,
                 device: str | None = None,
                 normalize_log: bool = True,     # z-score L before loss
                 class_balance_within: bool = True,
                 rewhiten_after: bool = True,     # re-whiten after rotation in transform()
                 clip_grad_norm: float | None = 5.0,
                 verbose_every: int = 60):
        super().__init__()
        self.steps = steps
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.center_gamma = center_gamma
        self.eps = eps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize_log = normalize_log
        self.class_balance_within = class_balance_within
        self.rewhiten_after = rewhiten_after
        self.clip_grad_norm = clip_grad_norm
        self.verbose_every = max(1, int(verbose_every))

        # Parameters (lazy-init A on first fit when d known)
        self.A_param: nn.Parameter | None = None   # skew generator → R = exp(A - Aᵀ)
        self._lambda_raw = nn.Parameter(torch.zeros((), dtype=torch.float64, device=self.device))

        # Fitted artifacts
        self.R: torch.Tensor | None = None
        self.lambda_disp: float | None = None
        self.I_d: torch.Tensor | None = None

        self.to(self.device)

    # --------- Rotation from skew-symmetric parameter (geodesic) ---------
    def _R_from_A(self) -> torch.Tensor:
        # R = exp(S), with S = A - Aᵀ skew-symmetric
        S = self.A_param - self.A_param.transpose(0, 1)
        return torch.matrix_exp(S)

    def _lambda_pos(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._lambda_raw) + 1e-6

    # -------------------- Fisher loss pieces (diag/offdiag) --------------------
    def _fisher_loss(self, L: torch.Tensor, y: torch.Tensor):
        """
        Returns:
          fisher_inv (scalar), between (scalar), within (scalar), center_penalty (scalar)
        - Between: sum_k n_k ||diag(Rᵀ (M_k - M) R)||_F^2
        - Within : class-balanced sum of ||offdiag(Rᵀ (L_i - M_{y_i}) R)||_F^2
        - Center : ||offdiag(mean_n Rᵀ L_n R)||_F^2 (keep rotated mean near diagonal/zero)
        """
        R = self._R_from_A()
        Rt = R.transpose(0, 1)
        classes = torch.unique(y)

        # class means (in log domain)
        M_k, n_k = [], []
        for c in classes:
            idx = (y == c)
            M_k.append(L[idx].mean(dim=0))
            n_k.append(int(idx.sum().item()))
        M_k = torch.stack(M_k, dim=0)                       # (K,d,d)
        n_k_t = torch.tensor(n_k, device=L.device, dtype=L.dtype)
        M = (M_k * (n_k_t / n_k_t.sum())[:, None, None]).sum(dim=0)

        # Between (diagonal energy)
        M_center = M_k - M                                   # (K,d,d)
        Bk = torch.einsum('ab,kbc,cd->kad', Rt, M_center, R) # K × d × d
        between = (diag_vec(Bk) ** 2).sum(dim=1)             # (K,)
        between = (between * n_k_t).sum()                    # scalar

        # Map each sample to its class mean
        Mk_map = {}
        for i, c in enumerate(classes):
            c_int = int(c.item()) if hasattr(c, "item") else int(c)
            Mk_map[c_int] = M_k[i]
        y_list = y.tolist() if hasattr(y, "tolist") else list(y)
        Mk_stack = torch.stack([Mk_map[int(ci)] for ci in y_list], dim=0)

        # Within (off-diagonal energy)
        Wi = L - Mk_stack                                    # (N,d,d)
        Wrot = torch.einsum('ab,nbc,cd->nad', Rt, Wi, R)
        e_i = (offdiag(Wrot) ** 2).sum(dim=(-1, -2))         # (N,)

        if self.class_balance_within:
            within = 0.0 * between
            # accumulate per class, normalized by class size
            # build index lists
            for i, c in enumerate(classes):
                cls = int(c.item()) if hasattr(c, "item") else int(c)
                idx = (torch.as_tensor(y_list, device=L.device) == cls).nonzero(as_tuple=False).flatten()
                nci = max(int(idx.numel()), 1)
                within = within + e_i[idx].sum() / (nci + 1e-9)
        else:
            within = e_i.sum()

        # Centering: keep rotated mean log close to (near-)diagonal zero
        Lrot_mean = torch.einsum('ab,nbc,cd->ad', Rt, L, R).mean(dim=0)
        center_penalty = torch.norm(offdiag(Lrot_mean)) ** 2

        fisher_inv = within / (between + self.eps)
        return fisher_inv, between, within, center_penalty

    # ------------------------------- Fit --------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        X_t = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        L = logm_spd(X_t)

        # optional z-score normalization in log domain (per element)
        if self.normalize_log:
            L_mean = L.mean(dim=0, keepdim=True)
            L_std = L.std(dim=0, keepdim=True) + 1e-6
            L = (L - L_mean) / L_std

        # infer d and lazily init parameters
        d = X_t.shape[-1]
        if self.A_param is None:
            self.A_param = nn.Parameter(torch.zeros((d, d), dtype=torch.float64, device=self.device))
            self.I_d = torch.eye(d, dtype=torch.float64, device=self.device)
            self.to(self.device)

        opt = optim.Adam([self.A_param, self._lambda_raw], lr=self.lr)

        for step in range(self.steps):
            opt.zero_grad(set_to_none=True)

            fisher_inv, B, W, center_penalty = self._fisher_loss(L, y_t)
            lam = self._lambda_pos()

            # cosine anneal β across steps (more flexible early, tighter late)
            beta_t = self.beta * (0.5 * (1.0 + math.cos(math.pi * step / max(1, self.steps-1))))

            R = self._R_from_A()
            # main loss
            loss = (
                self.gamma * fisher_inv
                + self.alpha * (lam - 1.0) ** 2
                + (beta_t * torch.norm(R - self.I_d) ** 2) / d
                + self.center_gamma * center_penalty
            )

            loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_([self.A_param, self._lambda_raw], self.clip_grad_norm)
            opt.step()

            if verbose and (step % self.verbose_every == 0 or step == self.steps - 1):
                print(f"[DCRPreAligner] step {step:4d} | loss={loss.item():.6e} | "
                      f"1/J={fisher_inv.item():.3e} | center={center_penalty.item():.3e}")

        with torch.no_grad():
            self.R = self._R_from_A().detach().clone()
            self.lambda_disp = float(self._lambda_pos().item())

        return self

    # ----------------------------- Transform -----------------------------
    @torch.no_grad()
    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.R is not None, "Call fit() before transform()."
        X_t = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        Rt = self.R.transpose(0, 1)
        X_rot = torch.einsum('ab,nbc,cd->nad', Rt, X_t, self.R)

        if self.rewhiten_after:
            # keep RA centering tight by mean re-whitening in the rotated space
            C_mean = X_rot.mean(dim=0)
            Wm = invsqrtm_spd(C_mean)
            X_rot = torch.einsum('ab,nbc,cd->nad', Wm, X_rot, Wm)

        return X_rot.cpu().numpy()

    def fit_transform(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> np.ndarray:
        self.fit(X, y, verbose=verbose)
        return self.transform(X)
