# ==============================================================
#  models/riemann/dcr_subject_ra.py
#  DCRPreAligner_SubjectRA: Scale → Rotate → Subject-wise RA
#  (No global alignment; Fisher computed AFTER subject RA)
# ==============================================================

from __future__ import annotations
import math
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------- defaults --------
torch.set_default_dtype(torch.float64)

# ==============================================================
#  SPD utilities (robust)
# ==============================================================

def _sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))

def _eigh_sym(C: torch.Tensor):
    C = _sym(C)
    return torch.linalg.eigh(C)

def _project_spd(C: torch.Tensor, floor: float = 1e-10) -> torch.Tensor:
    """Nearest-SPD via eigen floor. Cheap & stable."""
    w, V = _eigh_sym(C)
    w = torch.clamp(w, min=floor)
    return V @ torch.diag_embed(w) @ V.transpose(-1, -2)

def _eigh_spd(C: torch.Tensor):
    Cp = _project_spd(C, floor=1e-10)
    w, V = torch.linalg.eigh(Cp)
    return torch.clamp(w, min=1e-10), V

def logm_spd(C: torch.Tensor) -> torch.Tensor:
    w, V = _eigh_spd(C)
    return V @ torch.diag_embed(torch.log(w)) @ V.transpose(-1, -2)

def expm_spd(C: torch.Tensor) -> torch.Tensor:
    w, V = _eigh_spd(C)
    return V @ torch.diag_embed(torch.exp(w)) @ V.transpose(-1, -2)

def invsqrtm_spd(C: torch.Tensor) -> torch.Tensor:
    w, V = _eigh_spd(C)
    return V @ torch.diag_embed(torch.rsqrt(w)) @ V.transpose(-1, -2)

# ==============================================================
#  Means
# ==============================================================

@torch.no_grad()
def _riemann_mean_spd_robust(C: torch.Tensor, max_iter: int = 30, tol: float = 1e-7) -> torch.Tensor:
    """
    AIRM/Karcher mean with SPD projection every step.
    Init with Log-Euclidean mean for safety.
    """
    # safe init: log-euclidean
    L = logm_spd(C)
    M = expm_spd(L.mean(0))
    for _ in range(max_iter):
        M = _project_spd(M, floor=1e-10)
        Minv2 = invsqrtm_spd(M)
        L = logm_spd(torch.einsum("ij,njk,kl->nil", Minv2, C, Minv2))
        Delta = L.mean(0)
        if torch.linalg.norm(Delta) < tol:
            break
        step = expm_spd(Delta)
        Mhalf = torch.linalg.cholesky(_project_spd(M, floor=1e-10))
        M = torch.einsum("ij,jk,kl->il", Mhalf, step, Mhalf)
    return _project_spd(M, floor=1e-10)

# ==============================================================
#  vech* (orthonormal half-vectorization)
# ==============================================================

class VechStar:
    def __init__(self, d: int, device=None):
        i, j = torch.tril_indices(d, d, device=device)
        self.i, self.j = i, j
        self.off_mask = (i != j)
        self.sqrt2 = math.sqrt(2.0)

    def __call__(self, L: torch.Tensor) -> torch.Tensor:
        v = L[:, self.i, self.j]
        if self.off_mask.any():
            v[:, self.off_mask] = v[:, self.off_mask] * self.sqrt2
        return v

# ==============================================================
#  LDA scatters
# ==============================================================

def _scatter_mats(F: torch.Tensor, y: torch.Tensor, eps: float = 1e-4, adaptive: bool = True):
    """
    Returns Sw, Sb with adaptive ridge (trace-normalized) for stability.
    """
    K = int(torch.max(y).item() + 1)
    G = torch.zeros((F.shape[0], K), dtype=F.dtype, device=F.device)
    G.scatter_(1, y.view(-1,1), 1.0)
    nk = G.sum(0).clamp_min(1.0)
    mu_k = (G.T @ F) / nk.unsqueeze(1)
    mu = (nk.unsqueeze(1) * mu_k).sum(0, keepdim=True) / nk.sum()

    F_center = F - G @ mu_k
    p = F.shape[1]
    lam = eps if not adaptive else eps * float(p) / max(1.0, F.shape[0])
    Sw = F_center.T @ F_center + lam * torch.eye(p, device=F.device, dtype=F.dtype)

    Mk = (mu_k - mu) * torch.sqrt(nk.unsqueeze(1))
    Sb = Mk.T @ Mk
    return Sw, Sb

# ==============================================================
#  Subject-wise RA helper (exported)
# ==============================================================

@torch.no_grad()
def riemann_align_subjectwise(C: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """
    Pure subject-wise RA to identity: for each subject s,
      M_s = Riemann mean(C_s) ; return  W_s C W_s  with W_s = M_s^{-1/2}.
    """
    Ct = torch.as_tensor(C, dtype=torch.float64)
    subj = np.asarray(subjects)
    uniq = np.unique(subj)
    out = torch.empty_like(Ct)
    for sid in uniq:
        idx = torch.as_tensor(np.where(subj == sid)[0])
        M_s = _riemann_mean_spd_robust(Ct[idx])
        Ws  = invsqrtm_spd(M_s)
        out[idx] = torch.einsum("ab,nbc,cd->nad", Ws, Ct[idx], Ws)
    return out.cpu().numpy()

# ==============================================================
#  DCRPreAligner_SubjectRA
# ==============================================================

class DCRPreAligner_SubjectRA(nn.Module):
    """
    Pipeline:
      Input C            (no global alignment)
      └─> Scale: C' = I + α (C - I)        [optional α learned]
          Rotate: C'' = R^T C' R           [R is orthogonal (Cayley or exp)]
          Subject RA (train/eval): for each subject s,
              W_s = M_s(C'')^{-1/2},  Ĉ = W_s C'' W_s
          Features: F = vech*( log(Ĉ) )
          Loss: maximize log(1 + tr(Sw^{-1} Sb))  (class-only)
          Reg:  ||R - I||_F^2 / d  (+ prior on α if enabled)
    """

    def __init__(self,
                 steps: int = 400,
                 lr: float = 1e-3,
                 eps: float = 1e-4,
                 gamma_class: float = 0.8,
                 beta_reg: float = 1e-4,
                 use_cayley: bool = True,
                 enable_dispersion_scale: bool = True,
                 dispersion_prior: float = 1e-3,
                 skew_clip: float = 0.5,
                 early_stop_patience: int = 40,
                 device: Optional[str] = None):
        super().__init__()
        self.steps, self.lr, self.eps = steps, lr, eps
        self.gamma_class, self.beta_reg = gamma_class, beta_reg
        self.use_cayley = use_cayley
        self.enable_dispersion_scale = enable_dispersion_scale
        self.dispersion_prior = dispersion_prior
        self.skew_clip = skew_clip
        self.early_stop_patience = early_stop_patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.A_param: Optional[nn.Parameter] = None
        self.theta_alpha: Optional[nn.Parameter] = None
        self.R: Optional[torch.Tensor] = None
        self.alpha_final: float = 1.0
        self.Ws_train: Dict[int, torch.Tensor] = {}  # cached train-subject whiteners

        self.to(self.device)

    # ----- rotation parametrization -----

    def _skew(self) -> torch.Tensor:
        S = self.A_param - self.A_param.mT
        with torch.no_grad():
            nrm = torch.linalg.norm(S)
            if nrm > self.skew_clip:
                self.A_param *= (self.skew_clip / (nrm + 1e-12))
        return self.A_param - self.A_param.mT

    def _R_from_A(self, d: int) -> torch.Tensor:
        S = self._skew()
        I = torch.eye(d, device=self.device, dtype=S.dtype)
        if self.use_cayley:
            return torch.linalg.solve(I - S, I + S)
        else:
            return torch.matrix_exp(S)

    @staticmethod
    def _congruence(C: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        At = A.mT
        return torch.einsum("ab,nbc,cd->nad", At, C, A)

    @staticmethod
    def _dispersion(C: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        I = torch.eye(C.shape[-1], device=C.device, dtype=C.dtype)
        return I + alpha * (C - I)

    # ----- fit / transform -----

    def fit(self, X, y, subjects, verbose: bool = True):
        assert subjects is not None, "subjects array is required for subject-wise RA."
        # SPD-harden inputs
        C = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        C = _project_spd(_sym(C), floor=1e-10)
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        N, d, _ = C.shape
        vstar = VechStar(d, device=self.device)

        # params
        self.A_param = nn.Parameter(torch.zeros((d, d), dtype=torch.float64, device=self.device))
        params = [self.A_param]
        if self.enable_dispersion_scale:
            init = math.log(math.exp(1.0 - 1e-6) - 1.0)  # softplus^-1(1.0) ≈ 1.0
            self.theta_alpha = nn.Parameter(torch.full((), init, dtype=torch.float64, device=self.device))
            params.append(self.theta_alpha)
        opt = optim.Adam(params, lr=self.lr)

        subj = np.asarray(subjects)
        best_obj, stagnant = -float("inf"), 0
        best_R, best_alpha = None, 1.0

        for step in range(self.steps):
            opt.zero_grad(set_to_none=True)

            R = self._R_from_A(d)
            # optional dispersion first (around I), then rotate
            if self.enable_dispersion_scale:
                alpha = torch.nn.functional.softplus(self.theta_alpha) + 1e-6
                C_scaled = self._dispersion(C, alpha)
            else:
                alpha = None
                C_scaled = C

            C_rot = self._congruence(C_scaled, R)

            # ===== subject-wise RA inside loss =====
            uniq = np.unique(subj)
            C_ra = torch.empty_like(C_rot)
            for sid in uniq:
                idx = torch.as_tensor(np.where(subj == sid)[0], device=self.device)
                M_s = _riemann_mean_spd_robust(C_rot[idx])
                Ws  = invsqrtm_spd(M_s)
                C_ra[idx] = torch.einsum("ab,nbc,cd->nad", Ws, C_rot[idx], Ws)

            # Tangent features AFTER RA
            F = vstar(logm_spd(C_ra))
            Sw, Sb = _scatter_mats(F, y, eps=self.eps, adaptive=True)
            L = torch.linalg.cholesky(Sw)
            fisher = torch.trace(torch.cholesky_solve(Sb, L))

            # objective to maximize; train by minimizing negative
            rot_reg = (torch.norm(R - torch.eye(d, device=self.device))**2) / d
            obj = torch.log1p(torch.clamp(fisher, min=0.0)) - self.beta_reg * rot_reg
            if alpha is not None:
                obj = obj - self.dispersion_prior * (alpha - 1.0).pow(2)

            loss = -self.gamma_class * obj
            loss.backward()
            opt.step()

            if verbose and (step % 50 == 0 or step == self.steps - 1):
                a_val = float(alpha.item()) if alpha is not None else 1.0
                print(f"[DCR-SubjRA] step={step:3d} | Fisher={fisher.item():.3e} | obj={obj.item():.3e} | alpha={a_val:.3f}")

            # early stopping on objective
            curr = float(obj.item())
            if curr > best_obj + 1e-9:
                best_obj, stagnant = curr, 0
                best_R = R.detach().clone()
                best_alpha = float(alpha.item()) if alpha is not None else 1.0
            else:
                stagnant += 1
                if stagnant >= self.early_stop_patience:
                    if verbose:
                        print(f"[DCR-SubjRA] Early stop at step {step}.")
                    break

        # finalize learned params
        self.R = best_R
        self.alpha_final = best_alpha

        # cache train-subject whiteners (for speed at inference)
        self.Ws_train = {}
        with torch.no_grad():
            if self.enable_dispersion_scale:
                C_scaled = self._dispersion(C, torch.tensor(self.alpha_final, device=self.device))
            else:
                C_scaled = C
            C_rot = self._congruence(C_scaled, self.R)
            uniq = np.unique(subj)
            for sid in uniq:
                idx = torch.as_tensor(np.where(subj == sid)[0], device=self.device)
                M_s = _riemann_mean_spd_robust(C_rot[idx])
                self.Ws_train[int(sid)] = invsqrtm_spd(M_s)

        return self

    @torch.no_grad()
    def transform(self, X, subjects):
        assert subjects is not None, "subjects array is required for subject-wise RA."
        # SPD-harden inputs
        C = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        C = _project_spd(_sym(C), floor=1e-10)

        # scale → rotate
        if self.enable_dispersion_scale:
            C = self._dispersion(C, torch.tensor(self.alpha_final, device=self.device))
        C_rot = self._congruence(C, self.R)

        # subject RA (use cached Ws for seen subjects; compute for unseen)
        subj = np.asarray(subjects)
        uniq = np.unique(subj)
        C_out = torch.empty_like(C_rot)
        for sid in uniq:
            idx_np = np.where(subj == sid)[0]
            idx = torch.as_tensor(idx_np, device=self.device)
            if int(sid) in self.Ws_train:
                Ws = self.Ws_train[int(sid)]
            else:
                M_s = _riemann_mean_spd_robust(C_rot[idx])
                Ws = invsqrtm_spd(M_s)
            C_out[idx] = torch.einsum("ab,nbc,cd->nad", Ws, C_rot[idx], Ws)

        return C_out.cpu().numpy()

    def fit_transform(self, X, y, subjects, verbose: bool = True):
        self.fit(X, y, subjects=subjects, verbose=verbose)
        return self.transform(X, subjects=subjects)
