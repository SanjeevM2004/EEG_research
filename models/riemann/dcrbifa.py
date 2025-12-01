# models/riemann/dcr_dual_prealign.py
# =====================================================================
# DCRPreAlignerDualFast: Dual-Fisher optimizer with log-space rotation
#   • Maximize class Fisher:   minimize  Wc / (Bc + eps)
#   • Minimize subject Fisher: minimize  Bs / (Ws + eps)
#   • Vectorized across classes & subjects (no Python loops per step)
#   • Rotation happens in LOG space; transform() maps back with EXP
#   • NEW: Dispersion scaling after RA, before rotation (log-space)
#       - per_subject (default): scale each subject by its σ_s
#       - global: single σ over the batch
#       - none: disable scaling
# =====================================================================

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple

torch.set_default_dtype(torch.float64)

# ----------------------------- SPD / Sym utils -----------------------------
def _sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))

def _eigh_sym(M: torch.Tensor):
    # Batched symmetric eigendecomposition
    w, V = torch.linalg.eigh(_sym(M))
    return w, V

def _eigh_spd(C: torch.Tensor):
    w, V = _eigh_sym(C)
    return torch.clamp(w, min=1e-12), V

def logm_spd(C: torch.Tensor) -> torch.Tensor:
    w, V = _eigh_spd(C)
    return V @ torch.diag_embed(torch.log(w)) @ V.transpose(-1, -2)

def expm_sym(S: torch.Tensor) -> torch.Tensor:
    # exp of (batched) symmetric matrix
    w, V = _eigh_sym(S)
    return V @ torch.diag_embed(torch.exp(w)) @ V.transpose(-1, -2)

def invsqrtm_spd(C: torch.Tensor) -> torch.Tensor:
    w, V = _eigh_spd(C)
    return V @ torch.diag_embed(torch.rsqrt(w)) @ V.transpose(-1, -2)

def offdiag(M: torch.Tensor) -> torch.Tensor:
    d = M.shape[-1]
    mask = torch.ones(d, d, device=M.device, dtype=M.dtype) - torch.eye(d, d, device=M.device, dtype=M.dtype)
    return M * mask

def diag_vec(M: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(M, dim1=-2, dim2=-1)

def _ensure_long_subjects(s):
    """Return torch.long subject ids. Accepts None, list/np/torch (str or int)."""
    if s is None:
        return None
    if isinstance(s, torch.Tensor):
        if s.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
            return s.to(torch.long)
        s = s.detach().cpu().numpy()
    s = np.asarray(s)
    if np.issubdtype(s.dtype, np.integer):
        return torch.as_tensor(s.astype(np.int64), dtype=torch.long)
    uniq = np.unique(s)
    mapping = {u: i for i, u in enumerate(uniq)}
    idx = np.vectorize(mapping.get)(s)
    return torch.as_tensor(idx.astype(np.int64), dtype=torch.long)

def _one_hot(idx_long: torch.Tensor, K: int, dtype, device):
    N = idx_long.numel()
    out = torch.zeros((N, K), dtype=dtype, device=device)
    out.scatter_(1, idx_long.view(-1, 1), 1.0)
    return out

# ------------------ Dispersion scaling helpers (log-space) ------------------
@torch.no_grad()
def _dispersion_sigma_global(L: torch.Tensor, eps: float) -> torch.Tensor:
    # σ = sqrt( mean ||L_i||_F^2 )
    return torch.sqrt(torch.mean(torch.sum(L * L, dim=(-1, -2)))) + eps

@torch.no_grad()
def _dispersion_scale_per_subject(L: torch.Tensor, s_long: torch.Tensor, eps: float
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-subject dispersion scaling in log-space.
    Returns: (L_scaled, subjects(unique), sigmas_per_subject[|subjects|])
    """
    subjects = torch.unique(s_long)
    L_scaled = torch.empty_like(L)
    sigmas = torch.empty(len(subjects), dtype=L.dtype, device=L.device)
    for i, sid in enumerate(subjects):
        idx = (s_long == sid)
        Ls = L[idx]
        sigma = torch.sqrt(torch.mean(torch.sum(Ls * Ls, dim=(-1, -2)))) + eps
        L_scaled[idx] = Ls / sigma
        sigmas[i] = sigma
    return L_scaled, subjects, sigmas

@torch.no_grad()
def _apply_per_subject_scaling(L: torch.Tensor, s_long: torch.Tensor,
                               subjects_ref: torch.Tensor, sigmas_ref: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Apply known subject-wise σ to a new batch (e.g., transform phase),
    matching by subject ID if possible; if unseen subject appears, compute its
    own σ on-the-fly from its slice.
    """
    L_out = torch.empty_like(L)
    # Build a mapping tensor from subject id -> sigma
    sid_to_sigma = {int(subjects_ref[i].item()): sigmas_ref[i] for i in range(len(subjects_ref))}
    uniq = torch.unique(s_long)
    for sid in uniq:
        idx = (s_long == sid)
        if int(sid.item()) in sid_to_sigma:
            sigma = sid_to_sigma[int(sid.item())]
        else:
            # unseen subject at transform time: compute its own σ
            Li = L[idx]
            sigma = torch.sqrt(torch.mean(torch.sum(Li * Li, dim=(-1, -2)))) + eps
        L_out[idx] = L[idx] / sigma
    return L_out

# ------------------ Vectorized Fisher terms (no Python loops) ------------------
@torch.no_grad()
def _prep_indices(y_long: torch.Tensor, s_long: torch.Tensor | None):
    """Map labels/subjects to contiguous [0..K-1]/[0..S-1] and prebuild one-hots."""
    device, dtype = y_long.device, torch.float64
    classes = torch.unique(y_long)
    K = int(classes.numel())
    y_map = {int(c.item()): i for i, c in enumerate(classes)}
    y_idx = torch.as_tensor([y_map[int(t.item())] for t in y_long], dtype=torch.long, device=device)
    Gc = _one_hot(y_idx, K, dtype, device)          # (N,K)
    if s_long is None:
        return K, Gc, None, None, None
    subs = torch.unique(s_long)
    S = int(subs.numel())
    s_map = {int(u.item()): i for i, u in enumerate(subs)}
    s_idx = torch.as_tensor([s_map[int(t.item())] for t in s_long], dtype=torch.long, device=device)
    Gs = _one_hot(s_idx, S, dtype, device)          # (N,S)
    return K, Gc, S, Gs, (y_idx, s_idx)

def _dual_fishers_vectorized(L: torch.Tensor, R: torch.Tensor,
                             Gc: torch.Tensor, Gs: torch.Tensor | None):
    """
    Compute class Fisher (Bc, Wc, center) and subject Fisher (Bs, Ws) in a single
    vectorized pass. L is symmetric (N,d,d) in log-domain; R is (d,d).
    """
    Rt = R.transpose(0, 1)

    # ---------- Class ----------
    count_k = Gc.sum(dim=0).clamp_min(1.0)                          # (K,)
    M_k = torch.einsum('nk,nbc->kbc', Gc, L) / count_k[:, None, None]
    w_k = (count_k / count_k.sum()).view(-1, 1, 1)
    M = (M_k * w_k).sum(dim=0)                                      # global class mean
    # between on diag
    M_center = M_k - M                                              # (K,d,d)
    Bk = torch.einsum('ab,kbc,cd->kad', Rt, M_center, R)
    Bc = (diag_vec(Bk) ** 2).sum(dim=1)
    Bc = (Bc * count_k).sum()
    # within offdiag
    Mk_stack = torch.einsum('nk,kbc->nbc', Gc, M_k)
    Wi = L - Mk_stack
    Wrot = torch.einsum('ab,nbc,cd->nad', Rt, Wi, R)
    e_i = (offdiag(Wrot) ** 2).sum(dim=(-1, -2))
    Wc = e_i.sum()
    # center penalty
    Lrot_mean = torch.einsum('ab,nbc,cd->ad', Rt, L, R).mean(dim=0)
    center = torch.norm(offdiag(Lrot_mean)) ** 2

    # ---------- Subject ----------
    if Gs is None:
        Bs = L.new_zeros(())
        Ws = L.new_zeros(())
    else:
        count_s = Gs.sum(dim=0).clamp_min(1.0)                       # (S,)
        M_s = torch.einsum('ns,nbc->sbc', Gs, L) / count_s[:, None, None]
        w_s = (count_s / count_s.sum()).view(-1, 1, 1)
        Ms = (M_s * w_s).sum(dim=0)

        S_center = M_s - Ms
        Sk = torch.einsum('ab,sbc,cd->sad', Rt, S_center, R)
        Bs = (diag_vec(Sk) ** 2).sum(dim=1)
        Bs = (Bs * count_s).sum()

        Ms_stack = torch.einsum('ns,sbc->nbc', Gs, M_s)
        Wi_s = L - Ms_stack
        Wrot_s = torch.einsum('ab,nbc,cd->nad', Rt, Wi_s, R)
        e_i_s = (offdiag(Wrot_s) ** 2).sum(dim=(-1, -2))
        Ws = e_i_s.sum()

    return Bc, Wc, center, Bs, Ws

# ------------------------- DCR Pre-Aligner (Dual) --------------------
class DCRPreAlignerDualFast(nn.Module):
    """
    Learn R ∈ SO(d) maximizing class Fisher and minimizing subject Fisher.

    Loss:
        L = δ * (Bs / (Ws + eps))         # minimize subject Fisher (W/B)
          - γ * (Bc / (Wc + eps))         # maximize class Fisher   (B/W)
          + (β_t/d) * ||R - I||_F^2
          + κ * center

    Pipeline (fit/transform):
        RA Covariances (input SPD) →
        LOG →
        Dispersion scaling (per-subject/global/none) →
        (Optional) log standardization (mean/std over training set) →
        Rotation in log space (Rᵀ L R) →
        De-standardize →
        EXP (back to SPD) →
        (Optional) re-whiten mean in rotated SPD space.
    """
    def __init__(self,
                 steps: int = 200,
                 lr: float = 1e-3,
                 gamma_class: float = 1.0,
                 delta_subject: float = 1.0,
                 beta_reg: float = 3e-4,
                 kappa_center: float = 1e-4,
                 eps: float = 1e-12,
                 device: Optional[str] = None,
                 normalize_log: bool = True,
                 rewhiten_after: bool = True,
                 clip_grad_norm: Optional[float] = 5.0,
                 verbose_every: int = 40,
                 # NEW:
                 enable_dispersion_scale: bool = True,
                 dispersion_mode: str = "per_subject",   # 'per_subject' | 'global' | 'none'
                 dispersion_eps: float = 1e-8):
        super().__init__()
        self.steps = steps
        self.lr = lr
        self.gamma_class = gamma_class
        self.delta_subject = delta_subject
        self.beta_reg = beta_reg
        self.kappa_center = kappa_center
        self.eps = eps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize_log = normalize_log
        self.rewhiten_after = rewhiten_after
        self.clip_grad_norm = clip_grad_norm
        self.verbose_every = max(1, int(verbose_every))
        self.enable_dispersion_scale = enable_dispersion_scale
        self.dispersion_mode = dispersion_mode
        self.dispersion_eps = dispersion_eps

        # Parameters / buffers
        self.A_param: Optional[nn.Parameter] = None   # skew generator → R = exp(A - Aᵀ)
        self.R: Optional[torch.Tensor] = None
        self.I_d: Optional[torch.Tensor] = None
        self.L_mean: Optional[torch.Tensor] = None    # for log-space standardization
        self.L_std: Optional[torch.Tensor] = None

        # Dispersion cache (from fit)
        self.disp_subject_ids_: Optional[torch.Tensor] = None
        self.disp_sigmas_: Optional[torch.Tensor] = None
        self.disp_sigma_global_: Optional[torch.Tensor] = None

        self.to(self.device)

    def _R_from_A(self) -> torch.Tensor:
        S = self.A_param - self.A_param.transpose(0, 1)
        return torch.matrix_exp(S)

    # ------------------------------- Fit --------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, s: np.ndarray | None = None, verbose: bool = True):
        X_t = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        s_t = _ensure_long_subjects(s)
        if s_t is not None:
            s_t = s_t.to(self.device)

        # Precompute L = logm_spd(X) ONCE (batched)
        L = logm_spd(X_t)  # (N,d,d)

        # (NEW) Dispersion scaling (before standardization)
        if self.enable_dispersion_scale and self.dispersion_mode != "none":
            if self.dispersion_mode == "per_subject" and s_t is not None:
                L, subs, sigmas = _dispersion_scale_per_subject(L, s_t, self.dispersion_eps)
                # cache for transform:
                self.disp_subject_ids_ = subs.detach().clone()
                self.disp_sigmas_ = sigmas.detach().clone()
                self.disp_sigma_global_ = None
                if verbose:
                    mu = sigmas.mean().item()
                    sd = sigmas.std().item()
                    print(f"[Dispersion] per-subject σ: mean={mu:.3f} ± {sd:.3f} (|S|={len(sigmas)})")
            else:
                # global (or per_subject without s)
                sigma_g = _dispersion_sigma_global(L, self.dispersion_eps)
                L = L / sigma_g
                self.disp_subject_ids_ = None
                self.disp_sigmas_ = None
                self.disp_sigma_global_ = sigma_g.detach().clone()
                if verbose:
                    print(f"[Dispersion] global σ: {sigma_g.item():.3f}")
        else:
            self.disp_subject_ids_ = None
            self.disp_sigmas_ = None
            self.disp_sigma_global_ = None

        # Optional standardization in log-space (after dispersion scaling)
        if self.normalize_log:
            self.L_mean = L.mean(dim=0, keepdim=True).detach()
            self.L_std  = (L.std(dim=0, keepdim=True) + 1e-6).detach()
            L = (L - self.L_mean) / self.L_std
        else:
            self.L_mean = None
            self.L_std  = None

        # Prepare indices / one-hots once
        K, Gc, Snum, Gs, _ = _prep_indices(y_t, s_t)

        # Init params
        d = X_t.shape[-1]
        if self.A_param is None:
            self.A_param = nn.Parameter(torch.zeros((d, d), dtype=torch.float64, device=self.device))
            self.I_d = torch.eye(d, dtype=torch.float64, device=self.device)

        opt = optim.Adam([self.A_param], lr=self.lr)

        for step in range(self.steps):
            opt.zero_grad(set_to_none=True)
            R = self._R_from_A()

            Bc, Wc, center, Bs, Ws = _dual_fishers_vectorized(L, R, Gc, Gs)
            beta_t = self.beta_reg * (0.5 * (1.0 + math.cos(math.pi * step / max(1, self.steps - 1))))
            loss = (
                self.delta_subject * torch.log(Bs / (Ws + self.eps)) -
                self.gamma_class * torch.log(Bc / (Wc + self.eps)) +
                (beta_t * torch.norm(R - self.I_d) ** 2) / d +
                self.kappa_center * center
            )

            loss.backward()
            if self.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_([self.A_param], self.clip_grad_norm)
            opt.step()

            if verbose and (step % self.verbose_every == 0 or step == self.steps - 1):
                print(f"[DCRPreAlignerDualFast] {step:4d} | "
                      f"loss={loss.item():.3e} | "
                      f"class W/B={(Wc/(Bc+self.eps)).item():.3e} | "
                      f"subj B/W={(Bs/(Ws+self.eps)).item():.3e} | "
                      f"center={center.item():.3e}")

        with torch.no_grad():
            self.R = self._R_from_A().detach().clone()
        return self

    # ----------------------------- Transform -----------------------------
    @torch.no_grad()
    def transform(self, X: np.ndarray, s: np.ndarray | None = None) -> np.ndarray:
        """
        Apply log -> (optional dispersion scaling) -> (optional norm) ->
        rotate in log -> (denorm) -> exp back to SPD.
        Optionally re-whiten mean in the rotated space.

        If dispersion_mode='per_subject' and s is provided, we compute per-subject σ
        for the given batch if unseen subjects appear; otherwise, when training
        subjects are known, we reuse cached σ where possible.
        If s is None:
          - per_subject: compute σ per unique (implicit) subject over this batch as global fallback
          - global: compute a single σ over the batch (like fit global)
        """
        assert self.R is not None, "Call fit() before transform()."
        X_t = torch.as_tensor(X, dtype=torch.float64, device=self.device)

        # 1) log
        L = logm_spd(X_t)  # (N,d,d)

        # 2) dispersion scaling (same policy as fit)
        if self.enable_dispersion_scale and self.dispersion_mode != "none":
            if self.dispersion_mode == "per_subject":
                s_t = _ensure_long_subjects(s) if s is not None else None
                if s_t is not None:
                    s_t = s_t.to(X_t.device)
                    if (self.disp_subject_ids_ is not None) and (self.disp_sigmas_ is not None):
                        # try to reuse training sigmas where subjects overlap; unseen → on-the-fly
                        L = _apply_per_subject_scaling(
                            L, s_t, self.disp_subject_ids_.to(X_t.device),
                            self.disp_sigmas_.to(X_t.device), self.dispersion_eps
                        )
                    else:
                        # no cache (e.g., fit used global); compute per-subject on the fly
                        L, _, _ = _dispersion_scale_per_subject(L, s_t, self.dispersion_eps)
                else:
                    # no subject info at transform → fallback: global σ over this batch
                    sigma_g = _dispersion_sigma_global(L, self.dispersion_eps)
                    L = L / sigma_g
            else:
                # global dispersion
                sigma_g = _dispersion_sigma_global(L, self.dispersion_eps)
                L = L / sigma_g

        # 3) apply SAME normalization as training (if used)
        if self.normalize_log:
            assert self.L_mean is not None and self.L_std is not None, "Missing L_mean/L_std; call fit() first."
            L = (L - self.L_mean) / self.L_std

        # 4) rotate in LOG space
        Rt = self.R.transpose(0, 1)
        L_rot = torch.einsum('ab,nbc,cd->nad', Rt, L, self.R)

        # 5) de-normalize back (if used)
        if self.normalize_log:
            L_rot = L_rot * self.L_std + self.L_mean

        # 6) push back to SPD
        X_rot = expm_sym(L_rot)

        # 7) (optional) mean re-whitening in the rotated space
        if self.rewhiten_after:
            C_mean = X_rot.mean(dim=0)
            Wm = invsqrtm_spd(C_mean)
            X_rot = torch.einsum('ab,nbc,cd->nad', Wm, X_rot, Wm)

        return X_rot.cpu().numpy()

    def fit_transform(self, X: np.ndarray, y: np.ndarray, s: np.ndarray | None = None, verbose: bool = True) -> np.ndarray:
        self.fit(X, y, s, verbose=verbose)
        # Use same subjects on transform for consistent per-subject scaling
        return self.transform(X, s=s)

#======================================================================
#🧾 COMPARISON SUMMARY (Mean ± Std) delta = 0.3, gamma = 1.5
#======================================================================
#TSLR       | RA:  52.89% ± 14.48% | DCR:  53.01% ± 16.71%
#MDM        | RA:  52.43% ± 15.66% | DCR:  50.46% ± 14.82%
#TSA-LDA    | RA:  53.51% ± 15.29% | DCR:  53.97% ± 16.02%
#======================================================================
#Runtime → RA: 50.44s | DCR+RA: 104.77s
#======================================================================

if __name__ == "__main__":
    """
    Compare pre-aligned (RA) covariances vs. DCRPreAlignerDualFast refinement
    using TSLR / MDM / TSA-LDA on LOSO cross-subject evaluation.
    """
    
    import numpy as np
    import torch
    from time import time
    
    # ---------------------------------------------------------------
    # ⬇️ Import model + downstream classifiers
    # ---------------------------------------------------------------
    from models.riemann.dcrbifa import DCRPreAlignerDualFast
    from models.riemann.tslr import RiemannTSLR
    from models.riemann.mdm import RiemannMDM
    from models.riemann.tsa_lda import TSALDA
    
    # ============================================================== #
    # Config
    # ============================================================== #
    CACHE_PATH = "./EEG_data/bci_active4.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    DCR_CFG = dict(
        steps=400,
        lr=1e-3,
        gamma_class=1.5,
        delta_subject=0.3,
        beta_reg=4e-4,
        kappa_center=1e-4,
        eps=1e-12,
        device=DEVICE,
        normalize_log=True,
        rewhiten_after=True,
        clip_grad_norm=5.0,
        verbose_every=50,
    )
    
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 70)
    print(" DCRPreAlignerDualFast (Dual Fisher) vs Pure RA (Pre-aligned inputs) ")
    print(f" Device: {DEVICE}")
    print("=" * 70)
    
    # ============================================================== #
    # Load pre-aligned covariances
    # ============================================================== #
    data = torch.load(CACHE_PATH, map_location="cpu")
    covs_ra = np.stack([c.cpu().numpy() for c in data["ra_covs"]])   # already subject-wise RA’d
    labels  = np.asarray(data["labels"]).astype(int)
    subjects = np.asarray(data["subj"])
    
    S_ids = np.unique(subjects)
    sid_to_int = {sid: i for i, sid in enumerate(S_ids)}
    s_int = np.vectorize(sid_to_int.get)(subjects)
    
    S = len(S_ids)
    n_classes = int(labels.max() + 1)
    N, d, _ = covs_ra.shape
    print(f"N={N}, channels={d}, classes={n_classes}, subjects={S}\n")
    
    # ============================================================== #
    # Helpers
    # ============================================================== #
    def normalize_covs(X):
        """Trace-normalize for scale invariance."""
        Xn = np.empty_like(X)
        for i in range(len(X)):
            tr = np.trace(X[i])
            Xn[i] = X[i] / tr if tr > 0 else X[i]
        return Xn
    
    def evaluate_models(X_train, y_train, X_test, y_test, cov_type="RA"):
        tslr = RiemannTSLR(cov_type=cov_type)
        mdm  = RiemannMDM(cov_type=cov_type)
        lda  = TSALDA(cov_type=cov_type)
        tslr.fit(X_train, y_train)
        mdm.fit(X_train, y_train)
        lda.fit(X_train, y_train)
        return (
            tslr.score(X_test, y_test),
            mdm.score(X_test, y_test),
            lda.score(X_test, y_test),
        )
    
    # ============================================================== #
    # LOSO Cross-Subject Evaluation
    # ============================================================== #
    def run_fold(test_sid, use_dcr=False):
        train_mask = (subjects != test_sid)
        test_mask  = (subjects == test_sid)
        X_train, y_train, s_train = covs_ra[train_mask], labels[train_mask], s_int[train_mask]
        X_test,  y_test,  s_test  = covs_ra[test_mask],  labels[test_mask],  s_int[test_mask]
    
        if use_dcr:
            print(f"→ Training DCRPreAlignerDualFast (excluding {test_sid}) on {len(X_train)} trials")
            pre = DCRPreAlignerDualFast(**DCR_CFG)
            X_train = pre.fit_transform(X_train, y_train, s=s_train, verbose=False)
            X_test  = pre.transform(X_test)
        # else: no transform, use pre-aligned RA covariances as-is
    
        X_train = normalize_covs(X_train)
        X_test  = normalize_covs(X_test)
        return evaluate_models(X_train, y_train, X_test, y_test, cov_type="RA")
    
    def run_loso(tag, use_dcr=False):
        print(f"\n{'='*70}\n🧠 Running LOSO ({tag})\n{'='*70}")
        accs_tslr, accs_mdm, accs_lda = [], [], []
        t0 = time()
    
        for sid in S_ids:
            a_t, a_m, a_l = run_fold(sid, use_dcr)
            accs_tslr.append(a_t)
            accs_mdm.append(a_m)
            accs_lda.append(a_l)
            print(f"  ✅ Subject {sid:>6}: "
                  f"TSLR={100*a_t:5.2f}%  MDM={100*a_m:5.2f}%  TSA-LDA={100*a_l:5.2f}%")
    
        elapsed = time() - t0
        accs_tslr, accs_mdm, accs_lda = map(np.asarray, (accs_tslr, accs_mdm, accs_lda))
    
        print(f"\n→ LOSO Mean Accuracy ({tag})")
        print(f"   TSLR:    {100*accs_tslr.mean():.2f}% ± {100*accs_tslr.std():.2f}%")
        print(f"   MDM:     {100*accs_mdm.mean():.2f}% ± {100*accs_mdm.std():.2f}%")
        print(f"   TSA-LDA: {100*accs_lda.mean():.2f}% ± {100*accs_lda.std():.2f}%")
        print(f"⏱️  Time ({tag}): {elapsed:.2f}s\n")
        return accs_tslr, accs_mdm, accs_lda, elapsed
    
    # ============================================================== #
    # Run both setups
    # ============================================================== #
    accs_tslr_dcr, accs_mdm_dcr, accs_lda_dcr, time_dcr = run_loso(
        "DCRPreAlignerDualFast (Dual Fisher)", use_dcr=True
    )
    accs_tslr_ra, accs_mdm_ra, accs_lda_ra, time_ra = run_loso(
        "Pure Riemannian Alignment (RA baseline)", use_dcr=False
    )
    
    # ============================================================== #
    # Summary
    # ============================================================== #
    print("\n" + "="*70)
    print("🧾 COMPARISON SUMMARY (Mean ± Std)")
    print("="*70)
    def summary_line(name, a_ra, a_dcr):
        mean_ra, std_ra = a_ra.mean(), a_ra.std()
        mean_dcr, std_dcr = a_dcr.mean(), a_dcr.std()
        print(f"{name:<10} | RA: {100*mean_ra:6.2f}% ± {100*std_ra:5.2f}%"
              f" | DCR+RA: {100*mean_dcr:6.2f}% ± {100*std_dcr:5.2f}%")
    
    summary_line("TSLR", accs_tslr_ra, accs_tslr_dcr)
    summary_line("MDM",  accs_mdm_ra,  accs_mdm_dcr)
    summary_line("TSA-LDA", accs_lda_ra, accs_lda_dcr)
    print("="*70)
    print(f"Runtime → RA: {time_ra:.2f}s | DCR+RA: {time_dcr:.2f}s")
    print("="*70)
    