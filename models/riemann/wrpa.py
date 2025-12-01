# models/riemann/wrpa.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
import torch

# ---------------- helpers ----------------

def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def _eigh_spd(A: np.ndarray, eps: float = 1e-12):
    A = _sym(A)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return w, V

def _logm_I(X: np.ndarray) -> np.ndarray:
    w, V = _eigh_spd(X)
    return V @ np.diag(np.log(w)) @ V.T

def _expm_I(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(_sym(S))
    return V @ np.diag(np.exp(w)) @ V.T

def _dispersion_I(mats: np.ndarray) -> float:
    # If you prefer Euclidean C-I, swap this implementation accordingly.
    return float(np.mean([
        np.linalg.norm(_logm_I(_sym(C)), "fro") ** 2 for C in mats
    ]))

def _orth_proj_svd(W_t: torch.Tensor) -> torch.Tensor:
    U, _, Vt = torch.linalg.svd(W_t, full_matrices=False)
    return U @ Vt

def _normalize_domains(domains) -> np.ndarray:
    return np.array([str(d).strip() for d in np.asarray(domains)])

def _orth_from_weighted_U(U: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Build orthonormal basis from weighted eigenvectors: U @ diag(s) → polar factor.
    """
    X = U @ np.diag(s)
    # polar factor via SVD gives closest orthogonal
    U1, _, Vt1 = np.linalg.svd(X, full_matrices=False)
    return U1 @ Vt1

# ---------------- WRPA aligner ----------------

class WRPA:
    """
    Weighted-RPA with per-subject R_k = W @ Uhat_k, where
      Uhat_k = orth( U_k @ diag(lambda_k ** u_power) ).

    FIT:
      - σ_S from all covs (log-space dispersion by default)
      - Per subject:
          ρ_k = sqrt(σ_S / σ_k)
          scale in log-space: C' = exp_I(ρ_k * log_I(C))
          Mk = mean_riemann(C'), Tk = log_I(Mk)
          eig(Tk) -> (lambda_k, U_k)
          Uhat_k = orth(U_k @ diag(lambda_k ** u_power))
      - Global Ms -> Ts = log_I(Ms)
      - Learn shared orthogonal W by minimizing:
            Σ_k || (W Uhat_k)^T Tk (W Uhat_k) - (W^T Ts W) ||_F^2

    TRANSFORM:
      - For each subject batch:
          compute ρ, Mk, Tk, (lambda, U) → Uhat
          R = W @ Uhat
          For each trial: Ti = ρ * log_I(Ci);  Ci' = exp_I(R^T Ti R)
    """

    def __init__(self, max_iter=200, lr=5e-2, seed=0, u_power: float = 0.0):
        self.max_iter = max_iter
        self.lr = lr
        self.seed = seed
        self.u_power = u_power

        self.sigma_S_ = None
        self.Ms_ = None
        self.Ts_ = None
        self.W_ = None
        self.Uhat_dict_ = {}  # store Uhat_k for seen subjects (optional reuse)

    def _rho_from_sigmas(self, sigma_ref: float, sigma_cur: float) -> float:
        eps = 1e-12
        return float(np.sqrt(max(eps, sigma_ref) / max(eps, sigma_cur)))

    # ---------- fit ----------

    def fit(self, X: np.ndarray, y=None, domains=None):
        assert X.ndim == 3, "X must be (N,C,C)"
        domains = _normalize_domains(domains)
        torch.manual_seed(self.seed)
        subj_ids = np.unique(domains)
        Cdim = X.shape[1]

        # global dispersion
        self.sigma_S_ = _dispersion_I(X)

        Mks = []
        Tks = {}
        Uhats = {}

        for sid in subj_ids:
            idx = np.where(domains == sid)[0]
            Xs = X[idx]

            sigma_k = _dispersion_I(Xs)
            rho_k = self._rho_from_sigmas(self.sigma_S_, sigma_k)

            # scale in log-space
            Xs_log = [_logm_I(_sym(C)) for C in Xs]
            Xs_log_scaled = [rho_k * T for T in Xs_log]
            Xs_scaled = [_expm_I(T) for T in Xs_log_scaled]
            Xs_scaled = np.array([_sym(C) for C in Xs_scaled])

            Mk = mean_riemann(Xs_scaled)
            Tk = _logm_I(_sym(Mk))
            lam_k, U_k = _eigh_spd(Tk)

            # weighted eigen-directions, then orthonormalize
            s = (lam_k ** self.u_power) if self.u_power != 0.0 else np.ones_like(lam_k)
            Uhat_k = _orth_from_weighted_U(U_k, s)

            Mks.append(Mk)
            Tks[sid] = Tk
            Uhats[sid] = Uhat_k

        self.Ms_ = mean_riemann(np.stack(Mks, axis=0))
        self.Ts_ = _logm_I(_sym(self.Ms_))

        # torch buffers
        Ts = torch.tensor(self.Ts_, dtype=torch.double)
        T_stack = [torch.tensor(Tks[sid], dtype=torch.double) for sid in subj_ids]
        Uhat_stack = [torch.tensor(Uhats[sid], dtype=torch.double) for sid in subj_ids]

        # learn shared W
        W = torch.eye(Cdim, dtype=torch.double, requires_grad=True)
        opt = torch.optim.SGD([W], lr=self.lr, momentum=0.9)

        for _ in range(self.max_iter):
            opt.zero_grad()
            with torch.no_grad():
                W.copy_(_orth_proj_svd(W))
            lam_s, Us = _eigh_spd(Ts)
            s = (lam_s ** self.u_power) if self.u_power != 0.0 else np.ones_like(lam_s)
            Us = _orth_from_weighted_U(Us, s)   # ✅ correct reference
            Rs = W @ Us
            Ts_W = Rs.T @ Ts @ Rs

            loss = torch.zeros((), dtype=torch.double)
            for Tk, Uhat in zip(T_stack, Uhat_stack):
                Rk = W @ Uhat
                Tk_rot = Rk.T @ Tk @ Rk
                loss = loss + torch.norm(Tk_rot - Ts_W, p='fro') ** 2

            loss.backward()
            opt.step()

        with torch.no_grad():
            W.copy_(_orth_proj_svd(W))
        self.W_ = W.detach().cpu().numpy()
        self.Uhat_dict_ = Uhats  # save for reuse if same subjects appear
        return self

    # ---------- transform ----------

    def _batch_params(self, X_batch: np.ndarray):
        sigma_T = _dispersion_I(X_batch)
        rho = self._rho_from_sigmas(self.sigma_S_, sigma_T)

        Xb_log = [_logm_I(_sym(C)) for C in X_batch]
        Xb_log_scaled = [rho * T for T in Xb_log]
        Xb_scaled = [_expm_I(T) for T in Xb_log_scaled]
        Xb_scaled = np.array([_sym(C) for C in Xb_scaled])

        Mk = mean_riemann(Xb_scaled)
        Tk = _logm_I(_sym(Mk))
        lam, U = _eigh_spd(Tk)
        s = (lam ** self.u_power) if self.u_power != 0.0 else np.ones_like(lam)
        Uhat = _orth_from_weighted_U(U, s)
        return rho, Uhat

    def transform(self, X: np.ndarray, domains=None) -> np.ndarray:
        assert self.W_ is not None and self.Ts_ is not None and self.sigma_S_ is not None, "Call fit() first."
        X = np.asarray(X)
        X_out = np.empty_like(X)

        if domains is None:
            rho, Uhat = self._batch_params(X)
            R = self.W_ @ Uhat
            for i in range(len(X)):
                Ti = rho * _logm_I(_sym(X[i]))
                X_out[i] = _expm_I(R.T @ Ti @ R)
            return X_out

        domains = _normalize_domains(domains)
        for sid in np.unique(domains):
            idx = np.where(domains == sid)[0]
            Xs = X[idx]
            # if we saw this subject in training, we can reuse its Uhat; otherwise recompute
            if sid in self.Uhat_dict_:
                Uhat = self.Uhat_dict_[sid]
                # recompute rho per (unlabeled) batch
                sigma_T = _dispersion_I(Xs)
                rho = self._rho_from_sigmas(self.sigma_S_, sigma_T)
            else:
                rho, Uhat = self._batch_params(Xs)
            R = self.W_ @ Uhat
            for i in idx:
                Ti = rho * _logm_I(_sym(X[i]))
                X_out[i] = _expm_I(R.T @ Ti @ R)
        return X_out

    def fit_transform(self, X: np.ndarray, y=None, domains=None) -> np.ndarray:
        self.fit(X, y=y, domains=domains)
        return self.transform(X, domains=domains)