# models/riemann/rpa.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
import torch

# ============================ helpers ============================

def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def _eigh_spd(A: np.ndarray, eps: float = 1e-12):
    A = _sym(A)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return w, V

def _logm_I(X: np.ndarray) -> np.ndarray:
    # matrix log at identity: log(C)
    w, V = _eigh_spd(X)
    return V @ np.diag(np.log(w)) @ V.T

def _expm_I(S: np.ndarray) -> np.ndarray:
    # matrix exp at identity: exp(T)
    w, V = np.linalg.eigh(_sym(S))
    return V @ np.diag(np.exp(w)) @ V.T

def _dispersion_I(mats: np.ndarray) -> float:
    """
    True Riemannian dispersion at I:
      σ = mean_i || log_I(C_i) ||_F^2
    """
    return float(np.mean([
        np.linalg.norm(_logm_I(_sym(C)), "fro") ** 2 for C in mats
    ]))

def _orth_proj_svd(W_t: torch.Tensor) -> torch.Tensor:
    U, _, Vt = torch.linalg.svd(W_t, full_matrices=False)
    return U @ Vt

def _normalize_domains(domains) -> np.ndarray:
    return np.array([str(d).strip() for d in np.asarray(domains)])


# ============================ RPA (optimize R_k directly) ============================

class RPA:
    """
    Full RPA: per-subject orthogonal R_k learned directly.

    FIT (train only)
      - σ_S = dispersion_I(all training covs)
      - For each subject k:
          σ_k = dispersion_I(X_k)
          ρ_k = sqrt(σ_S / σ_k)
          C'_i = exp_I(ρ_k * log_I(C_i))               # scale in log-space
          M_k = mean_riemann({C'_i})
          T_k = log_I(M_k)
      - M_s = mean_riemann({M_k}), T_s = log_I(M_s)
      - Optimize {R_k} (each orthogonal) to minimize:
          Σ_k || R_k^T T_k R_k - T_s ||_F^2
      - Save: σ_S, T_s, {R_k} for known subjects

    TRANSFORM (train/test, unlabeled per-subject batches)
      - For each subject batch:
          σ_T = dispersion_I(batch), ρ = sqrt(σ_S / σ_T)
          Mk_batch from C'_i = exp_I(ρ * log_I(C_i))   # batch mean after scaling
          Tk_batch = log_I(Mk_batch)
          If subject seen in training: use stored R_k
          Else: solve a short inner alignment to get R (unsupervised) w.r.t. T_s
          For each trial:
              Ti = ρ * log_I(Ci)
              Ti' = R^T Ti R
              Ci' = exp_I(Ti')
    """

    def __init__(self, max_iter=200, lr=5e-2, seed=0):
        self.max_iter = max_iter
        self.lr = lr
        self.seed = seed

        self.sigma_S_ = None
        self.Ms_ = None
        self.Ts_ = None
        self.R_dict_ = {}    # subject_id -> R_k (np.ndarray, orthogonal)

        # test-time inner solver iters (unsupervised); keep small
        self.test_inner_iter_ = max(20, min(100, max_iter // 2))

    def _rho_from_sigmas(self, sigma_ref: float, sigma_cur: float) -> float:
        eps = 1e-12
        return float(np.sqrt(max(eps, sigma_ref) / max(eps, sigma_cur)))

    # ------------- fit -------------

    def fit(self, X: np.ndarray, y=None, domains=None):
        assert X.ndim == 3, "X must be (N,C,C)"
        domains = _normalize_domains(domains)
        torch.manual_seed(self.seed)
        subj_ids = np.unique(domains)
        Cdim = X.shape[1]

        # Global dispersion over ALL training covariances (log-space at I)
        self.sigma_S_ = _dispersion_I(X)

        Mks = []
        Tks = {}

        # Per-subject scaled means in tangent at I
        for sid in subj_ids:
            idx = np.where(domains == sid)[0]
            idx = idx[idx < len(X)]  # safety
            Xs = X[idx]

            sigma_k = _dispersion_I(Xs)
            rho_k = self._rho_from_sigmas(self.sigma_S_, sigma_k)

            Xs_log = [_logm_I(_sym(C)) for C in Xs ]
            Xs_log_scaled = [ rho_k * T for T in Xs_log ]
            Xs_scaled = [ _expm_I(T) for T in Xs_log_scaled ]
            Xs_scaled = np.array([ _sym(C) for C in Xs_scaled ])

            Mk = mean_riemann(Xs_scaled)
            Tk = _logm_I(_sym(Mk))

            Mks.append(Mk)
            Tks[sid] = Tk

        # Global mean & target Ts
        self.Ms_ = mean_riemann(np.stack(Mks, axis=0))
        self.Ts_ = _logm_I(_sym(self.Ms_))
        Ts = torch.tensor(self.Ts_, dtype=torch.double)

        # Set up one learnable R_k per subject
        R_params = []
        R_keys = []
        for sid in subj_ids:
            Rk = torch.eye(Cdim, dtype=torch.double, requires_grad=True)
            R_params.append(Rk)
            R_keys.append(sid)

        opt = torch.optim.SGD(R_params, lr=self.lr, momentum=0.9)

        # Optimize {R_k}
        for _ in range(self.max_iter):
            opt.zero_grad()
            loss = torch.zeros((), dtype=torch.double)

            # project each R_k to orthogonal first
            with torch.no_grad():
                for Rk in R_params:
                    Rk.copy_(_orth_proj_svd(Rk))

            for sid, Rk in zip(R_keys, R_params):
                Tk = torch.tensor(Tks[sid], dtype=torch.double)
                Tk_rot = Rk.T @ Tk @ Rk
                loss = loss + torch.norm(Tk_rot - Ts, p='fro') ** 2

            loss.backward()
            opt.step()

        # final projection and store
        with torch.no_grad():
            for sid, Rk in zip(R_keys, R_params):
                Rk.copy_(_orth_proj_svd(Rk))
                self.R_dict_[sid] = Rk.detach().cpu().numpy()

        return self

    # ------------- inner solver for a single subject (test-time) -------------

    def _solve_R_for_batch(self, Tk_np: np.ndarray, Cdim: int) -> np.ndarray:
        """Unsupervised alignment for unseen subject: minimize ||R^T Tk R - Ts||_F^2."""
        Ts = torch.tensor(self.Ts_, dtype=torch.double)
        Tk = torch.tensor(Tk_np, dtype=torch.double)
        R = torch.eye(Cdim, dtype=torch.double, requires_grad=True)
        opt = torch.optim.SGD([R], lr=self.lr, momentum=0.9)

        for _ in range(self.test_inner_iter_):
            opt.zero_grad()
            with torch.no_grad():
                R.copy_(_orth_proj_svd(R))
            Tk_rot = R.T @ Tk @ R
            loss = torch.norm(Tk_rot - Ts, p='fro') ** 2
            loss.backward()
            opt.step()

        with torch.no_grad():
            R.copy_(_orth_proj_svd(R))
        return R.detach().cpu().numpy()

    # ------------- transform -------------

    def _batch_params(self, X_batch: np.ndarray, sid: str | None, Cdim: int):
        # compute ρ vs global σ_S; get batch mean in log-space scaling; get Tk
        sigma_T = _dispersion_I(X_batch)
        rho = self._rho_from_sigmas(self.sigma_S_, sigma_T)

        Xb_log = [ _logm_I(_sym(C)) for C in X_batch ]
        Xb_log_scaled = [ rho * T for T in Xb_log ]
        Xb_scaled = [ _expm_I(T) for T in Xb_log_scaled ]
        Xb_scaled = np.array([ _sym(C) for C in Xb_scaled ])

        Mk = mean_riemann(Xb_scaled)
        Tk = _logm_I(_sym(Mk))

        # choose R: trained if available, otherwise solve on the fly
        if sid is not None and sid in self.R_dict_:
            R = self.R_dict_[sid]
        else:
            R = self._solve_R_for_batch(Tk, Cdim)
        return rho, R

    def transform(self, X: np.ndarray, domains=None) -> np.ndarray:
        assert self.Ts_ is not None and self.sigma_S_ is not None, "Call fit() first."
        X = np.asarray(X)
        N, Cdim, _ = X.shape
        X_out = np.empty_like(X)

        if domains is None:
            rho, R = self._batch_params(X, sid=None, Cdim=Cdim)
            for i in range(N):
                Ti = _logm_I(_sym(X[i]))
                Ti = rho * Ti
                X_out[i] = _expm_I(R.T @ Ti @ R)
            return X_out

        domains = _normalize_domains(domains)
        for sid in np.unique(domains):
            idx = np.where(domains == sid)[0]
            idx = idx[idx < len(X)]
            rho, R = self._batch_params(X[idx], sid=sid, Cdim=Cdim)
            for i in idx:
                Ti = _logm_I(_sym(X[i]))
                Ti = rho * Ti
                X_out[i] = _expm_I(R.T @ Ti @ R)
        return X_out

    def fit_transform(self, X: np.ndarray, y=None, domains=None) -> np.ndarray:
        self.fit(X, y=y, domains=domains)
        return self.transform(X, domains=domains)