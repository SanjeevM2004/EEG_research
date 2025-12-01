# models/riemann/polar_rpa.py

import numpy as np
import torch
from pyriemann.utils.mean import mean_riemann


# ---------------- helpers ----------------

def _sym(A): return 0.5 * (A + A.T)

def _eigh_spd(A, eps=1e-12):
    A = _sym(A)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return w, V

def _logm_I(X):
    w, V = _eigh_spd(X)
    return V @ np.diag(np.log(w)) @ V.T

def _expm_I(S):
    w, V = np.linalg.eigh(_sym(S))
    return V @ np.diag(np.exp(w)) @ V.T

def _dispersion_I(mats):
    return float(np.mean([np.linalg.norm(_logm_I(_sym(C)), "fro") ** 2 for C in mats]))

def _orth_proj_svd(W_t):
    U, _, Vt = torch.linalg.svd(W_t, full_matrices=False)
    return U @ Vt

def _normalize_domains(domains):
    return np.array([str(d).strip() for d in np.asarray(domains)])


# ---------------- POLAR-RPA ----------------

class PolarRPA:
    """
    Polar RPA: align each subject's mean orientation to the global mean
    using the polar rotation from their eigenbases.

    FIT:
      - Compute σ_S from all covs.
      - For each subject:
          ρ_k = sqrt(σ_S / σ_k)
          scale in log-space: C' = exp_I(ρ_k * log_I(C))
          Mk = mean_riemann(C'), Tk = log_I(Mk)
      - Compute global mean Ms and Ts.
      - For each subject:
          Uk, Us = eigbases(Tk, Ts)
          Rk = polar(Us @ Uk.T)
          store Rk.

    TRANSFORM:
      - For each subject:
          compute ρ, Mk
          use stored Rk to rotate each trial:
              Ti = ρ * log_I(Ci)
              Ci' = exp_I(Rk.T @ Ti @ Rk)
    """

    def __init__(self, seed=0):
        self.seed = seed
        self.sigma_S_ = None
        self.Ms_ = None
        self.Ts_ = None
        self.R_dict_ = {}

    def _rho_from_sigmas(self, sigma_ref, sigma_cur):
        eps = 1e-12
        return float(np.sqrt(max(eps, sigma_ref) / max(eps, sigma_cur)))

    def _polar_rotation(self, Uk, Us):
        """Orthogonal Procrustes / polar factor between Uk and Us."""
        Uv, _, Vt = np.linalg.svd(Us @ Uk.T, full_matrices=False)
        return Uv @ Vt

    # ---------- fit ----------

    def fit(self, X, y=None, domains=None):
        assert X.ndim == 3, "X must be (N,C,C)"
        domains = _normalize_domains(domains)
        np.random.seed(self.seed)
        subj_ids = np.unique(domains)

        # global dispersion
        self.sigma_S_ = _dispersion_I(X)

        Mks, Tks = [], {}
        for sid in subj_ids:
            idx = np.where(domains == sid)[0]
            Xs = X[idx]
            sigma_k = _dispersion_I(Xs)
            rho_k = self._rho_from_sigmas(self.sigma_S_, sigma_k)

            Xs_log = [_logm_I(_sym(C)) for C in Xs]
            Xs_scaled = [_expm_I(rho_k * T) for T in Xs_log]
            Xs_scaled = np.array([_sym(C) for C in Xs_scaled])

            Mk = mean_riemann(Xs_scaled)
            Tk = _logm_I(_sym(Mk))

            Mks.append(Mk)
            Tks[sid] = Tk

        # global mean
        self.Ms_ = mean_riemann(np.stack(Mks, axis=0))
        self.Ts_ = _logm_I(_sym(self.Ms_))

        # compute global eigenbasis
        _, Us = _eigh_spd(self.Ts_)

        # compute per-subject rotation
        for sid in subj_ids:
            _, Uk = _eigh_spd(Tks[sid])
            Rk = self._polar_rotation(Uk, Us)
            self.R_dict_[sid] = Rk

        return self

    # ---------- transform ----------

    def transform(self, X, domains=None):
        assert self.Ts_ is not None and self.sigma_S_ is not None, "Call fit() first."
        X = np.asarray(X)
        X_out = np.empty_like(X)
        Cdim = X.shape[1]

        if domains is None:
            # If no domain info, use identity rotation and global scaling
            rho = 1.0
            R = np.eye(Cdim)
            for i in range(len(X)):
                Ti = rho * _logm_I(_sym(X[i]))
                X_out[i] = _expm_I(R.T @ Ti @ R)
            return X_out

        domains = _normalize_domains(domains)
        for sid in np.unique(domains):
            idx = np.where(domains == sid)[0]
            Xs = X[idx]
            sigma_T = _dispersion_I(Xs)
            rho = self._rho_from_sigmas(self.sigma_S_, sigma_T)
            R = self.R_dict_.get(sid, np.eye(Cdim))
            for i in idx:
                Ti = rho * _logm_I(_sym(X[i]))
                X_out[i] = _expm_I(R.T @ Ti @ R)
        return X_out

    def fit_transform(self, X, y=None, domains=None):
        self.fit(X, y, domains)
        return self.transform(X, domains)

