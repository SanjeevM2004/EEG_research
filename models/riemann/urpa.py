import numpy as np
from typing import Dict, Tuple

# -------------------- Utility Functions --------------------

def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def _fro_mean(mats: np.ndarray) -> np.ndarray:
    return _sym(np.mean(mats, axis=0))

def _fro_dispersion(mats: np.ndarray) -> float:
    """Average Frobenius dispersion of SPD set."""
    M = _fro_mean(mats)
    diffs = mats - M
    return float(np.mean([np.linalg.norm(_sym(D), "fro") ** 2 for D in diffs]))

def _normalize_domains(domains) -> np.ndarray:
    """Convert domains to string labels for consistency."""
    return np.array([str(d) for d in np.asarray(domains)])

def _eigvecs_np(M: np.ndarray) -> np.ndarray:
    """Return eigenvectors (columns) of SPD matrix M."""
    _, V = np.linalg.eigh(_sym(M))
    return V


class URPA:
       
    """URPA: Dispersion-based subject normalization (no rotation)."""

    def __init__(self, split_ratio: float = 0.8, seed: int = 0):
        assert 0.5 <= split_ratio < 1.0
        self.split_ratio = split_ratio
        self.seed = seed
        self.scale_: Dict[str, float] = {}
        self.disp_: Dict[str, Tuple[float, float]] = {}

    def _split_indices(self, n):
        m = max(1, int(self.split_ratio * n))
        major = np.arange(n)[:m]
        small = np.arange(n)[m:] if m < n else np.array([n - 1])
        return major, small

    def fit(self, X, y=None, domains=None):
        assert X.ndim == 3
        if domains is None:
            raise ValueError("URPA.fit requires subject/domain IDs.")
        np.random.seed(self.seed)
        domains = np.array([str(d) for d in np.asarray(domains)])
        subj_ids = np.unique(domains)
        self.scale_.clear()
        self.disp_.clear()

        for sid in subj_ids:
            idx = np.where(domains == sid)[0]
            X_blk = X[idx]
            rng = np.random.default_rng(self.seed)
            n = len(X_blk)
            if n < 2:
                self.scale_[sid] = 1.0
                continue
            idxs = np.arange(n)
            rng.shuffle(idxs)
            major, small = self._split_indices(n)
            X_major, X_small = X_blk[major], X_blk[small]
            d_maj = _fro_dispersion(X_major)
            d_sml = _fro_dispersion(X_small)
            lam = 1.0 if d_maj <= 1e-12 or d_sml <= 1e-12 else np.sqrt(d_maj / d_sml)
            self.scale_[sid] = lam
            self.disp_[sid] = (d_maj, d_sml)
        return self

    def transform(self, X, domains=None):
        assert self.scale_, "Call fit() first."
        if domains is None:
            raise ValueError("URPA.transform requires domains.")
        domains = np.array([str(d) for d in np.asarray(domains)])
        X_out = np.empty_like(X)
        for i in range(len(X)):
            sid = domains[i]
            lam = self.scale_.get(sid, 1.0)
            X_out[i] = lam * X[i]
        return X_out

    def fit_transform(self, X, y=None, domains=None):
        self.fit(X, y=y, domains=domains)
        return self.transform(X, domains)

    def __repr__(self):
        return f"URPA(split_ratio={self.split_ratio}, seed={self.seed})"
