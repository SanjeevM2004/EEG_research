# models/riemann/rpa_dispersion_galda.py
# ---------------------------------------------------------------
# RPA Dispersion → Generalized Alignment-Aware LDA (GA-LDA)
# Projection learned once; classification = nearest mean + alignment
# ---------------------------------------------------------------

import numpy as np
from numpy.linalg import eigh, svd
from typing import List, Optional, Iterable, Dict, Any, Tuple

# ---------- SPD helpers ----------
def _sym(A): return 0.5 * (A + A.T)

def _eig_clip(C, eps=1e-10, hi=1e12):
    C = _sym(C)
    w, V = eigh(C)
    w = np.clip(np.nan_to_num(w, nan=eps), eps, hi)
    return w, V

def _project_spd(C, eps=1e-8):
    w, V = _eig_clip(C, eps)
    return V @ np.diag(w) @ V.T

def _logm_spd(C, eps=1e-10):
    w, V = _eig_clip(C, eps)
    lw = np.clip(np.log(w), -50, 50)
    return V @ np.diag(lw) @ V.T

def _invsqrtm_spd(C, eps=1e-10):
    w, V = _eig_clip(C, eps)
    iw = 1.0 / np.sqrt(w)
    return V @ np.diag(iw) @ V.T

def _sqrtm_spd(C, eps=1e-10):
    w, V = _eig_clip(C, eps)
    sw = np.sqrt(w)
    return V @ np.diag(sw) @ V.T

# ---------- RPA dispersion scaling ----------
def frob_dispersion_to_I(covs: List[np.ndarray]) -> float:
    I = np.eye(covs[0].shape[0])
    return np.mean([np.linalg.norm(C - I, "fro") ** 2 for C in covs])

def rpa_dispersion_scale(covs, subject_ids=None, ref_strategy="pooled"):
    if subject_ids is None:
        subject_ids = np.zeros(len(covs), int)
    subject_ids = np.asarray(subject_ids)
    # subject dispersions
    subj_covs = {}
    for C, sid in zip(covs, subject_ids):
        subj_covs.setdefault(sid, []).append(_project_spd(C))
    subj_sigma = {sid: np.sqrt(frob_dispersion_to_I(lst)) for sid, lst in subj_covs.items()}
    if ref_strategy == "pooled":
        sigma_ref = np.sqrt(frob_dispersion_to_I(covs))
    else:
        sigma_ref = np.median(list(subj_sigma.values()))
    scaled = []
    for C, sid in zip(covs, subject_ids):
        a = (sigma_ref / max(subj_sigma[sid], 1e-4)) ** 2
        a = np.clip(a, 1e-3, 1e3)
        scaled.append(_project_spd(a * C))
    return scaled

# ---------- GA-LDA features ----------
def _spd_to_vecs(C: np.ndarray) -> np.ndarray:
    """Half-vectorize log-domain SPD (sqrt(2) on off-diagonals)."""
    C = _sym(C)
    idx = np.triu_indices_from(C)
    v = np.sqrt(2) * C[idx]
    v[idx[0] == idx[1]] /= np.sqrt(2)
    return v

# ---------- Alignment distance (principal angles) ----------
def _alignment_distance(C: np.ndarray, G: np.ndarray, r: Optional[int] = None) -> float:
    """
    Sum of principal angles between eigen-bases of C and G (in radians).
    If r is provided, use top-r eigenvectors; otherwise use full basis.
    """
    _, VC = eigh(_sym(C))
    _, VG = eigh(_sym(G))
    if r is not None:
        VC = VC[:, -r:]
        VG = VG[:, -r:]
    # principal angles via SVD of VC^T VG
    M = VC.T @ VG
    # singular values are cos(theta), clip to [0,1]
    s = np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0)
    theta = np.arccos(s)
    return float(np.sum(theta))

# ---------- Model ----------
class RPA_Disperse_GALDA:
    """
    Pipeline:
      (1) Dispersion-scale subjects (RPA)
      (2) Log-domain flatten features
      (3) Solve Fisher (LDA) once → projection W (best space)
      (4) Classify by nearest class mean in projected space + alignment penalty

    Parameters
    ----------
    n_components : int
        Projection dimension. Use None for K-1 (chosen at fit-time).
    reg_lambda : float
        Ridge on Sw in the Fisher step.
    ref_strategy : {'pooled','median'}
        Reference for RPA scaling.
    w_proj : float
        Weight for distance to class centroid in projected space.
    w_align : float
        Weight for eigen-basis misalignment penalty.
    align_top_r : Optional[int]
        If set, compute principal angles using top-r eigenvectors only.
    subject_ids : Optional[Iterable[Any]]
        Default subject ids for scaling when not passed to predict/transform.
    """

    def __init__(self,
                 n_components: Optional[int] = 2,
                 reg_lambda: float = 1e-5,
                 ref_strategy: str = "pooled",
                 w_proj: float = 1.0,
                 w_align: float = 0.3,
                 align_top_r: Optional[int] = None,
                 subject_ids: Optional[Iterable[Any]] = None):
        self.n_components = n_components
        self.reg_lambda = reg_lambda
        self.ref_strategy = ref_strategy
        self.w_proj = w_proj
        self.w_align = w_align
        self.align_top_r = align_top_r
        self.subject_ids = subject_ids

        # learned state
        self.W_ = None                # [D, m]
        self.classes_ = None
        self.mu_k_ = None             # class centroids in projected space
        self.G_k_ = None              # class mean SPD matrices (for alignment)
        self.D_feat_ = None           # input feature dimension (after half-vec)

    # -------- utils --------
    def _build_features(self, covs: List[np.ndarray]) -> np.ndarray:
        vecs = []
        for C in covs:
            Cn = _logm_spd(_project_spd(C))
            vecs.append(_spd_to_vecs(Cn))
        X = np.vstack(vecs)
        if self.D_feat_ is None:
            self.D_feat_ = X.shape[1]
        return X

    def _scatter_matrices(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        classes = np.unique(y)
        mu = X.mean(axis=0, keepdims=True)
        Sw = np.zeros((d, d))
        Sb = np.zeros((d, d))
        for k in classes:
            Xk = X[y == k]
            muk = Xk.mean(axis=0, keepdims=True)
            Sw += (Xk - muk).T @ (Xk - muk)
            diff = (muk - mu).T @ (muk - mu)
            Sb += Xk.shape[0] * diff
        return _sym(Sw), _sym(Sb)

    def _fisher_projection(self, X: np.ndarray, y: np.ndarray, m: int) -> np.ndarray:
        Sw, Sb = self._scatter_matrices(X, y)
        d = Sw.shape[0]
        A = _sym(Sb)
        B = _sym(Sw) + self.reg_lambda * np.eye(d)
        wB, VB = eigh(B)
        wB = np.clip(wB, 1e-12, 1e12)
        B_inv_sqrt = VB @ np.diag(1.0 / np.sqrt(wB)) @ VB.T
        M = _sym(B_inv_sqrt @ A @ B_inv_sqrt)
        w, U = eigh(M)
        idx = np.argsort(w)[::-1]
        U = U[:, idx]
        return B_inv_sqrt @ U[:, :m]  # [d, m]

    # -------- API --------
    def fit(self, covs: List[np.ndarray], y: np.ndarray, subject_ids=None):
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        sid = subject_ids if subject_ids is not None else self.subject_ids

        # 1) RPA scaling
        covs_scaled = rpa_dispersion_scale(covs, sid, ref_strategy=self.ref_strategy)

        # 2) features
        X = self._build_features(covs_scaled)

        # 3) Fisher projection
        K = len(self.classes_)
        D = X.shape[1]
        m = self.n_components if self.n_components is not None else min(D, max(1, K - 1))
        self.W_ = self._fisher_projection(X, y, m)

        # 4) Class centroids in projected space
        Xproj = X @ self.W_
        self.mu_k_ = {int(k): Xproj[y == k].mean(axis=0) for k in self.classes_}

        # 5) Class mean SPDs for alignment
        self.G_k_ = {}
        for k in self.classes_:
            Ck = [C for C, yy in zip(covs_scaled, y) if yy == k]
            self.G_k_[int(k)] = _project_spd(np.mean(Ck, axis=0))

        return self

    def transform(self, covs: List[np.ndarray], subject_ids=None):
        covs_scaled = rpa_dispersion_scale(covs, subject_ids, ref_strategy=self.ref_strategy)
        X = self._build_features(covs_scaled)
        return X @ self.W_

    def _decision_scores(self, covs: List[np.ndarray], subject_ids=None) -> np.ndarray:
        """
        Combined distance D_k = w_proj * ||x - mu_k||_2  +  w_align * principal-angle(C, G_k)
        Returns [n, K] matrix of distances.
        """
        covs_scaled = rpa_dispersion_scale(covs, subject_ids, ref_strategy=self.ref_strategy)
        X = self._build_features(covs_scaled)
        Xp = X @ self.W_

        Ks = sorted(self.classes_.tolist())
        M = np.stack([self.mu_k_[k] for k in Ks])  # [K, m]

        # projected-space distances
        d_proj = np.sqrt(((Xp[:, None, :] - M[None, :, :]) ** 2).sum(axis=2))  # [n, K]

        # alignment distances
        d_align = np.zeros_like(d_proj)
        for j, k in enumerate(Ks):
            Gk = self.G_k_[k]
            # compute alignment per sample
            for i, C in enumerate(covs_scaled):
                d_align[i, j] = _alignment_distance(C, Gk, r=self.align_top_r)

        return self.w_proj * d_proj + self.w_align * d_align  # [n, K]

    def predict(self, covs: List[np.ndarray], subject_ids=None):
        D = self._decision_scores(covs, subject_ids)  # smaller is better
        Ks = sorted(self.classes_.tolist())
        idx = D.argmin(axis=1)
        return np.array([Ks[i] for i in idx])

    def predict_proba(self, covs: List[np.ndarray], subject_ids=None):
        # Convert distances to pseudo-probabilities via softmax over -D
        D = self._decision_scores(covs, subject_ids)
        logits = -D
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        return P

# ---------------- Quick test ----------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, C, K = 60, 6, 3
    y = np.repeat(np.arange(K), n // K)

    def rand_spd(C):
        A = rng.standard_normal((C, C))
        return _project_spd(A @ A.T + 0.2 * np.eye(C))

    covs = [rand_spd(C) for _ in range(n)]
    subj = np.repeat([0, 1, 2], n // 3)

    clf = RPA_Disperse_GALDA(n_components=None, w_proj=1.0, w_align=1.0, align_top_r=3, ref_strategy="median")
    clf.fit(covs, y, subject_ids=subj)
    preds = clf.predict(covs, subject_ids=subj)
    print("Train acc (dummy):", np.mean(preds == y))
