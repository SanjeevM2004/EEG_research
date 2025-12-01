# ---------------------------------------------------------------
# EBA-TSLR: Riemannian Alignment + Dispersion Scaling + Eigen-Basis Alignment
# Steps:
#   (1) Per-subject Riemannian mean  Ms = mean_riemann(Xs)      (SPD)
#   (2) Global  Riemannian mean      Mg = mean_riemann({Ms})     (SPD)
#   (3) RA:      C' = Mg^{-1/2} C Mg^{-1/2}
#   (4) Scale:   C'' = (C')^(1/sigma_s)  with sigma_s from ||C'-I||_F
#   (5) Rotate:  Rs = Ug Us^T   where Mg = Ug Λg Ug^T,  Ms = Us Λs Us^T
#   (6) Tangent-space (at I) + Standardize + Logistic Regression
# ---------------------------------------------------------------

from __future__ import annotations
import numpy as np
from numpy.linalg import eigh
from scipy.linalg import fractional_matrix_power
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pyriemann.utils.mean import mean_riemann
from pyriemann.tangentspace import TangentSpace


# ========== utilities ==========

def _to_numpy(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)

def _ensure_square_2d(A, name: str) -> np.ndarray:
    A = _to_numpy(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name}: expected (c,c), got {A.shape}")
    return A

def _ensure_trials_3d(block, name: str) -> np.ndarray:
    if block is None:
        raise ValueError(f"{name}: None")
    B = _to_numpy(block)
    if B.ndim == 3:
        if B.shape[1] != B.shape[2]:
            raise ValueError(f"{name}: expected (n,c,c), got {B.shape}")
        return B
    if B.ndim == 2:
        return B[np.newaxis, :, :]
    if isinstance(block, (list, tuple)):
        mats = []
        for i, C in enumerate(block):
            C = _ensure_square_2d(C, f"{name}[{i}]")
            mats.append(C)
        out = np.stack(mats, axis=0)
        if out.shape[1] != out.shape[2]:
            raise ValueError(f"{name}: non-square matrices inside list")
        return out
    raise ValueError(f"{name}: unsupported type {type(block)}")

def _regularize_spd(M: np.ndarray, eps: float = 1e-6, name: str = "") -> np.ndarray:
    M = _ensure_square_2d(M, name)
    w, _ = eigh(M)
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        c = M.shape[0]
        # print(f"[Regularize] {name}: min(eig)={np.nanmin(w):.2e} → +{eps:g}I")
        M = M + eps * np.eye(c)
    return M

def _riemann_mean_from_block(block3d, where: str) -> np.ndarray:
    A = _ensure_trials_3d(block3d, f"{where}:trials")
    M = mean_riemann(A)
    return _regularize_spd(M, 1e-6, f"{where}:mean")


# ========== model ==========

class EBA_TSLR:
    """RA + per-subject dispersion scaling + eigen-basis alignment + TS(LR)."""

    def __init__(self, C=1.0, solver="lbfgs", max_iter=1000, verbose=False):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
        self.ts = None
        self.M_global = None
        self.U_global = None  # eigenvectors of M_global (orthonormal)

    # ---------- steps ----------

    def _riemann_align(self, covs, M_global, tag="RA"):
        C3 = _ensure_trials_3d(covs, f"{tag}:covs")
        Mg = _regularize_spd(M_global, 1e-6, f"{tag}:M_global")
        Mg_mhalf = fractional_matrix_power(Mg, -0.5)
        # plain matmul per trial (no einsum)
        return np.array([Mg_mhalf @ C @ Mg_mhalf for C in C3], dtype=np.float64)

    def _dispersion_scale(self, covs, tag="SC"):
        C3 = _ensure_trials_3d(covs, f"{tag}:covs")
        c = C3.shape[1]
        I = np.eye(c)
        diffs = [np.linalg.norm(C - I, "fro")**2 for C in C3]
        sigma = np.sqrt(np.mean(diffs)) if len(diffs) else 1.0
        if not np.isfinite(sigma) or sigma < 1e-8:
            sigma = 1.0
        if self.verbose:
            print(f"{tag}: sigma={sigma:.6f}")
        out = []
        for C in C3:
            C = _regularize_spd(C, 1e-8, f"{tag}:C")
            out.append(fractional_matrix_power(C, 1.0 / sigma))
        return np.array(out, dtype=np.float64)

    def _eigenbasis_align(self, covs, U_global, M_subject, tag="ROT"):
        C3 = _ensure_trials_3d(covs, f"{tag}:covs")
        # DO NOT regularize or eig U_global! It is an orthonormal eigenvector matrix.
        Ug = _ensure_square_2d(U_global, f"{tag}:U_global")

        Ms = _regularize_spd(M_subject, 1e-6, f"{tag}:M_subject")
        _, Us = eigh(Ms)  # eigenvectors (columns)
        R = Ug @ Us.T     # (c,c) rotation

        return np.array([R.T @ C @ R for C in C3], dtype=np.float64)

    # ---------- API ----------

    def fit(self, X_list, y_list):
        X = [_ensure_trials_3d(x, f"fit:X{i}") for i, x in enumerate(X_list)]
        y = [np.asarray(lbl, dtype=int).reshape(-1) for lbl in y_list]

        # 1) per-subject means
        subj_means = [_riemann_mean_from_block(x, f"fit:subj{i}") for i, x in enumerate(X)]
        # 2) global mean
        subj_means_3d = np.stack(subj_means, axis=0)
        M_global = _riemann_mean_from_block(subj_means_3d, "fit:global")
        self.M_global = _regularize_spd(M_global, 1e-6, "fit:M_global")
        _, U_global = eigh(self.M_global)
        self.U_global = U_global  # orthonormal, leave as-is

        # 3–5) transform all subjects
        covs_all, labels_all = [], []
        for i, (Xi, yi, Ms) in enumerate(zip(X, y, subj_means)):
            Xi_ra  = self._riemann_align(Xi, self.M_global, f"fit:subj{i}:RA")
            Xi_sc  = self._dispersion_scale(Xi_ra, f"fit:subj{i}:SC")
            Xi_rot = self._eigenbasis_align(Xi_sc, self.U_global, Ms, f"fit:subj{i}:ROT")
            if Xi_rot.shape[0] != len(yi):
                raise ValueError(f"fit: subject {i} trials/labels mismatch {Xi_rot.shape[0]} vs {len(yi)}")
            covs_all.append(Xi_rot)
            labels_all.append(yi)

        X_all = np.concatenate(covs_all, axis=0)
        y_all = np.concatenate(labels_all, axis=0)

        self.ts = TangentSpace()
        X_ts = self.ts.fit_transform(X_all)
        X_z  = self.scaler.fit_transform(X_ts)
        self.clf.fit(X_z, y_all)
        return self

    def predict(self, X_block) -> np.ndarray:
        if self.ts is None:
            raise RuntimeError("predict called before fit()")
        Xb = _ensure_trials_3d(X_block, "predict:X")
        Ms = _riemann_mean_from_block(Xb, "predict:Ms")
        X_ra  = self._riemann_align(Xb, self.M_global, "predict:RA")
        X_sc  = self._dispersion_scale(X_ra, "predict:SC")
        X_rot = self._eigenbasis_align(X_sc, self.U_global, Ms, "predict:ROT")
        X_ts  = self.ts.transform(X_rot)
        X_z   = self.scaler.transform(X_ts)
        return self.clf.predict(X_z)

    def score(self, X_block, y_true) -> float:
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = self.predict(X_block)
        return float(np.mean(y_pred == y_true))
