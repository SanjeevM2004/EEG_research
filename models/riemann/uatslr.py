# ---------------------------------------------------------------
# UAlignTSLR: Eigen-Basis (U) Alignment + Tangent Space Logistic Regression
# ---------------------------------------------------------------
# What it does
#   • Per subject: compute SPD mean of covariances, eigendecompose to get U_s
#   • Global:      mean of subject means, eigendecompose to get U_g
#   • Rotation:    find R_s that maps U_s → U_g (Procrustes by default)
#   • Apply R_s to every trial of that subject: C' = R_s^T C R_s
#   • Classify:    TangentSpace + StandardScaler + LogisticRegression
#
# Fit interfaces
#   fit(X_list, y_list)          # X_list = [ (n_i, c, c), ... ] per subject
#   fit(X_3d, y, groups=...)     # X_3d   = (N, c, c); groups = subject id per trial
#
# Predict interfaces
#   predict(X_3d)                # new subject trials (n, c, c)
#   predict(X_3d, group_id=...)  # if you want to cache/use its rotation mapping
#
# Requires: numpy, scipy, scikit-learn, pyriemann
# ---------------------------------------------------------------

from __future__ import annotations
import numpy as np
from numpy.linalg import eigh, svd
from scipy.linalg import fractional_matrix_power
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pyriemann.utils.mean import mean_riemann
from pyriemann.tangentspace import TangentSpace

try:
    from scipy.optimize import linear_sum_assignment  # only used if method='pair'
    _HAS_HUNGARIAN = True
except Exception:
    _HAS_HUNGARIAN = False


class UAlignTSLR:
    """
    Eigen-basis alignment (using only U) + TS(LogReg).

    Parameters
    ----------
    method : {'procrustes', 'pair'}, optional
        How to compute per-subject rotation R_s s.t. R_s @ U_s ≈ U_g.
        - 'procrustes' (default): Orthogonal Procrustes (fast & robust)
        - 'pair'      : Column pairing + sign fix (needs scipy.optimize)
    ts_at_identity : bool, optional
        If True (default), project covariances in tangent space at I.
        If False, project at the (pyriemann) default; for most cases True is fine.
    C : float
        LogisticRegression C parameter.
    solver : str
        LogisticRegression solver.
    max_iter : int
        LogisticRegression max_iter.
    """

    def __init__(self,
                 method: str = "procrustes",
                 ts_at_identity: bool = True,
                 C: float = 1.0,
                 solver: str = "lbfgs",
                 max_iter: int = 1000):
        if method not in ("procrustes", "pair"):
            raise ValueError("method must be 'procrustes' or 'pair'")
        if method == "pair" and not _HAS_HUNGARIAN:
            raise ImportError("method='pair' requires scipy.optimize.linear_sum_assignment")

        self.method = method
        self.ts_at_identity = ts_at_identity
        self.C = C
        self.solver = solver
        self.max_iter = max_iter

        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
        self.ts = None

        # learned state
        self.U_global = None        # (c, c)
        self.M_global = None        # (c, c)
        self.subject_rotations = {} # sid -> (c, c)

    # ---------- utilities ----------

    @staticmethod
    def _to_numpy(a):
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        return np.asarray(a, dtype=np.float64)

    @staticmethod
    def _ensure_trials_3d(block, name="X"):
        """Coerce to (n, c, c)."""
        B = UAlignTSLR._to_numpy(block)
        if B.ndim == 3:
            if B.shape[1] != B.shape[2]:
                raise ValueError(f"{name}: expected (n,c,c), got {B.shape}")
            return B
        if B.ndim == 2:
            return B[np.newaxis, :, :]
        if isinstance(block, (list, tuple)):
            mats = []
            for i, C in enumerate(block):
                C = UAlignTSLR._to_numpy(C)
                if C.ndim != 2 or C.shape[0] != C.shape[1]:
                    raise ValueError(f"{name}[{i}]: expected (c,c), got {C.shape}")
                mats.append(C)
            out = np.stack(mats, axis=0)
            return out
        raise ValueError(f"{name}: unsupported type {type(block)}")

    @staticmethod
    def _regularize_spd(M, eps=1e-6):
        """Ensure SPD by adding eps*I if eigenvalues ≤ 0 or not finite."""
        M = UAlignTSLR._to_numpy(M)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"regularize_spd: expected (c,c), got {M.shape}")
        w, _ = eigh(M)
        if np.any(~np.isfinite(w)) or np.any(w <= 0):
            M = M + eps * np.eye(M.shape[0])
        return M

    @staticmethod
    def _eig_sorted(C):
        """Eigen-decompose SPD, return (U_sorted, w_sorted desc)."""
        w, U = eigh(C)
        idx = np.argsort(w)[::-1]
        return U[:, idx], w[idx]

    @staticmethod
    def _procrustes_rotation(U_s, U_g):
        """R in O(d) minimizing ||R U_s - U_g||_F."""
        A = U_g @ U_s.T
        U, _, Vt = svd(A)
        R = U @ Vt
        # enforce det=+1 (proper rotation)
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R

    @staticmethod
    def _pairwise_rotation(U_s, U_g):
        """
        Column pairing (Hungarian) + sign fix.
        R = U_g (P D) U_s^T.
        """
        S = np.abs(U_g.T @ U_s)  # (c,c)
        cost = 1.0 - S
        row_ind, col_ind = linear_sum_assignment(cost)
        c = U_s.shape[0]
        P = np.zeros((c, c))
        P[row_ind, col_ind] = 1.0
        # choose signs so that inner products are positive
        US_perm = U_s @ P.T
        signs = np.sign(np.sum(U_g * US_perm, axis=0))
        signs[signs == 0] = 1.0
        D = np.diag(signs)
        R = U_g @ (P @ D) @ U_s.T
        return R

    def _rotation(self, U_s, U_g):
        if self.method == "procrustes":
            return self._procrustes_rotation(U_s, U_g)
        return self._pairwise_rotation(U_s, U_g)

    # ---------- core steps ----------

    @staticmethod
    def _subject_mean(covs_3d):
        """Riemannian mean of a subject's trials (SPD)."""
        X = UAlignTSLR._ensure_trials_3d(covs_3d, "subject")
        M = mean_riemann(X)
        return UAlignTSLR._regularize_spd(M, 1e-6)

    @staticmethod
    def _global_mean(subject_means):
        """Riemannian mean of subject means (SPD)."""
        M = mean_riemann(np.stack(subject_means, axis=0))
        return UAlignTSLR._regularize_spd(M, 1e-6)

    @staticmethod
    def _apply_rotation_block(X_3d, R):
        """Return [R^T C R for C in X_3d]."""
        X = UAlignTSLR._ensure_trials_3d(X_3d, "apply_rotation")
        return np.array([R.T @ C @ R for C in X], dtype=np.float64)

    # ---------- public API ----------

    def fit(self, X, y, groups=None):
        """
        Fit the model.

        Parameters
        ----------
        X : list of (n_i, c, c) arrays  OR  (N, c, c) array
            Covariances per subject or all trials combined.
        y : list of 1D arrays  OR  (N,) array
            Labels aligned with X.
        groups : (N,) array-like, optional
            Subject IDs per trial (required if X is a single 3D array).
        """
        # Normalize inputs into per-subject lists
        per_subj_covs, per_subj_labels, subj_ids = self._group_by_subject(X, y, groups)

        # Per-subject means & bases
        Ms_list, Us_list = [], []
        for covs in per_subj_covs:
            M_s = self._subject_mean(covs)
            U_s, _ = self._eig_sorted(M_s)
            Ms_list.append(M_s)
            Us_list.append(U_s)

        # Global mean & basis
        self.M_global = self._global_mean(Ms_list)
        self.U_global, _ = self._eig_sorted(self.M_global)

        # Compute subject rotations & rotate all trials
        rotated_blocks, labels_blocks = [], []
        self.subject_rotations = {}
        for sid, covs, labels, U_s in zip(subj_ids, per_subj_covs, per_subj_labels, Us_list):
            R_s = self._rotation(U_s, self.U_global)
            self.subject_rotations[sid] = R_s
            rotated_blocks.append(self._apply_rotation_block(covs, R_s))
            labels_blocks.append(np.asarray(labels, dtype=int))

        X_rot = np.concatenate(rotated_blocks, axis=0)
        y_all = np.concatenate(labels_blocks, axis=0)

        # Tangent space + LR
        # (pyriemann TangentSpace defaults to reference=identity; good after rotations)
        self.ts = TangentSpace()
        X_ts = self.ts.fit_transform(X_rot) if self.ts_at_identity else self.ts.fit_transform(X_rot)
        X_z = self.scaler.fit_transform(X_ts)
        self.clf.fit(X_z, y_all)
        return self

    def predict(self, X_block, group_id=None):
        """
        Predict labels for a new subject block.

        Parameters
        ----------
        X_block : (n, c, c) array or list of (c,c)
        group_id : optional subject id; if provided and known from fit(),
                   uses its stored rotation R_s; otherwise computes rotation
                   from this block's mean.
        """
        Xb = self._ensure_trials_3d(X_block, "predict:X")
        if group_id is not None and group_id in self.subject_rotations:
            R = self.subject_rotations[group_id]
        else:
            # compute rotation on-the-fly from this subject's mean
            M_s = self._subject_mean(Xb)
            U_s, _ = self._eig_sorted(M_s)
            R = self._rotation(U_s, self.U_global)

        X_rot = self._apply_rotation_block(Xb, R)
        X_ts = self.ts.transform(X_rot)
        X_z = self.scaler.transform(X_ts)
        return self.clf.predict(X_z)

    def score(self, X_block, y_true, group_id=None):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = self.predict(X_block, group_id=group_id)
        return float(np.mean(y_pred == y_true))

    # ---------- helpers for grouping ----------

    def _group_by_subject(self, X, y, groups):
        """
        Return (per_subj_covs, per_subj_labels, subj_ids)
        """
        # Case 1: already lists per subject
        if isinstance(X, (list, tuple)):
            if not isinstance(y, (list, tuple)):
                raise ValueError("When X is a list of per-subject blocks, y must be a list too.")
            per_subj_covs = [self._ensure_trials_3d(x, f"fit:X_subj{i}") for i, x in enumerate(X)]
            per_subj_labels = [np.asarray(yi, dtype=int).reshape(-1) for yi in y]
            subj_ids = np.arange(len(per_subj_covs))
            # sanity
            for i, (Xi, yi) in enumerate(zip(per_subj_covs, per_subj_labels)):
                if Xi.shape[0] != len(yi):
                    raise ValueError(f"Subject {i}: trials/labels mismatch {Xi.shape[0]} vs {len(yi)}")
            return per_subj_covs, per_subj_labels, subj_ids

        # Case 2: single 3D with groups
        X = self._ensure_trials_3d(X, "fit:X")
        y = np.asarray(y, dtype=int).reshape(-1)
        if groups is None:
            raise ValueError("groups must be provided when X is a single 3D array.")
        groups = np.asarray(groups)

        if not (len(X) == len(y) == len(groups)):
            raise ValueError("Lengths of X, y, groups must match.")

        subj_ids = np.unique(groups)
        per_subj_covs, per_subj_labels = [], []
        for sid in subj_ids:
            idx = (groups == sid)
            per_subj_covs.append(X[idx])
            per_subj_labels.append(y[idx])
        return per_subj_covs, per_subj_labels, subj_ids
