# ---------------------------------------------------------------
# ULambdaAlignTSLR: U-alignment + Λ-equalization + Tangent Space LR
# ---------------------------------------------------------------
# Ensures: per-subject mean -> same eigenvectors (U_G) and same diagonal (Λ_G).
# Pipeline:
#   1) M_s (per subject) and M_G (global) via mean_riemann
#   2) Eigendecompose: M_s = U_s Λ_s U_s^T; M_G = U_G Λ_G U_G^T
#   3) Rotate:    R_s = U_G U_s^T;  C' = R_s C R_s^T
#   4) Color/diag: S_s = U_G Λ_G^{1/2} Λ_s^{-1/2} U_G^T;  C'' = S_s C' S_s
#   5) (optional) Final whitening to identity with M_G^{-1/2} before TS
#   6) TangentSpace + StandardScaler + LogisticRegression
# ---------------------------------------------------------------

from __future__ import annotations
import numpy as np
from numpy.linalg import eigh, svd
from scipy.linalg import fractional_matrix_power
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pyriemann.utils.mean import mean_riemann
from pyriemann.tangentspace import TangentSpace


class ULambdaAlignTSLR:
    def __init__(self,
                 project_at_identity: bool = True,
                 C: float = 1.0,
                 solver: str = "lbfgs",
                 max_iter: int = 1000,
                 verbose: bool = False):
        self.project_at_identity = project_at_identity
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
        self.ts = None

        # learned state
        self.M_global = None      # SPD
        self.U_global = None      # orthonormal eigenvectors of M_global
        self.Lam_global = None    # eigenvalues (diag) of M_global
        self.R_map = {}           # sid -> rotation R_s
        self.S_map = {}           # sid -> coloring S_s

    # --------------- utilities ---------------

    @staticmethod
    def _to_np(a):
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        return np.asarray(a, dtype=np.float64)

    @classmethod
    def _ensure_trials_3d(cls, block, name="X"):
        B = cls._to_np(block)
        if B.ndim == 3:
            if B.shape[1] != B.shape[2]:
                raise ValueError(f"{name}: expected (n,c,c), got {B.shape}")
            return B
        if B.ndim == 2:
            return B[np.newaxis, :, :]
        if isinstance(block, (list, tuple)):
            mats = []
            for i, C in enumerate(block):
                C = cls._to_np(C)
                if C.ndim != 2 or C.shape[0] != C.shape[1]:
                    raise ValueError(f"{name}[{i}]: expected (c,c), got {C.shape}")
                mats.append(C)
            return np.stack(mats, axis=0)
        raise ValueError(f"{name}: unsupported type {type(block)}")

    @staticmethod
    def _regularize_spd(M, eps=1e-6):
        M = np.asarray(M, dtype=np.float64)
        w, _ = eigh(M)
        if np.any(~np.isfinite(w)) or np.any(w <= 0):
            M = M + eps * np.eye(M.shape[0])
        return M

    @staticmethod
    def _eig_sorted(C):
        w, U = eigh(C)
        idx = np.argsort(w)[::-1]
        return U[:, idx], w[idx]

    @classmethod
    def _subject_mean(cls, X_3d):
        X = cls._ensure_trials_3d(X_3d, "subject")
        M = mean_riemann(X)
        return cls._regularize_spd(M, 1e-6)

    @staticmethod
    def _apply_congruence_block(X_3d, A):
        X = np.asarray(X_3d, dtype=np.float64)
        return np.array([A @ C @ A.T for C in X], dtype=np.float64)

    # --------------- grouping ---------------

    def _group_by_subject(self, X, y, groups):
        # Already per-subject lists
        if isinstance(X, (list, tuple)):
            if not isinstance(y, (list, tuple)):
                raise ValueError("When X is a list of per-subject blocks, y must be a list as well.")
            X_list = [self._ensure_trials_3d(x, f"fit:X_subj{i}") for i, x in enumerate(X)]
            y_list = [np.asarray(yi, dtype=int).reshape(-1) for yi in y]
            for i, (Xi, yi) in enumerate(zip(X_list, y_list)):
                if Xi.shape[0] != len(yi):
                    raise ValueError(f"Subject {i}: trials/labels mismatch {Xi.shape[0]} vs {len(yi)}")
            subj_ids = np.arange(len(X_list))
            return X_list, y_list, subj_ids

        # Stacked trials with groups
        X = self._ensure_trials_3d(X, "fit:X")
        y = np.asarray(y, dtype=int).reshape(-1)
        if groups is None:
            raise ValueError("groups must be provided when X is a single 3D array.")
        groups = np.asarray(groups)
        if not (len(X) == len(y) == len(groups)):
            raise ValueError("Lengths of X, y, groups must match.")

        subj_ids = np.unique(groups)
        X_list, y_list = [], []
        for sid in subj_ids:
            idx = (groups == sid)
            X_list.append(X[idx])
            y_list.append(y[idx])
        return X_list, y_list, subj_ids

    # --------------- core steps ---------------

    def _compute_global_basis(self, Ms_list):
        # Global mean & eigen-decomposition
        M_g = mean_riemann(np.stack(Ms_list, axis=0))
        M_g = self._regularize_spd(M_g, 1e-6)
        U_g, lam_g = self._eig_sorted(M_g)
        return M_g, U_g, lam_g

    @staticmethod
    def _rotation_to_global(U_s, U_g):
        # Map subject basis to global basis
        A = U_g @ U_s.T
        U, _, Vt = svd(A)
        R = U @ Vt
        # enforce det=+1
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R

    @staticmethod
    def _build_coloring(U_g, lam_s, lam_g):
        # S_s = U_g Λ_g^{1/2} Λ_s^{-1/2} U_g^T
        inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(lam_s, 1e-12, None)))
        sqrt_g   = np.diag(np.sqrt(np.clip(lam_g, 1e-12, None)))
        return U_g @ (sqrt_g @ inv_sqrt) @ U_g.T

    # --------------- public API ---------------

    def fit(self, X, y, groups=None):
        """
        Fit with per-subject U-alignment (rotation) and Λ-equalization (coloring),
        then train TS + LR for action classification.
        """
        per_subj_covs, per_subj_labels, subj_ids = self._group_by_subject(X, y, groups)

        # Per-subject means (SPD) and eigen-decompositions
        Ms_list, Us_list, lam_s_list = [], [], []
        for covs in per_subj_covs:
            M_s = self._subject_mean(covs)
            U_s, lam_s = self._eig_sorted(M_s)
            Ms_list.append(M_s)
            Us_list.append(U_s)
            lam_s_list.append(lam_s)

        # Global mean & basis
        self.M_global, self.U_global, self.Lam_global = self._compute_global_basis(Ms_list)

        # Build rotations & colorings per subject, transform all trials
        self.R_map.clear()
        self.S_map.clear()
        transformed_blocks, labels_blocks = [], []

        # Optional final whitening to identity for TS at I
        Wg = fractional_matrix_power(self.M_global, -0.5) if self.project_at_identity else None

        for sid, covs, labels, U_s, lam_s in zip(subj_ids, per_subj_covs, per_subj_labels, Us_list, lam_s_list):
            R_s = self._rotation_to_global(U_s, self.U_global)              # rotate to U_G
            S_s = self._build_coloring(self.U_global, lam_s, self.Lam_global) # match Λ to Λ_G

            self.R_map[sid] = R_s
            self.S_map[sid] = S_s

            C_rot = self._apply_congruence_block(covs, R_s)    # C' = R C R^T
            C_col = self._apply_congruence_block(C_rot, S_s)   # C'' = S C' S^T

            if self.project_at_identity:
                C_fin = self._apply_congruence_block(C_col, Wg)  # bring all to identity mean
            else:
                C_fin = C_col

            transformed_blocks.append(C_fin)
            labels_blocks.append(np.asarray(labels, dtype=int))

            if self.verbose:
                # Check subject mean after transform
                M_chk = mean_riemann(C_col)
                devU = np.linalg.norm(self.U_global.T @ self.U_global - np.eye(self.U_global.shape[0]))
                devM = np.linalg.norm(M_chk - self.M_global, 'fro')
                print(f"Subject {sid}: ||M''-M_G||_F={devM:.2e}, ||U_G^T U_G - I||_F={devU:.2e}")

        X_all = np.concatenate(transformed_blocks, axis=0)
        y_all = np.concatenate(labels_blocks, axis=0)

        # Tangent space + LR
        self.ts = TangentSpace()
        X_ts = self.ts.fit_transform(X_all)
        X_z  = self.scaler.fit_transform(X_ts)
        self.clf.fit(X_z, y_all)
        return self

    def predict(self, X_block):
        """
        Predict for a new subject block:
          - compute its M_s, eigendecompose for U_s, Λ_s
          - build R_s and S_s using (U_G, Λ_G) from training
          - transform trials and classify
        """
        if self.M_global is None:
            raise RuntimeError("Model not fitted yet.")

        Xb = self._ensure_trials_3d(X_block, "predict:X")
        M_s = self._subject_mean(Xb)
        U_s, lam_s = self._eig_sorted(M_s)

        R_s = self._rotation_to_global(U_s, self.U_global)
        S_s = self._build_coloring(self.U_global, lam_s, self.Lam_global)

        C_rot = self._apply_congruence_block(Xb, R_s)
        C_col = self._apply_congruence_block(C_rot, S_s)

        if self.project_at_identity:
            Wg = fractional_matrix_power(self.M_global, -0.5)
            C_fin = self._apply_congruence_block(C_col, Wg)
        else:
            C_fin = C_col

        X_ts = self.ts.transform(C_fin)
        X_z  = self.scaler.transform(X_ts)
        return self.clf.predict(X_z)

    def score(self, X_block, y_true):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = self.predict(X_block)
        return float(np.mean(y_pred == y_true))
