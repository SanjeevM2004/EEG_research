"""
riemann.py
A unified, educational Riemannian-manifold utilities module for EEG covariance
learning on the SPD manifold. This file is designed to be both practical and
readable for learning, with detailed docstrings explaining what each method
does and how to use it.

What is SPD? In EEG, we often summarize an epoch by its channel covariance
matrix C ∈ R^{C×C}. Such covariances are Symmetric Positive Definite (SPD), and
the set of all SPD matrices forms a curved Riemannian manifold. Geometry-aware
methods (distances, means, mappings) on this manifold typically outperform
naïve Euclidean treatments of matrices.

Included features:
- Torch-only, batched, GPU-friendly SPD ops (autograd-safe)
- Optional PyRiemann-backed ops on CPU (if installed)
- Distances: AIRM (affine-invariant) and Log-Euclidean
- Means: Karcher/Fréchet (AIRM) and Log-Euclidean
- Tangent-space mappings (log/exp at a reference) and vectorization
- Riemannian gradient step (natural gradient under AIRM)
- EEG → covariance helper
- Classifiers built in the Riemannian space:
    * MDM (Minimum Distance to Mean)
    * FgMDM (pyriemann-only, filter-geodesic variant)
    * Tangent-space + linear / MLP

Device routing:
- On GPU: torch backend is used for speed and autograd.
- On CPU: if pyriemann is installed, its well-tested CPU routines are used;
  otherwise we fall back to the torch backend.
"""

import torch
from torch import Tensor
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod

# ============================================================
# ---------- Core SPD helpers (Torch, batched, GPU) ----------
# ============================================================

def _symmetrize(X: Tensor) -> Tensor:
    return 0.5 * (X + X.transpose(-1, -2))


def ensure_spd(C: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Ensure SPD-ness by symmetrizing and shifting eigenvalues a bit.
    C: (..., d, d)
    """
    C = _symmetrize(C)
    eye = torch.eye(C.shape[-1], device=C.device, dtype=C.dtype)
    return C + eps * eye.expand_as(C)


def _eigh_spd(C: Tensor) -> Tuple[Tensor, Tensor]:
    # Works batched
    w, U = torch.linalg.eigh(C)
    w = torch.clamp(w, min=1e-12)
    return w, U


def spd_log(C: Tensor) -> Tensor:
    w, U = _eigh_spd(C)
    return U @ torch.diag_embed(torch.log(w)) @ U.transpose(-1, -2)


def spd_exp(S: Tensor) -> Tensor:
    w, U = torch.linalg.eigh(S)
    return U @ torch.diag_embed(torch.exp(w)) @ U.transpose(-1, -2)


def spd_sqrtm(C: Tensor) -> Tensor:
    w, U = _eigh_spd(C)
    return U @ torch.diag_embed(torch.sqrt(w)) @ U.transpose(-1, -2)


def spd_invsqrtm(C: Tensor) -> Tensor:
    w, U = _eigh_spd(C)
    return U @ torch.diag_embed(torch.rsqrt(w)) @ U.transpose(-1, -2)


def airm_distance(A: Tensor, B: Tensor) -> Tensor:
    """
    Affine-Invariant Riemannian Metric (AIRM) distance.
    Intuition: whiten B by A (A^{-1/2} B A^{-1/2}), take matrix-log, then its
    Frobenius norm. Invariant to any invertible linear re-referencing of EEG.

    Parameters
    - A, B: SPD matrices, broadcastable shape (..., d, d)

    Returns
    - distances: shape (...,)
    """
    A = ensure_spd(A)
    B = ensure_spd(B)
    A_is = spd_invsqrtm(A)
    M = A_is @ B @ A_is
    L = spd_log(M)
    return torch.linalg.norm(L, ord="fro", dim=(-2, -1))


def logeuclidean_distance(A: Tensor, B: Tensor) -> Tensor:
    """
    Log-Euclidean distance: treat log(C) as Euclidean points.
    Faster than AIRM; often a strong baseline with closed-form mean.

    Returns the Frobenius norm of log(A) - log(B).
    """
    LA = spd_log(ensure_spd(A))
    LB = spd_log(ensure_spd(B))
    return torch.linalg.norm(LA - LB, ord="fro", dim=(-2, -1))


def log_map(X: Tensor, P: Tensor) -> Tensor:
    """
    Log_P(X) = P^{1/2} log( P^{-1/2} X P^{-1/2} ) P^{1/2}
    """
    P = ensure_spd(P)
    P_s = spd_sqrtm(P)
    P_is = spd_invsqrtm(P)
    M = P_is @ X @ P_is
    return P_s @ spd_log(M) @ P_s


def exp_map(V: Tensor, P: Tensor) -> Tensor:
    """
    Exp_P(V) = P^{1/2} exp( P^{-1/2} V P^{-1/2} ) P^{1/2}
    """
    P = ensure_spd(P)
    P_s = spd_sqrtm(P)
    P_is = spd_invsqrtm(P)
    M = P_is @ V @ P_is
    return ensure_spd(P_s @ spd_exp(M) @ P_s)


def tangent_space_map(C: Tensor, ref: Tensor) -> Tensor:
    """
    Map SPD points to the tangent space at a reference SPD matrix `ref` using
    the AIRM log map. Outputs are symmetric matrices (tangent vectors).

    Shapes
    - C: (B, d, d)
    - ref: (d, d)
    - Returns: (B, d, d)
    """
    ref = ensure_spd(ref)
    ref_s = spd_sqrtm(ref)
    ref_is = spd_invsqrtm(ref)
    M = ref_is @ ensure_spd(C) @ ref_is
    return ref_s @ spd_log(M) @ ref_s


def inverse_tangent_space_map(T: Tensor, ref: Tensor) -> Tensor:
    ref = ensure_spd(ref)
    ref_s = spd_sqrtm(ref)
    ref_is = spd_invsqrtm(ref)
    M = ref_is @ T @ ref_is
    return ensure_spd(ref_s @ spd_exp(M) @ ref_s)


def parallel_transport(V: Tensor, A: Tensor, B: Tensor) -> Tensor:
    """
    Parallel transport of V ∈ T_A(M) to T_B(M) on SPD with AIRM.
    """
    A = ensure_spd(A)
    B = ensure_spd(B)
    B_s = spd_sqrtm(B)
    B_is = spd_invsqrtm(B)
    inner = B_is @ A @ B_is
    T = B_s @ spd_sqrtm(inner) @ B_is
    return _symmetrize(T @ V @ T.transpose(-1, -2))


def upper_triangle_vectorize(S: Tensor, keep_diag: bool = True) -> Tensor:
    """
    Vectorize upper triangle of SPD matrices.
    S: (..., d, d)
    returns: (..., d*(d+1)/2) if keep_diag else ...
    """
    C = S.shape[-1]
    iu = torch.triu_indices(C, C, offset=0 if keep_diag else 1, device=S.device)
    return S[..., iu[0], iu[1]]


def upper_triangle_invert(v: Tensor, C: int, keep_diag: bool = True) -> Tensor:
    M = torch.zeros(*v.shape[:-1], C, C, device=v.device, dtype=v.dtype)
    iu = torch.triu_indices(C, C, offset=0 if keep_diag else 1, device=v.device)
    M[..., iu[0], iu[1]] = v
    return _symmetrize(M)


def riemann_log_euclidean_adjacency(signals: Tensor) -> Tensor:
    """
    Build an adjacency-like matrix per batch using the absolute value of the
    log of the covariance (log-Euclidean). Diagonal is zeroed.

    signals: (B, C, T)  -> returns (B, C, C)
    """
    B, C, T = signals.shape
    X = signals - signals.mean(dim=2, keepdim=True)
    cov = (X @ X.transpose(1, 2)) / (T - 1)
    L = spd_log(ensure_spd(cov)).abs()
    return L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1))


# ============================================================
# ---------- EEG → Covariance convenience --------------------
# ============================================================

def eeg_to_cov(signals: Tensor, shrink: float = 0.0, eps: float = 1e-6) -> Tensor:
    """
    Convert batched EEG epochs to SPD covariance matrices.
    - Mean-centers each channel, computes unbiased covariance, enforces SPD.
    - Optional shrinkage pulls toward identity to improve conditioning.

    Parameters
    - signals: (B, C, T)
    - shrink: scalar in [0,1]; 0 = sample covariance; 1 = identity
    - eps: jitter added to diagonal for numerical stability

    Returns: (B, C, C)
    """
    B, C, T = signals.shape
    X = signals - signals.mean(dim=2, keepdim=True)
    cov = (X @ X.transpose(1, 2)) / (T - 1)
    cov = ensure_spd(cov, eps=eps)
    if shrink > 0:
        eye = torch.eye(C, device=cov.device, dtype=cov.dtype)
        cov = (1 - shrink) * cov + shrink * eye.expand_as(cov)
    return cov


# ============================================================
# ---------- Riemann Ops Abstraction -------------------------
# ============================================================

class RiemannOps(ABC):
    """
    Abstract interface for core SPD-manifold operations.
    Concrete implementations:
    - TorchRiemannOps: pure torch, batched, GPU-capable
    - PyRiemannOps: CPU path, wraps pyriemann where appropriate
    """
    @abstractmethod
    def mean(self, covs: Tensor, metric: str = "airm") -> Tensor:
        ...

    @abstractmethod
    def distance(self, A: Tensor, B: Tensor, metric: str = "airm") -> Tensor:
        ...

    @abstractmethod
    def tangent(self, covs: Tensor, ref: Tensor) -> Tensor:
        ...

    @abstractmethod
    def inv_tangent(self, tangents: Tensor, ref: Tensor) -> Tensor:
        ...

    @abstractmethod
    def gd_step(self, X: Tensor, euclid_grad: Tensor, step_size: float) -> Tensor:
        ...


class TorchRiemannOps(RiemannOps):
    """
    Torch backend implementing batched SPD operations suitable for GPU.
    - mean(metric="airm"): iterative Karcher mean (AIRM) with LE init
    - mean(metric="logeuclidean"): closed-form LE mean
    - distance: AIRM or Log-Euclidean
    - tangent/inv_tangent: AIRM log/exp maps at a reference
    - gd_step: natural gradient step under AIRM metric
    """
    def mean(self, covs: Tensor, metric: str = "airm") -> Tensor:
        covs = ensure_spd(covs)
        if metric == "airm":
            # Karcher mean init with LE mean
            G = spd_exp(spd_log(covs).mean(dim=0))
            for _ in range(50):
                G_s = spd_sqrtm(G)
                G_is = spd_invsqrtm(G)
                M = G_is @ covs @ G_is  # (B, d, d)
                Delta = spd_log(M).mean(dim=0)
                if torch.linalg.norm(Delta, ord="fro") < 1e-6:
                    break
                G = ensure_spd(G_s @ spd_exp(Delta) @ G_s)
            return G
        elif metric in ("le", "logeuclidean"):
            return spd_exp(spd_log(covs).mean(dim=0))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def distance(self, A: Tensor, B: Tensor, metric: str = "airm") -> Tensor:
        if metric == "airm":
            return airm_distance(A, B)
        elif metric in ("le", "logeuclidean"):
            return logeuclidean_distance(A, B)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def tangent(self, covs: Tensor, ref: Tensor) -> Tensor:
        return tangent_space_map(covs, ref)

    def inv_tangent(self, tangents: Tensor, ref: Tensor) -> Tensor:
        return inverse_tangent_space_map(tangents, ref)

    def gd_step(self, X: Tensor, euclid_grad: Tensor, step_size: float) -> Tensor:
        X = ensure_spd(X)
        # natural gradient on SPD (AIRM): grad_t = X * grad_euclid * X
        grad_t = _symmetrize(X @ euclid_grad @ X)
        return exp_map(-step_size * grad_t, X)


# ------------------------------------------------------------
# Try to load pyriemann
# ------------------------------------------------------------
_PYR_AVAILABLE = False
try:
    from pyriemann.classification import MDM as PR_MDM, FgMDM as PR_FgMDM
    from pyriemann.tangentspace import TangentSpace as PR_TangentSpace
    from pyriemann.utils.mean import mean_riemann as pr_mean_riemann, mean_logeuclid as pr_mean_le
    from pyriemann.utils.distance import distance_riemann as pr_dist_riem, distance_logeuclid as pr_dist_le
    _PYR_AVAILABLE = True
except Exception:
    _PYR_AVAILABLE = False


class PyRiemannOps(RiemannOps):
    """
    CPU-only ops that delegate to pyriemann/numpy where possible.
    Recommended on CPU for robustness and parity with the PyRiemann literature.
    For tangent/inv_tangent we reuse the torch implementation to keep API
    identical and avoid redundant conversions.
    """
    def __init__(self):
        if not _PYR_AVAILABLE:
            raise ImportError("pyriemann not installed. pip install pyriemann")

    @staticmethod
    def _to_np(x: Tensor):
        return x.detach().cpu().numpy()

    @staticmethod
    def _to_tensor(x, like: Tensor) -> Tensor:
        return torch.from_numpy(x).to(device=like.device, dtype=like.dtype)

    def mean(self, covs: Tensor, metric: str = "airm") -> Tensor:
        X = self._to_np(covs)
        if metric == "airm":
            m = pr_mean_riemann(X)
        elif metric in ("le", "logeuclidean"):
            m = pr_mean_le(X)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return self._to_tensor(m, like=covs)

    def distance(self, A: Tensor, B: Tensor, metric: str = "airm") -> Tensor:
        A_, B_ = self._to_np(A), self._to_np(B)
        if metric == "airm":
            d = pr_dist_riem(A_, B_)
        elif metric in ("le", "logeuclidean"):
            d = pr_dist_le(A_, B_)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return self._to_tensor(d, like=A)

    def tangent(self, covs: Tensor, ref: Tensor) -> Tensor:
        # for consistency, just reuse torch tangent
        return tangent_space_map(covs, ref)

    def inv_tangent(self, tangents: Tensor, ref: Tensor) -> Tensor:
        return inverse_tangent_space_map(tangents, ref)

    def gd_step(self, X: Tensor, euclid_grad: Tensor, step_size: float) -> Tensor:
        X = ensure_spd(X)
        grad_t = _symmetrize(X @ euclid_grad @ X)
        return exp_map(-step_size * grad_t, X)


def get_ops(device: str = "cpu", prefer_pyr: bool = True) -> RiemannOps:
    """
    Select an ops backend based on device.
    - GPU (e.g., "cuda:0"): uses TorchRiemannOps (fast, autograd-safe)
    - CPU: uses PyRiemannOps if installed, else TorchRiemannOps
    """
    if device.startswith("cuda") or device.startswith("gpu"):
        return TorchRiemannOps()
    if prefer_pyr and _PYR_AVAILABLE:
        return PyRiemannOps()
    return TorchRiemannOps()


# ============================================================
# ---------- Tangent Space Transform -------------------------
# ============================================================

class TangentSpaceTransform:
    """
    Shared transform: SPD -> tangent (at learned reference mean) -> vector.
    - fit: computes class-agnostic reference mean using selected metric/device
    - transform: applies log map at the reference and vectorizes the triangle
    """
    def __init__(self, metric: str = "airm", device: str = "cpu"):
        self.metric = metric
        self.device = device
        self.ops = get_ops(device)
        self._ref: Optional[Tensor] = None

    @property
    def ref_(self) -> Tensor:
        if self._ref is None:
            raise RuntimeError("TangentSpaceTransform not fitted")
        return self._ref

    def fit(self, covs: Tensor):
        covs = ensure_spd(covs)
        self._ref = self.ops.mean(covs, metric=self.metric)
        return self

    def transform(self, covs: Tensor, keep_diag: bool = True) -> Tensor:
        T = self.ops.tangent(ensure_spd(covs), self.ref_)
        return upper_triangle_vectorize(T, keep_diag=keep_diag)


# ============================================================
# ---------- Classifiers -------------------------------------
# ============================================================

class MDMClassifier:
    """
    Minimum Distance to Mean classifier on SPD.
    Training:
      - For each class, compute the Riemannian mean of its covariances.
    Inference:
      - Assign to the class whose mean is closest under the chosen metric.
    Device behavior:
      - device="cuda" → torch backend; device="cpu" → pyriemann if available.
    """
    def __init__(self, metric: str = "airm", device: str = "cpu"):
        self.metric = metric
        self.ops = get_ops(device)
        self._classes: Optional[Tensor] = None
        self._means: Optional[Tensor] = None

    def fit(self, covs: Tensor, labels: Tensor):
        covs = ensure_spd(covs)
        classes = torch.unique(labels)
        means: List[Tensor] = []
        for c in classes:
            Cc = covs[labels == c]
            means.append(self.ops.mean(Cc, metric=self.metric))
        self._classes = classes
        self._means = torch.stack(means, dim=0)
        return self

    def predict(self, covs: Tensor) -> Tensor:
        assert self._means is not None and self._classes is not None
        A = covs[:, None, :, :]
        B = self._means[None, :, :, :]
        d = self.ops.distance(A, B, metric=self.metric)
        idx = torch.argmin(d, dim=1)
        return self._classes[idx]


class FgMDMClassifier:
    """
    Filter-geodesic MDM (CPU only, requires pyriemann).
    Applies bank(s) of temporal filters, computes covariances, and averages
    in Riemannian space before distance-to-mean classification. Popular for
    ERP-like paradigms where distinct frequency bands matter.
    """
    def __init__(self, **fg_kwargs):
        if not _PYR_AVAILABLE:
            raise ImportError("pyriemann required for FgMDM")
        self._fg = PR_FgMDM(**fg_kwargs)

    def fit(self, covs: Tensor, labels: Tensor):
        self._fg.fit(covs.detach().cpu().numpy(), labels.detach().cpu().numpy())
        return self

    def predict(self, covs: Tensor) -> Tensor:
        y = self._fg.predict(covs.detach().cpu().numpy())
        return torch.from_numpy(y).to(device=covs.device, dtype=torch.long)


class TangentSpaceClassifier:
    """
    Tangent space + linear classifier.
    Pipeline:
      1) Fit reference mean on SPD (AIRM or LE) using device’s ops
      2) Map covariances to tangent space at the reference
      3) Vectorize upper-triangular entries
      4) Train simple linear classifier (torch Linear or sklearn LR)
    Device behavior:
      - GPU: pure torch
      - CPU: prefers pyriemann TangentSpace + sklearn LogisticRegression;
        falls back to torch Linear if sklearn/pyriemann absent.
    """
    def __init__(
        self,
        metric: str = "airm",
        device: str = "cpu",
        epochs: int = 50,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
    ):
        self.metric = metric
        self.device = device
        self.ops = get_ops(device)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self._ts_transform: Optional[TangentSpaceTransform] = None
        self._clf = None
        self._use_torch = (device.startswith("cuda") or device == "gpu")

    def fit(self, covs: Tensor, labels: Tensor):
        covs = ensure_spd(covs)
        self._ts_transform = TangentSpaceTransform(self.metric, self.device).fit(covs)

        if self._use_torch:
            X = self._ts_transform.transform(covs, keep_diag=True)
            num_classes = int(labels.max().item() + 1)
            feat_dim = X.shape[-1]
            model = torch.nn.Linear(feat_dim, num_classes).to(covs.device)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            loss_fn = torch.nn.CrossEntropyLoss()
            model.train()
            for _ in range(self.epochs):
                opt.zero_grad(set_to_none=True)
                logits = model(X)
                loss = loss_fn(logits, labels.long())
                loss.backward()
                opt.step()
            self._clf = model.eval()
        else:
            # CPU path: try sklearn + pyriemann TS
            try:
                from sklearn.linear_model import LogisticRegression
                if _PYR_AVAILABLE:
                    ts = PR_TangentSpace(metric=("riemann" if self.metric == "airm" else "logeuclid"))
                    ts.fit(covs.detach().cpu().numpy())
                    X = ts.transform(covs.detach().cpu().numpy())
                    self._sk_ts = ts
                else:
                    X = self._ts_transform.transform(covs, keep_diag=True).detach().cpu().numpy()
                clf = LogisticRegression(max_iter=2000)
                clf.fit(X, labels.detach().cpu().numpy())
                self._clf = clf
            except Exception:
                # fallback: torch linear on CPU
                X = self._ts_transform.transform(covs, keep_diag=True)
                num_classes = int(labels.max().item() + 1)
                feat_dim = X.shape[-1]
                model = torch.nn.Linear(feat_dim, num_classes)
                opt = torch.optim.Adam(model.parameters(), lr=self.lr)
                loss_fn = torch.nn.CrossEntropyLoss()
                for _ in range(self.epochs):
                    opt.zero_grad(set_to_none=True)
                    logits = model(X)
                    loss = loss_fn(logits, labels.long())
                    loss.backward()
                    opt.step()
                self._clf = model.eval()

        return self

    def predict(self, covs: Tensor) -> Tensor:
        assert self._ts_transform is not None and self._clf is not None
        covs = ensure_spd(covs)

        if self._use_torch or isinstance(self._clf, torch.nn.Module):
            X = self._ts_transform.transform(covs, keep_diag=True)
            with torch.no_grad():
                logits = self._clf(X)
                return torch.argmax(logits, dim=1)
        else:
            # sklearn path
            if hasattr(self, "_sk_ts"):
                X = self._sk_ts.transform(covs.detach().cpu().numpy())
            else:
                X = self._ts_transform.transform(covs, keep_diag=True).detach().cpu().numpy()
            y = self._clf.predict(X)
            return torch.from_numpy(y).to(device=covs.device, dtype=torch.long)


class RiemannLogReg:
    """
    Tangent space + torch Logistic Regression (single-layer softmax).
    Equivalent to TangentSpaceClassifier’s GPU branch but kept explicit for
    clarity and easy customization (regularization, schedules, etc.).
    """
    def __init__(self, metric: str = "airm", device: str = "cpu", lr: float = 1e-2, epochs: int = 100):
        self.metric = metric
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.ts: Optional[TangentSpaceTransform] = None
        self.model: Optional[torch.nn.Module] = None

    def fit(self, covs: Tensor, labels: Tensor):
        covs = ensure_spd(covs)
        self.ts = TangentSpaceTransform(self.metric, self.device).fit(covs)
        X = self.ts.transform(covs, keep_diag=True)
        num_classes = int(labels.max().item() + 1)
        feat_dim = X.shape[-1]
        model = torch.nn.Linear(feat_dim, num_classes).to(covs.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, labels.long())
            loss.backward()
            opt.step()
        self.model = model.eval()
        return self

    def predict(self, covs: Tensor) -> Tensor:
        assert self.ts is not None and self.model is not None
        X = self.ts.transform(ensure_spd(covs), keep_diag=True)
        with torch.no_grad():
            logits = self.model(X)
            return torch.argmax(logits, dim=1)


class RiemannMLP:
    """
    Tangent space + small MLP.
    When linear separability in the tangent space is insufficient, this offers
    a lightweight nonlinear head while keeping manifold-aware feature mapping.
    """
    def __init__(self, metric: str = "airm", device: str = "cpu",
                 hidden_dim: int = 128, epochs: int = 100, lr: float = 1e-3):
        self.metric = metric
        self.device = device
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.ts: Optional[TangentSpaceTransform] = None
        self.model: Optional[torch.nn.Module] = None

    def fit(self, covs: Tensor, labels: Tensor):
        covs = ensure_spd(covs)
        self.ts = TangentSpaceTransform(self.metric, self.device).fit(covs)
        X = self.ts.transform(covs, keep_diag=True)
        num_classes = int(labels.max().item() + 1)
        feat_dim = X.shape[-1]

        class MLP(torch.nn.Module):
            def __init__(self, in_dim, h_dim, out_dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, h_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(h_dim, out_dim),
                )

            def forward(self, x):
                return self.net(x)

        model = MLP(feat_dim, self.hidden_dim, num_classes).to(covs.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        model.train()
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, labels.long())
            loss.backward()
            opt.step()

        self.model = model.eval()
        return self

    def predict(self, covs: Tensor) -> Tensor:
        assert self.ts is not None and self.model is not None
        X = self.ts.transform(ensure_spd(covs), keep_diag=True)
        with torch.no_grad():
            logits = self.model(X)
            return torch.argmax(logits, dim=1)


# ============================================================
# ---------- Convenience API (module-level) ------------------
# ============================================================

def riemann_mean(covs: Tensor, metric: str = "airm", device: str = "cpu") -> Tensor:
    """
    Convenience wrapper to compute a Riemannian mean using the appropriate
    backend for the given device.
    """


def riemann_distance(A: Tensor, B: Tensor, metric: str = "airm", device: str = "cpu") -> Tensor:
    """
    Convenience wrapper to compute distances under the chosen metric/device.
    """


def riemann_gd_step(X: Tensor, euclid_grad: Tensor, step_size: float, device: str = "cpu") -> Tensor:
    """
    One Riemannian gradient step on the SPD manifold under AIRM.
    - Projects Euclidean gradient to the tangent space at X (natural gradient)
    - Retracts back to the manifold using the exponential map.
    Use inside custom optimization loops whose parameters live on SPD.
    """
    return get_ops(device).gd_step(X, euclid_grad, step_size)
