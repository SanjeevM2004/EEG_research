import numpy as np
import torch
from scipy.linalg import logm
from scipy.signal import butter, filtfilt

# -----------------------------
# Riemann Log-Euclidean adjacency
# -----------------------------
def riemann_log_euclidean(signals: torch.Tensor, fs: float, fmax: float = None) -> torch.Tensor:
    """
    Batched log-Euclidean adjacency.
    signals: (B, C, T)
    Returns: (B, C, C)
    """
    B, C, T = signals.shape

    # Mean-center
    X = signals - signals.mean(dim=2, keepdim=True)

    # Covariance per batch
    cov = torch.matmul(X, X.transpose(1,2)) / (T - 1)  # (B, C, C)

    # Matrix logarithm (approx via eigen decomposition)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # safe for symmetric
    eigvals = torch.clamp(eigvals, min=1e-12)
    log_cov = eigvecs @ torch.diag_embed(torch.log(eigvals)) @ eigvecs.transpose(-2, -1)  # (B, C, C)

    adj = log_cov.abs()
    adj = adj - torch.diag_embed(torch.diagonal(adj, dim1=1, dim2=2))  # zero diag
    return adj

