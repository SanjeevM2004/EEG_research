import torch
from typing import Literal, Tuple

def ledoit_wolf_shrinkage(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Ledoit–Wolf shrinkage covariance for zero-mean data X.

    Parameters
    ----------
    X : torch.Tensor, shape (N, C, T)
        Zero-mean signals with N epochs, C channels, T time points.

    Returns
    -------
    C_hat : torch.Tensor, shape (N, C, C)
        Shrinkage covariance matrices.
    lam : torch.Tensor, shape (N,)
        Optimal shrinkage intensities per epoch.
    """
    N, C, T = X.shape

    # Sample covariance (N, C, C)
    S = torch.matmul(X, X.transpose(-1, -2)) / (T - 1)

    mu = torch.einsum("nii->n", S) / C  # trace per batch
    F = torch.eye(C, device=X.device, dtype=X.dtype).expand(N, C, C) * mu[:, None, None]

    # Variance term
    X2 = X**2
    var_s = torch.matmul(X2, X2.transpose(-1, -2)) / (T - 1) - S**2
    phi = var_s.sum(dim=(1, 2))

    # Distance term
    gamma = torch.norm(S - F, p="fro", dim=(1, 2))**2  # (N,)

    lam = torch.clamp(phi / (gamma * T + 1e-12), 0.0, 1.0)

    C_hat = (1 - lam)[:, None, None] * S + lam[:, None, None] * F
    return C_hat, lam


def shrink_cov(epoch_signal: torch.Tensor,
               method: Literal["ledoit_wolf", "ridge"] = "ledoit_wolf",
               eps: float = 1e-6,
               device: str = "cuda") -> torch.Tensor:
    """
    Compute SPD covariance with shrinkage for batched epochs (GPU-accelerated).

    Parameters
    ----------
    epoch_signal : torch.Tensor, shape (N, C, T)
        Batch of EEG epochs.
    method : str
        "ledoit_wolf" (default) or "ridge".
    eps : float
        Diagonal jitter for SPD.
    device : str
        Device to run on.

    Returns
    -------
    C : torch.Tensor, shape (N, C, C)
        Symmetric positive-definite covariance matrices.
    """
    X = epoch_signal.to(device)
    X = X - X.mean(dim=2, keepdim=True)  # center per channel

    if method == "ledoit_wolf":
        C, _ = ledoit_wolf_shrinkage(X)
    else:
        T = X.shape[2]
        C = torch.matmul(X, X.transpose(-1, -2)) / (T - 1)

    # Ensure symmetry & SPD
    eye = torch.eye(C.shape[1], device=device, dtype=C.dtype).expand(C.shape[0], -1, -1)
    C = 0.5 * (C + C.transpose(-1, -2)) + eps * eye
    return C
