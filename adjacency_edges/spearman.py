import numpy as np
import torch
from scipy.stats import spearmanr
from feature_extraction.spectral import welch_psd_batched   # <- reuse your existing PSD function

# -----------------------------
# Spearman Correlation adjacency
# -----------------------------
def spearman_adjacency_psd(signals: torch.Tensor, fs: float, fmax: float = None) -> torch.Tensor:
    """
    Batched Spearman correlation adjacency.
    signals: (B, C, T)
    Returns: (B, C, C)
    """
    freqs, psd = welch_psd_batched(signals, fs=fs)  # (F,), (B, C, F)

    if fmax is not None:
        keep = freqs <= fmax
        psd = psd[:, :, keep]

    # Convert to ranks along frequency axis
    ranks = psd.argsort(dim=2).argsort(dim=2).float()  # (B, C, F)
    ranks = ranks - ranks.mean(dim=2, keepdim=True)
    ranks = ranks / (ranks.norm(dim=2, keepdim=True) + 1e-12)

    # Spearman correlation = cosine similarity of ranks
    A = torch.matmul(ranks, ranks.transpose(1, 2)) / ranks.shape[2]
    return A.abs()
