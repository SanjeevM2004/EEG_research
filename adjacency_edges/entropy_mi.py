import torch
import torch.nn.functional as F
from feature_extraction.spectral import welch_psd_batched
# -----------------------------
# Mutual Information adjacency
# -----------------------------
def mutual_info_adjacency_psd(signals: torch.Tensor, fs: float, fmax: float = None) -> torch.Tensor:
    """
    Batched MI adjacency using PSDs.
    signals: (B, C, T)  torch.Tensor
    Returns: (B, C, C)  torch.Tensor
    """
    freqs, psd = welch_psd_batched(signals, fs=fs)  # (F,), (B, C, F)

    if fmax is not None:
        keep = freqs <= fmax
        psd = psd[:, :, keep]

    # Normalize into distributions per channel
    psd = torch.clamp(psd, min=1e-24)
    psd = psd / psd.sum(dim=2, keepdim=True)

    # Entropy H(X) for each channel
    H = -(psd * torch.log(psd)).sum(dim=2)  # (B, C)

    # Compute joint entropies
    B, C, F = psd.shape
    adj = torch.zeros((B, C, C), device=signals.device, dtype=signals.dtype)

    for i in range(C):
        for j in range(i+1, C):
            Pxy = psd[:, i, :, None] * psd[:, j, None, :]  # (B, F, F)
            Pxy = Pxy / Pxy.sum(dim=(1,2), keepdim=True)
            Hxy = -(Pxy * torch.log(Pxy + 1e-24)).sum(dim=(1,2))  # (B,)
            mi = H[:, i] + H[:, j] - Hxy
            adj[:, i, j] = adj[:, j, i] = torch.clamp(mi, min=0.0)
    return adj