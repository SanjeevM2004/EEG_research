import torch
from typing import Dict, Tuple

from .temporal import temporal_stats        # GPU version
from .spectral import spectral_stats        # GPU version
from .nonlinear import nonlinear_features   # GPU version
from .build_tsfel import tsfel_features     # CPU version

DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),  # clip to Nyquist
}

def minmax_scale(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scales each feature dimension into [0, 1].
    x: (N, C, d)
    """
    min_vals = x.amin(dim=(0, 1), keepdim=True)
    max_vals = x.amax(dim=(0, 1), keepdim=True)
    return (x - min_vals) / (max_vals - min_vals + eps)


def build_feature_vector(epoch_signals, fs: float,
                         bands=None, device="cpu"):
    """
    Build per-channel feature vectors z_{i,t} for one or many epochs.
    Scales each feature block to [0, 1] before concatenation.
    """
    if bands is None:
        bands = DEFAULT_BANDS.copy()

    # --- ensure shape (N, C, T) ---
    if epoch_signals.ndim == 2:
        epoch_signals = epoch_signals.unsqueeze(0)  # (1, C, T)
    N, C, T = epoch_signals.shape
    if N == 0:
        return torch.empty((0, C, 0), device=device)

    # Compute features
    temp      = temporal_stats(epoch_signals, device=device)   # (N, C, d_temp)
    spec      = spectral_stats(epoch_signals, fs, bands, device=device)  # (N, C, d_spec)
    nonlinear = nonlinear_features(epoch_signals, device=device)   # (N, C, d_nonlin)
    tsfelf    = tsfel_features(epoch_signals, fs)   # (N, C, d_tsfel) (CPU → Torch)

    # Move to device
    temp, spec, nonlinear, tsfelf = (
        temp.to(device), spec.to(device),
        nonlinear.to(device), tsfelf.to(device)
    )

    # 🔑 Normalize each block to [0, 1] separately
    temp      = minmax_scale(temp)
    spec      = minmax_scale(spec)
    nonlinear = minmax_scale(nonlinear)
    tsfelf    = minmax_scale(tsfelf)

    # Concatenate along feature dimension
    Z = torch.cat([temp, spec, nonlinear, tsfelf], dim=2)  # (N, C, d_total)
    return Z
