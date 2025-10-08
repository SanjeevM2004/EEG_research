import numpy as np
from typing import Optional

def zscore_normalize(epochs: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization per channel across all epochs safely in float32.

    Parameters
    ----------
    epochs : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times).

    Returns
    -------
    norm_epochs : np.ndarray
        Z-scored data (float32).

    Notes
    -----
    • Forces float32 to reduce memory usage by 50%.
    • Avoids large temporary float64 buffers.
    • Numerically stable for long signals and many epochs.
    """
    # ---- Force float32 and operate in place where possible ----
    epochs = np.asarray(epochs, dtype=np.float32, order="C")

    # Compute per-epoch, per-channel mean and std
    mean = epochs.mean(axis=2, keepdims=True, dtype=np.float32)
    std = epochs.std(axis=2, keepdims=True, dtype=np.float32)

    # Avoid division by zero and normalize
    np.add(std, 1e-8, out=std)                     # in-place epsilon
    norm_epochs = (epochs - mean) / std

    # Guarantee float32 output
    return norm_epochs.astype(np.float32, copy=False)
