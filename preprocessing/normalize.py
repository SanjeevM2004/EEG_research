import numpy as np
from typing import Optional

def zscore_normalize(epochs: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization per channel across all epochs.
    
    Parameters
    ----------
    epochs : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times).
    
    Returns
    -------
    norm_epochs : np.ndarray
        Z-scored data.
    
    Mathematical Explanation
    ------------------------
    For each channel c:
        z = (x - μ_c) / σ_c
    where μ_c = mean over all epochs and time,
          σ_c = std over all epochs and time.
    """
    mean = epochs.mean(axis=(0, 2), keepdims=True)
    std = epochs.std(axis=(0, 2), keepdims=True)
    norm_epochs = (epochs - mean) / (std + 1e-8)
    return norm_epochs
