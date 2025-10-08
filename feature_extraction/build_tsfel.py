'''import tsfel
import numpy as np
import torch

# Load TSFEL config once
TSFEL_CFG = tsfel.get_features_by_domain()

def tsfel_features(epoch_signals, fs):
    """
    epoch_signals: torch.Tensor (N, C, T)
    returns: torch.Tensor (N, C, d_tsfel)
    """
    feats = []
    N, C, T = epoch_signals.shape
    for n in range(N):
        ch_feats = []
        for c in range(C):
            sig = epoch_signals[n, c].cpu().numpy()
            f = tsfel.time_series_features_extractor(TSFEL_CFG, sig, fs=fs)
            # select first 100 (or do PCA later)
            ch_feats.append(f.values.flatten())
        feats.append(np.stack(ch_feats))
    return torch.tensor(np.stack(feats), dtype=torch.float32)

'''

# build_tsfel.py
import tsfel
import numpy as np
import torch
from joblib import Parallel, delayed

# Preload TSFEL config once
TSFEL_CFG = tsfel.get_features_by_domain()

# Optional: keep only a subset (~65 features) to reduce cost
# TSFEL_CFG = {k: v for k, v in TSFEL_CFG.items() if k in ["statistical", "temporal"]}


def _extract_channel(sig, fs):
    """Helper: extract features for one channel (1D numpy)."""
    f = tsfel.time_series_features_extractor(
        TSFEL_CFG, sig, fs=fs, verbose=0
    )
    return f.values.flatten()


def tsfel_features(epoch_signals: torch.Tensor, fs: float, n_jobs: int = -1):
    """
    epoch_signals: torch.Tensor (N, C, T)
    fs: sampling frequency
    n_jobs: number of parallel workers (default = all cores)
    returns: torch.Tensor (N, C, d_tsfel)
    """
    N, C, T = epoch_signals.shape
    feats = []

    for n in range(N):
        # Parallelize across channels in this epoch
        sigs = [epoch_signals[n, c].cpu().numpy() for c in range(C)]
        ch_feats = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_extract_channel)(sig, fs) for sig in sigs
        )
        feats.append(np.stack(ch_feats))  # (C, d)

    return torch.tensor(np.stack(feats), dtype=torch.float32)
