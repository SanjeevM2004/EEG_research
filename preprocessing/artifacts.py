import mne
import numpy as np
from typing import Optional

import mne
from typing import Optional

def run_ica(
    raw: mne.io.BaseRaw, 
    n_components: Optional[int] = None, 
    random_state: int = 42
) -> mne.io.BaseRaw:
    """
    Apply Independent Component Analysis (ICA) to clean EEG data.
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        EEG raw data.
    n_components : int, optional
        Number of ICA components (default = n_channels).
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    raw_clean : mne.io.BaseRaw
        ICA-cleaned EEG data.
    
    Notes
    -----
    - This function does NOT use EOG/ECG channels.
    - If you want to manually inspect artifacts:
        * ica.plot_components()
        * ica.plot_sources(raw)
    """
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        max_iter="auto"
    )
    ica.fit(raw)

    # No automatic EOG/ECG removal
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    return raw_clean

def reject_bad_epochs(epochs: mne.Epochs, threshold_uV: float = 300.0, drop: bool = True):
    data = epochs.get_data()
    max_per_epoch = data.max(axis=(1, 2))
    print("Max amplitude per epoch:", max_per_epoch[:10])

    threshold_V = threshold_uV * 1e-6
    bad_idx = [i for i, val in enumerate(max_per_epoch) if val > threshold_V]

    print(f"Threshold = {threshold_uV} µV ({threshold_V} V). Bad epochs: {bad_idx}")

    if drop:
        epochs_clean = epochs.copy().drop(bad_idx)
        print(f"Epochs before: {len(epochs)}, after rejection: {len(epochs_clean)}")
        return epochs_clean
    else:
        return epochs, bad_idx
