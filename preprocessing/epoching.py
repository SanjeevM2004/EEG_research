import mne
from typing import Optional

def create_epochs(raw: mne.io.BaseRaw, tmin: float, tmax: float) -> mne.Epochs:
    """
    Create epochs from annotations in raw EEG data.
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        EEG raw data with annotations.
    tmin : float
        Start time before event (s).
    tmax : float
        End time after event (s).
    
    Returns
    -------
    epochs : mne.Epochs
    
    Mathematical Explanation
    ------------------------
    Epoching extracts segments aligned to stimulus events:
        Epoch_k = x(t) for t ∈ [event_k + tmin, event_k + tmax]
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True, verbose=False)
    return epochs
