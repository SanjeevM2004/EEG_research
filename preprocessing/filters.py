import mne
import numpy as np
from typing import Optional

def bandpass_filter(raw: mne.io.BaseRaw, l_freq: float = 1.0, h_freq: float = 80.0) -> mne.io.BaseRaw:
    """
    Apply band-pass filter to EEG data.
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw EEG recording.
    l_freq : float
        Low cutoff frequency (Hz).
    h_freq : float
        High cutoff frequency (Hz).

    Returns
    -------
    raw_filtered : mne.io.BaseRaw
        Band-pass filtered raw object.
    
    Mathematical Explanation
    ------------------------
    Filtering in EEG removes unwanted frequency components.
    A band-pass filter allows:
        x_filtered(t) = H{x(t)} 
    where H is a filter that keeps only [l_freq, h_freq].
    """
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    return raw_filtered


def notch_filter(raw: mne.io.BaseRaw, freq: float = 50.0) -> mne.io.BaseRaw:
    """
    Apply notch filter to remove line noise (e.g., 50 Hz in India/Europe).
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw EEG recording.
    freq : float
        Frequency to notch filter out.

    Returns
    -------
    raw_notched : mne.io.BaseRaw
    
    Mathematical Explanation
    ------------------------
    A notch filter removes a very narrow band around freq.
    Useful to suppress line noise:
        H(f) ≈ 0 at f = 50 Hz
        H(f) ≈ 1 elsewhere
    """
    raw_notched = raw.copy().notch_filter(freqs=[freq], verbose=False)
    return raw_notched
