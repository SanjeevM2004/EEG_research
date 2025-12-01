# 🧹 Preprocessing

This directory contains utilities for cleaning and preparing EEG data before analysis.

## Key Modules

| File | Description |
| :--- | :--- |
| **[`filters.py`](filters.py)** | **Signal Filtering**. Implements Bandpass (e.g., 4-38Hz), Notch (50/60Hz), and Lowpass/Highpass filters using `scipy.signal`. |
| **[`artifacts.py`](artifacts.py)** | **Artifact Removal**. Methods to detect and remove eye blinks (EOG), muscle noise (EMG), and other artifacts, potentially using ICA (Independent Component Analysis). |
| **[`normalize.py`](normalize.py)** | **Normalization**. Standardizes EEG signals (Z-score normalization) per channel or per trial to ensure consistent scaling. |
| **[`epoching.py`](epoching.py)** | **Epoch Extraction**. Segments continuous EEG data into trials based on event markers (e.g., motor imagery cues). |
| **[`riemann_manifold_alignment.py`](riemann_manifold_alignment.py)** | **Manifold Alignment**. Aligns covariance matrices from different subjects to a common reference space using Riemannian operations. |
