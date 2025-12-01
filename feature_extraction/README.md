# 📊 Feature Extraction

This directory provides tools to extract meaningful features from raw EEG time-series data.

## Key Modules

| File | Description |
| :--- | :--- |
| **[`covariance.py`](covariance.py)** | **Covariance Estimation**. Computes Sample Covariance, Ledoit-Wolf shrinkage, or OAS covariance matrices. Essential for Riemannian geometry methods. |
| **[`spectral.py`](spectral.py)** | **Spectral Features**. Computes Power Spectral Density (PSD) using Welch's method or FFT. Extracts band power (Alpha, Beta, Theta, etc.). |
| **[`temporal.py`](temporal.py)** | **Temporal Features**. Extracts statistics like Mean, Variance, Skewness, Kurtosis, and Hjorth parameters from the time domain. |
| **[`nonlinear.py`](nonlinear.py)** | **Non-linear Features**. Computes Approximate Entropy, Sample Entropy, and Fractal Dimension. |
| **[`build_tsfel.py`](build_tsfel.py)** | **TSFEL Integration**. Wrapper to extract a comprehensive set of features using the `tsfel` library. |
