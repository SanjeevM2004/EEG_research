# 🏗️ Data Construction

This directory handles the loading, formatting, and dataset creation for PyTorch training.

## Key Modules

| File | Description |
| :--- | :--- |
| **[`EEGCovDataset.py`](EEGCovDataset.py)** | **Base Dataset Class**. A PyTorch `Dataset` wrapper that returns (Signal, Covariance, Label) tuples. |
| **[`EEGCovDataset_BCIIV.py`](EEGCovDataset_BCIIV.py)** | **BCI IV 2a Loader**. Specialized loader for the BCI Competition IV 2a dataset (Motor Imagery). Handles 4-class classification. |
| **[`data_cref_physio.py`](data_cref_physio.py)** | **Physionet Loader**. Loads data from the Physionet MI dataset (100+ subjects). Computes subject-specific reference covariances ($C_{ref}$) for whitening. |
| **[`eeg_dataloader.py`](eeg_dataloader.py)** | **Data Loading Utilities**. Helper functions to create PyTorch `DataLoader` instances with batching and shuffling. |
| **[`bci_cref.py`](bci_cref.py)** | **Reference Computation**. Pre-computes the geometric mean of covariance matrices for each subject in the BCI dataset. |
