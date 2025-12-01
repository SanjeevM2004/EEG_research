# 🧠 Models Directory

This directory contains the deep learning architectures and graph neural networks used in the GADCN framework.

## Key Files

| File | Description |
| :--- | :--- |
| **[`neuronet.py`](neuronet.py)** | **NeuroGraphNet (GADCN)**. The main model architecture. Integrates LSTM (temporal), GCN (spatial), and Domain Adversarial Learning. |
| **[`gcn.py`](gcn.py)** | **Graph Convolutional Network**. Standard GCN layer implementation $H' = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H W)$. |
| **[`rgcn.py`](rgcn.py)** | **Relational / Shallow GCN**. Implements shallow graph attention mechanisms and transformer blocks for EEG graphs. |
| **[`dann.py`](dann.py)** | **Domain Adversarial Neural Network**. Gradient Reversal Layer (GRL) and domain discriminator for subject-invariant learning. |
| **[`mae.py`](mae.py)** | **Masked Autoencoder**. Self-supervised learning component for pre-training on EEG data. |
| **[`lstm.py`](lstm.py)** | **Channel LSTM**. LSTM wrapper applied independently to each EEG channel. |

## Subdirectories

- **[`riemann/`](riemann/README.md)**: Contains specialized Riemannian geometry algorithms and classifiers (MDM, TSLR, CovCNN).
