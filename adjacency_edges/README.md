# 🕸️ Adjacency & Edges

This directory contains utility scripts for constructing graph adjacency matrices from EEG data. These methods define the edges (connections) between EEG channels for Graph Neural Networks.

## Key Modules

| File | Description |
| :--- | :--- |
| **[`entropy_mi.py`](entropy_mi.py)** | **Mutual Information**. Computes adjacency based on the mutual information or entropy between channel signals. |
| **[`riemann.py`](riemann.py)** | **Riemannian Distance**. Defines edges based on the Riemannian distance between channel covariances. |
| **[`spearman.py`](spearman.py)** | **Spearman Correlation**. Constructs adjacency matrices using Spearman's rank correlation coefficient between time-series. |

> **Note**: These are experimental utility functions used to explore different graph construction strategies for the GADCN framework.
