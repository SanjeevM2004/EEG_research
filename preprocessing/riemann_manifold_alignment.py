import os, glob, gc, torch, mne
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np, scipy.linalg
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann, mean_euclid


def riemann_alignment_trace(covs):
    """Subject-wise Riemannian Alignment with trace normalization."""
    C = torch.stack(covs).numpy().astype(np.float64)
    G = mean_riemann(C)
    G_sqrt = scipy.linalg.sqrtm(G).real
    G_inv_sqrt = scipy.linalg.inv(G_sqrt)
    aligned = []
    for c in C:
        a = G_inv_sqrt @ c @ G_inv_sqrt.T
        a = a / np.trace(a)
        aligned.append(torch.from_numpy(a.astype(np.float32)))
    return aligned


def euclidean_alignment_trace(covs):
    """Subject-wise Euclidean Alignment with trace normalization."""
    C = torch.stack(covs).numpy().astype(np.float64)
    G = mean_euclid(C)
    G_sqrt = scipy.linalg.sqrtm(G).real
    G_inv_sqrt = scipy.linalg.inv(G_sqrt)
    aligned = []
    for c in C:
        a = G_inv_sqrt @ c @ G_inv_sqrt.T
        a = a / np.trace(a)
        aligned.append(torch.from_numpy(a.astype(np.float32)))
    return aligned


def logeuclidean_alignment_trace(covs):
    """Subject-wise Log-Euclidean Alignment with trace normalization."""
    C = torch.stack(covs).numpy().astype(np.float64)
    logs = np.array([scipy.linalg.logm(c) for c in C])
    mean_log = np.mean(logs, axis=0)
    G = scipy.linalg.expm(mean_log).real
    G_sqrt = scipy.linalg.sqrtm(G).real
    G_inv_sqrt = scipy.linalg.inv(G_sqrt)
    aligned = []
    for c in C:
        a = G_inv_sqrt @ c @ G_inv_sqrt.T
        a = a / np.trace(a)
        aligned.append(torch.from_numpy(a.astype(np.float32)))
    return aligned
