import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from adjacency_edges.riemann import riemann_log_euclidean
from adjacency_edges.entropy_mi import mutual_info_adjacency_psd
from adjacency_edges.spearman import spearman_adjacency_psd


class WeightedAdjacency(nn.Module):
    """
    Learnable fusion of multiple adjacency matrices with selective affine correction.
    Vectorized version (no Python loops).
    """
    def __init__(self, num_relations: int = 3):
        super().__init__()
        self.num_relations = num_relations

        # Learnable weights
        self.alphas = nn.Parameter(torch.ones(num_relations))    # fusion weights
        self.scales = nn.Parameter(torch.ones(num_relations))    # γ_i per relation
        self.bias = nn.Parameter(torch.zeros(1))                 # β only for Riemann

    def forward(self, adj_list, riemann_index: int = 1) -> torch.Tensor:
        """
        adj_list: list of adjacency matrices [A1, A2, ..., Ak],
                  each (B, C, C).
        Returns: fused adjacency (B, C, C).
        """
        # Stack into tensor: (B, num_relations, C, C)
        A = torch.stack(adj_list, dim=1)  

        # Normalize weights: (num_relations,)
        w = torch.softmax(self.alphas, dim=0)  

        # Apply scales: (num_relations,) → (1, num_relations, 1, 1)
        g = self.scales.view(1, -1, 1, 1)

        # Apply bias only on Riemann relation
        bias = torch.zeros_like(A)
        bias[:, riemann_index, :, :] = self.bias

        # Fusion: weighted sum across relations
        fused = torch.sum(w.view(1, -1, 1, 1) * (g * A + bias), dim=1)  # (B, C, C)
        return fused


if __name__ == "__main__":
    C, T, fs = 64, 256, 160
    signal = np.random.randn(C, T)   # fake EEG (64 channels × 256 samples)

    # build adjacencies
    A_mi = torch.tensor(mutual_info_adjacency_psd(signal, fs), dtype=torch.float32).unsqueeze(0)
    A_riem = torch.tensor(riemann_log_euclidean(signal, fs), dtype=torch.float32).unsqueeze(0)
    A_spear = torch.tensor(spearman_adjacency_psd(signal, fs), dtype=torch.float32).unsqueeze(0)

    # fuse
    fuser = WeightedAdjacency(num_relations=3, riemann_index=1)
    fused_adj = fuser([A_mi, A_riem, A_spear])

    print("Mutual Info adj shape:", A_mi.shape)
    print("Riemann adj shape:", A_riem.shape)
    print("Spearman adj shape:", A_spear.shape)
    print("Fused adjacency shape:", fused_adj.shape)
    print("Fused adjacency (top-left 5x5):\n", fused_adj[0, :5, :5])
