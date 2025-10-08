import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.res_proj = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, A):
        X = self.lin(H)                           # (B, C, d_out)
        out = torch.einsum("bij,bjd->bid", A, X)  # (B, C, d_out)
        res = self.res_proj(H)                    # project if needed
        out = self.norm(out + res)
        return self.dropout(F.gelu(out))
    
class GCN(nn.Module):
    def __init__(self, d_in: int, d_hidden: int,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GCNLayer(d_in if i == 0 else d_hidden, d_hidden, dropout=dropout)
            for i in range(num_layers)
        ])
    def forward(self, H, A):
        """
        H: (B, C, d_in) - node embeddings
        A: (B, C, C)    - adjacency
        """
        for layer in self.layers:
            H = layer(H, A)

        return H  # (B, C, d_hidden)
