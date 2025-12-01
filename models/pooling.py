# pooling.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Attention-based Global Pooling
# -----------------------------
class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over nodes.
    Input:  H (B, C, d)
    Output: g (B, d)
    """
    def __init__(self, d_in: int):
        super().__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.Tanh(),
            nn.Linear(d_in, 1)
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B, C, d)
        scores = self.att_mlp(H).squeeze(-1)     # (B, C)
        weights = F.softmax(scores, dim=1)       # (B, C)
        g = torch.einsum("bc, bcd -> bd", weights, H)
        return g                                  # (B, d)
