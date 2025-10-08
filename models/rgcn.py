import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_attention_mask_from_adj(A: torch.Tensor, attn_bias: float = 0.0):
    """
    Convert adjacency matrix to attention mask and bias.
    """
    B, C, _ = A.shape
    I = torch.eye(C, device=A.device, dtype=A.dtype).unsqueeze(0).expand_as(A)
    A_allow = (A + I) > 0
    mask = A_allow.unsqueeze(1)  # (B,1,C,C)

    if attn_bias != 0.0:
        bias = attn_bias * A.unsqueeze(1)
    else:
        bias = torch.zeros_like(A).unsqueeze(1)
    return mask, bias

class ShallowGraphAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.dk = d_out
        self.q_proj = nn.Linear(d_in, d_out, bias=False)
        self.k_proj = nn.Linear(d_in, d_out, bias=False)
        self.v_proj = nn.Linear(d_in, d_out, bias=False)
        self.o_proj = nn.Linear(d_out, d_out, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor):
        """
        X: (B, C, d_in)
        mask: (B,1,C,C)
        bias: (B,1,C,C)
        """
        Q = self.q_proj(X)  # (B,C,d_out)
        K = self.k_proj(X)  # (B,C,d_out)
        V = self.v_proj(X)  # (B,C,d_out)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dk)
        logits = logits.unsqueeze(1)  # (B,1,C,C)

        logits = logits.masked_fill(~mask, -1e9) + bias
        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)

        Y = torch.matmul(attn, V.unsqueeze(1)).squeeze(1)  # (B,C,d_out)
        Y = self.o_proj(Y)
        return Y, attn.squeeze(1)

class ShallowGraphBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 128, attn_dropout: float = 0.1, ffn_dropout: float = 0.1, attn_bias_kappa: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ShallowGraphAttention(d_model, d_model, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(ffn_dropout)
        self.attn_bias_kappa = attn_bias_kappa

    def forward(self, X, A):
        mask, bias = build_attention_mask_from_adj(A, attn_bias=self.attn_bias_kappa)
        Xn = self.norm1(X)
        Y, _ = self.attn(Xn, mask, bias)
        X = X + self.dropout(Y)

        Xn = self.norm2(X)
        Z = self.ffn(Xn)
        X = X + self.dropout(Z)
        return X

class ShallowGraphTransformer(nn.Module):
    def __init__(self, d_in: int, d_model: int = 128, num_layers: int = 2, d_ff: int = 256, attn_dropout: float = 0.1, ffn_dropout: float = 0.1, attn_bias_kappa: float = 0.0, project_in: bool = True, num_relations: int = 3):
        super().__init__()
        self.project_in = nn.Linear(d_in, d_model) if project_in or d_in != d_model else nn.Identity()
        self.blocks = nn.ModuleList([
            ShallowGraphBlock(
                d_model=d_model, d_ff=d_ff,
                attn_dropout=attn_dropout, ffn_dropout=ffn_dropout,
                attn_bias_kappa=attn_bias_kappa
            ) for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, X, A):
        H = self.project_in(X)  # fused adjacency from your class
        for blk in self.blocks:
            H = blk(H, A)
        return self.norm_out(H)
