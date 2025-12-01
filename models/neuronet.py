# neurographnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lstm import ChannelLSTM
from .gcn import GCN
from .dann import DomainDiscriminator
from .pooling import AttentionPooling


# ============================================================
# NeuroGraphNet (with Cref whitening)
# ============================================================
class NeuroGraphNet(nn.Module):
    """
    - Whiten signals using subject mean covariance (Cref)
    - Build adjacency using RA covariances
    - LSTM → GCN → Attention Pooling
    - Two DANN blocks: local + global
    """

    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        lstm_hidden: int = 64,
        gcn_hidden: int = 128,
        gcn_layers: int = 2,
        global_hidden: int = 64,
        dropout: float = 0.1,
        lambda_local: float = 0.1,
        lambda_global: float = 0.1,
        whiten_signals: bool = True,
        self_loop: float = 0.01,
    ):
        super().__init__()

        self.lambda_local = lambda_local
        self.lambda_global = lambda_global
        self.whiten_signals = whiten_signals
        self.self_loop = self_loop

        # Temporal encoder
        self.lstm = ChannelLSTM(
            d_hidden=lstm_hidden,
            num_layers=1,
            dropout=dropout,
        )

        # Local domain discriminator
        self.dom_local = DomainDiscriminator(
            d_in=lstm_hidden,
            num_domains=num_domains,
        )

        # GCN stack
        self.gcn = GCN(
            d_in=lstm_hidden,
            d_hidden=gcn_hidden,
            num_layers=gcn_layers,
            dropout=dropout,
        )

        # Global pooling
        self.pool = AttentionPooling(gcn_hidden)

        # Global domain discriminator
        self.dom_global = DomainDiscriminator(
            d_in=gcn_hidden,
            num_domains=num_domains,
        )

        # Classifier
        self.cls_head = nn.Sequential(
            nn.Linear(gcn_hidden, global_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden, num_classes),
        )

    # ============================================================
    # Adjacency (unchanged)
    # ============================================================
    def _build_adjacency(self, ra_covs: torch.Tensor) -> torch.Tensor:
        A = ra_covs.abs()

        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        A = A - torch.diag_embed(diag)

        if self.self_loop and self.self_loop > 0:
            B, C, _ = A.shape
            eye = torch.eye(C, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
            A = A + self.self_loop * eye

        return A

    # ============================================================
    # FIXED: Whitening ALWAYS uses Cref
    # ============================================================
    def _whiten_signals(self, signals: torch.Tensor, cref: torch.Tensor, eps: float = 1e-6):
        """
        X_white = Cref^{-1/2} X
        signals: (B, C, T)
        cref:    (B, C, C)  subject mean covariance
        """
        w, V = torch.linalg.eigh(cref)
        w = torch.clamp(w, min=eps)
        inv_sqrt = w.rsqrt()

        C_inv_sqrt = V @ torch.diag_embed(inv_sqrt) @ V.transpose(-1, -2)

        return torch.matmul(C_inv_sqrt, signals)

    # ============================================================
    # forward()
    # ============================================================
    def forward(
        self,
        signals: torch.Tensor,   # (B, C, T)
        ra_covs: torch.Tensor,   # (B, C, C)
        cref: torch.Tensor,      # (B, C, C)
        lambda_local: float | None = None,
        lambda_global: float | None = None,
    ):

        # -------------------------
        # 1) WHITEN using CREF
        # -------------------------
        if self.whiten_signals:
            X = self._whiten_signals(signals, cref)
        else:
            X = signals

        # -------------------------
        # 2) Build adjacency from RA covs
        # -------------------------
        A = self._build_adjacency(ra_covs)

        # -------------------------
        # 3) LSTM temporal encoding
        # -------------------------
        H0 = self.lstm(X)

        # -------------------------
        # 4) Local DAN (per-channel mean)
        # -------------------------
        lam_loc = self.lambda_local if lambda_local is None else lambda_local
        g_local = H0.mean(dim=1)
        dom_local_logits = self.dom_local(g_local, lam_loc)

        # -------------------------
        # 5) Spatial GCN
        # -------------------------
        H = self.gcn(H0, A)

        # -------------------------
        # 6) Attention pooling
        # -------------------------
        g_global = self.pool(H)

        # -------------------------
        # 7) Global DAN
        # -------------------------
        lam_glob = self.lambda_global if lambda_global is None else lambda_global
        dom_global_logits = self.dom_global(g_global, lam_glob)

        # -------------------------
        # 8) Classification
        # -------------------------
        logits = self.cls_head(g_global)

        return logits, dom_local_logits, dom_global_logits
