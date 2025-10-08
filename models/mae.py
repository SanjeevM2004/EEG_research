from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn


class TransformerMAE(nn.Module):
    def __init__(
        self,
        d_in: int,                 # input feature dimension per channel (e.g., 35)
        n_channels: int,           # number of EEG channels (e.g., 64)
        d_model: int = 128,        # embedding size for context-rich node features
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_huber: bool = False,   # SmoothL1 instead of MSE
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_channels = n_channels
        self.d_model = d_model

        # Project per-channel features to token embeddings
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable channel (positional) embeddings
        self.channel_embed = nn.Parameter(torch.zeros(1, n_channels, d_model))
        nn.init.trunc_normal_(self.channel_embed, std=0.02)

        # Transformer encoder (channels attend to each other)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Lightweight decoder from token embeddings -> original feature dims
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_in),
        )

        self.criterion = nn.SmoothL1Loss(reduction="none") if use_huber else nn.MSELoss(reduction="none")

    def forward(self, Z_masked: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        Z_masked : (B, C, d_in) tensor
            Features with masked dims already replaced by your mask token.
        mask : (B, C, d_in) bool tensor
            True where masked; used only for loss outside, not required for forward mechanics.

        Returns
        -------
        Z_hat : (B, C, d_in)
            Reconstruction of per-channel features.
        H : (B, C, d_model)
            Context-rich token embeddings from the encoder (use these as node embeddings).
        """
        B, C, d = Z_masked.shape
        if C != self.n_channels or d != self.d_in:
            raise ValueError(f"Shape mismatch: got (C,d)=({C},{d}) expected ({self.n_channels},{self.d_in})")

        tokens = self.input_proj(Z_masked)            # (B,C,d_model)
        tokens = tokens + self.channel_embed          # channel positional embeddings
        H = self.encoder(tokens)                      # (B,C,d_model)
        Z_hat = self.decoder(H)                       # (B,C,d_in)
        return Z_hat, H

    def masked_reconstruction_loss(self, Z_hat: torch.Tensor, Z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked element loss (MSE/Huber) only on masked dims.
        Returns a scalar.
        """
        per_elem = self.criterion(Z_hat, Z)           # (B,C,d)
        masked = per_elem * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
        return masked.sum() / denom

    @torch.no_grad()
    def encode(self, Z: torch.Tensor) -> torch.Tensor:
        """Return encoder token embeddings without masking (context-rich node features).
        Parameters
        ----------
        Z : (B, C, d_in)
        Returns
        -------
        H : (B, C, d_model)
        """
        self.eval()
        B, C, d = Z.shape
        if C != self.n_channels or d != self.d_in:
            raise ValueError(f"Shape mismatch: got (C,d)=({C},{d}) expected ({self.n_channels},{self.d_in})")
        tokens = self.input_proj(Z)
        tokens = tokens + self.channel_embed
        H = self.encoder(tokens)
        return H
    
"""
Transformer-based Masked Autoencoder (MAE) for EEG per-channel feature arrays.

- Expects masking to be handled OUTSIDE the model (you already generate `mask` and `feats_masked`).
- Encoder is a Transformer over channels (sequence length = C).
- Decoder is a light MLP to reconstruct original features (dim = d_in).
- `encode(Z)` returns context-rich node embeddings of size `d_model` (can be > d_in).

Usage (inside training loop):
    Z         : (B, C, d_in)
    mask      : (B, C, d_in)  # True where masked
    Z_masked  : (B, C, d_in)  # you already replaced masked dims with a mask token

    Z_hat, H = mae(Z_masked, mask)               # forward
    loss = mae.masked_reconstruction_loss(Z_hat, Z, mask)

    # After pretraining:
    H = mae.encode(Z)                            # (B, C, d_model)

"""