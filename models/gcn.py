import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Backward-compatible GCN layer.
    Behaves EXACTLY like your old implementation when defaults are used.
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layernorm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()

        # Linear transform
        self.lin = nn.Linear(d_in, d_out, bias=False)

        # Residual path
        if use_residual:
            self.res_proj = (
                nn.Linear(d_in, d_out, bias=False)
                if d_in != d_out else nn.Identity()
            )
        else:
            self.res_proj = None

        # LayerNorm
        self.norm = nn.LayerNorm(d_out) if use_layernorm else nn.Identity()

        # Activation
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Store for compatibility
        self.use_residual = use_residual

    def forward(self, H, A):
        """
        H: (B, C, d_in)
        A: (B, C, C)
        """
        # Linear transform node features
        X = self.lin(H)  # (B, C, d_out)

        # Message passing
        out = torch.einsum("bij,bjd->bid", A, X)

        # Residual connection
        if self.use_residual:
            out = out + self.res_proj(H)

        # LayerNorm + Activation + Dropout
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)

        return out



class GCN(nn.Module):
    """
    Stack of GCN layers.
    This replicates your previous behavior 100% with defaults.
    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_residual: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(
                GCNLayer(
                    d_in=d_in if i == 0 else d_hidden,
                    d_out=d_hidden,
                    dropout=dropout,
                    activation=activation,
                    use_residual=use_residual,
                    use_layernorm=use_layernorm,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, H, A):
        for layer in self.layers:
            H = layer(H, A)
        return H  # (B, C, d_hidden)
