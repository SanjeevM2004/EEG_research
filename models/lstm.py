import torch
import torch.nn as nn

class LSTMWrapper(nn.Module):
    """
    LSTM that maintains hidden states across layers.
    Input:  (B, C, d) -> treat channels as "sequence steps".
    Output: (B, C, d_out), hidden (h, c)
    """
    def __init__(self, d_in: int, d_hidden: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x, hidden=None):
        """
        x: (B, C, d_in)
        hidden: (h, c) if provided
        """
        out, hidden = self.lstm(x, hidden)  # (B, C, d_hidden)
        return out, hidden
