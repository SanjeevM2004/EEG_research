# models/lstm.py

import torch
import torch.nn as nn

class SignalLSTM(nn.Module):
    """
    LSTM encoder for a SINGLE time-series per batch.

    Input:  (B, T)
    Output: (B, d_hidden)
    """
    def __init__(self, d_hidden=64, num_layers=1, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.d_hidden = d_hidden

    def forward(self, x):
        """
        x: (B, T)
        """
        x = x.unsqueeze(-1)            # -> (B, T, 1)
        out, _ = self.lstm(x)          # -> (B, T, d_hidden)
        return out[:, -1, :]           # last timestep -> (B, d_hidden)

class ChannelLSTM(nn.Module):
    """
    Applies SignalLSTM independently to each channel.

    Input:  (B, C, T)
    Output: (B, C, d_hidden)
    """
    def __init__(self, d_hidden=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.single_lstm = SignalLSTM(
            d_hidden=d_hidden,
            num_layers=num_layers,
            dropout=dropout
        )
        self.d_hidden = d_hidden

    def forward(self, x):
        """
        x: (B, C, T)
        """
        B, C, T = x.shape

        x = x.reshape(B * C, T)               # merge channels into batch
        h = self.single_lstm(x)               # -> (B*C, d_hidden)

        return h.reshape(B, C, self.d_hidden)
