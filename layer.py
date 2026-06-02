import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding theo paper gốc (sinusoidal).
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: chiều embedding
        max_len: độ dài tối đa của sequence
        dropout: dropout sau khi cộng PE vào embedding
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)               # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        # Dùng log-space để tính div_term ổn định hơn
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )                                                 # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)     # chiều chẵn
        pe[:, 1::2] = torch.cos(position * div_term)     # chiều lẻ

        pe = pe.unsqueeze(0)                              # (1, max_len, d_model)
        self.register_buffer("pe", pe)                   # không train, nhưng lưu vào state_dict

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + PE: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network theo paper gốc.
    FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        d_model: chiều embedding (512)
        d_ff:    chiều ẩn bên trong FFN (2048 theo paper, tức gấp 4 lần d_model)
        dropout: dropout sau lớp đầu tiên
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        return self.net(x)