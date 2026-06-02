import math
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import PositionalEncoding, FeedForwardNetwork


class EncoderLayer(nn.Module):
    """
    1 tầng Encoder gồm:
      1. Multi-Head Self-Attention
      2. Add & Norm
      3. Feed-Forward Network
      4. Add & Norm

    Dùng Pre-LN (LayerNorm trước sublayer) — ổn định hơn khi train sâu.
    Paper gốc dùng Post-LN nhưng Pre-LN thường hội tụ tốt hơn trong thực tế.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        Args:
            x:        (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len) — che padding token của encoder input
        Returns:
            x:        (batch, src_len, d_model)
        """
        # --- 1. Multi-Head Self-Attention + Add & Norm ---
        # Pre-LN: normalize trước khi đưa vào sublayer
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, src_mask))

        # --- 2. Feed-Forward Network + Add & Norm ---
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x


class TransformerEncoder(nn.Module):
    """
    Encoder hoàn chỉnh:
      Embedding → Scale → Positional Encoding → N x EncoderLayer

    Args:
        vocab_size: kích thước từ điển nguồn
        d_model:    chiều embedding (512 theo paper)
        num_layers: số tầng Encoder (6 theo paper)
        num_heads:  số head attention (8 theo paper)
        d_ff:       chiều FFN ẩn (2048 theo paper)
        max_len:    độ dài tối đa sequence
        dropout:    dropout rate (0.1 theo paper)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int = 512,
        num_layers: int = 6,
        num_heads:  int = 8,
        d_ff:       int = 2048,
        max_len:    int = 5000,
        dropout:    float = 0.1,
    ):
        super().__init__()

        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)  # Final LayerNorm sau tất cả các layer

        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số theo Xavier uniform như paper gốc."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        """
        Args:
            src:      (batch, src_len)         — token id của câu nguồn
            src_mask: (batch, 1, 1, src_len)   — 1 = token thật, 0 = padding
        Returns:
            x:        (batch, src_len, d_model) — encoder output
        """
        # Scale embedding theo paper: nhân sqrt(d_model) trước khi cộng PE
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)