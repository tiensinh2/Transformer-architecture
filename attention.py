import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Scaled Dot-Product Attention theo paper gốc.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        q: (batch, heads, seq_q, head_dim)
        k: (batch, heads, seq_k, head_dim)
        v: (batch, heads, seq_v, head_dim)  -- seq_v == seq_k
        mask: (batch, 1, seq_q, seq_k) hoặc (batch, 1, 1, seq_k)
    Returns:
        out: (batch, heads, seq_q, head_dim)
        attn_weights: (batch, heads, seq_q, seq_k)
    """
    d_k = q.size(-1)
    # (batch, heads, seq_q, seq_k)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask == 0 là vị trí bị che → gán -inf để softmax ra 0
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)

    # Tránh NaN khi toàn bộ một hàng là -inf (padding row)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    out = attn_weights @ v
    return out, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention hỗ trợ cả self-attention và cross-attention.

    - Self-attention:   query = key = value = x
    - Cross-attention:  query từ decoder, key/value từ encoder output

    Args:
        d_model:   chiều embedding (512 theo paper)
        num_heads: số head (8 theo paper)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"

        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.d_model   = d_model

        # Tách W_Q, W_K, W_V thành 3 linear riêng để cross-attention dễ dùng
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """(batch, seq, d_model) → (batch, heads, seq, head_dim)"""
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        """(batch, heads, seq, head_dim) → (batch, seq, d_model)"""
        B, _, T, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_v, d_model)  -- seq_v == seq_k
            mask:  (batch, 1, seq_q, seq_k) hoặc None
        Returns:
            out:   (batch, seq_q, d_model)
        """
        q = self._split_heads(self.W_q(query))   # (B, H, T_q, head_dim)
        k = self._split_heads(self.W_k(key))     # (B, H, T_k, head_dim)
        v = self._split_heads(self.W_v(value))   # (B, H, T_v, head_dim)

        out, _ = scaled_dot_product_attention(q, k, v, mask)

        out = self._merge_heads(out)              # (B, T_q, d_model)
        out = self.dropout(self.W_o(out))
        return out