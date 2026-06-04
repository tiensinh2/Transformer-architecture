import math
import torch.nn as nn

from attention import MultiHeadAttention, scaled_dot_product_attention
from layer import PositionalEncoding, FeedForwardNetwork


class DecoderLayer(nn.Module):
    """
    1 tầng Decoder gồm:
      1. Masked Multi-Head Self-Attention  (chỉ nhìn được token trước đó)
      2. Add & Norm
      3. Multi-Head Cross-Attention        (Q từ decoder, K/V từ encoder)
      4. Add & Norm
      5. Feed-Forward Network
      6. Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 1. Masked self-attention
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        # 2. Cross-attention: Q từ decoder, K/V từ encoder output
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 3. FFN
        self.ffn        = FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        """
        Args:
            x:        (batch, tgt_len, d_model)  — decoder input
            enc_out:  (batch, src_len, d_model)  — output của encoder
            tgt_mask: (batch, 1, tgt_len, tgt_len) — causal mask (che token tương lai)
            src_mask: (batch, 1, 1, src_len)       — che padding của encoder input
        Returns:
            x:        (batch, tgt_len, d_model)
        """
        # --- 1. Masked Self-Attention ---
        # decoder chỉ attend được các token trước nó (autoregressive)
        # Q = K = V = x (decoder hidden state)
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(
            self.self_attn(query=x, key=x, value=x, mask=tgt_mask)
        )

        # --- 2. Cross-Attention ---
        # Q = decoder hidden state (x)
        # K = V = encoder output (enc_out)
        # Đây là chỗ decoder "đọc" thông tin từ encoder
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(
            self.cross_attn(query=x, key=enc_out, value=enc_out, mask=src_mask)
        )

        # --- 3. Feed-Forward Network ---
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.ffn(x))

        return x


class TransformerDecoder(nn.Module):
    """
    Decoder hoàn chỉnh:
      Embedding → Scale → Positional Encoding → N x DecoderLayer → Linear out

    Args:
        vocab_size: kích thước từ điển đích (target)
        d_model:    chiều embedding (512 theo paper)
        num_layers: số tầng Decoder (6 theo paper)
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
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm    = nn.LayerNorm(d_model)
        self.fc_out  = nn.Linear(d_model, vocab_size)  # project ra vocab

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        """
        Args:
            tgt:      (batch, tgt_len)           — token id của câu đích (shifted right)
            enc_out:  (batch, src_len, d_model)  — output của encoder
            tgt_mask: (batch, 1, tgt_len, tgt_len) — causal mask
            src_mask: (batch, 1, 1, src_len)       — padding mask encoder
        Returns:
            logits:   (batch, tgt_len, vocab_size) — chưa qua softmax
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)

        x = self.norm(x)
        return self.fc_out(x)  # (batch, tgt_len, vocab_size)
