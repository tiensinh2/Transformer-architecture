import torch
import torch.nn as nn

from encoder import TransformerEncoder
from decoder import TransformerDecoder


def make_src_mask(src, pad_idx=0):
    """
    Tạo mask che padding token của encoder input.

    Args:
        src:     (batch, src_len) — token id câu nguồn
        pad_idx: id của padding token (mặc định 0)
    Returns:
        mask: (batch, 1, 1, src_len) — 1=token thật, 0=padding
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, pad_idx=0):
    """
    Tạo mask cho decoder gồm 2 phần kết hợp:
      1. Padding mask: che padding token
      2. Causal mask:  che token tương lai (autoregressive)

    Args:
        tgt:     (batch, tgt_len) — token id câu đích
        pad_idx: id của padding token
    Returns:
        mask: (batch, 1, tgt_len, tgt_len)
    """
    tgt_len = tgt.size(1)

    # Padding mask: (batch, 1, 1, tgt_len)
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal mask (tam giác dưới): (1, 1, tgt_len, tgt_len)
    causal_mask = torch.tril(
        torch.ones(tgt_len, tgt_len, device=tgt.device)
    ).unsqueeze(0).unsqueeze(0)

    # Kết hợp: chỉ attend được token thật VÀ token trước đó
    return pad_mask & causal_mask.bool()


class Transformer(nn.Module):
    """
    Seq2Seq Transformer hoàn chỉnh theo paper "Attention is All You Need".

    Args:
        src_vocab_size: kích thước từ điển nguồn (input)
        tgt_vocab_size: kích thước từ điển đích (output/summary)
        d_model:        chiều embedding (512 theo paper)
        num_layers:     số tầng encoder và decoder (6 theo paper)
        num_heads:      số attention head (8 theo paper)
        d_ff:           chiều ẩn FFN (2048 theo paper)
        max_len:        độ dài tối đa sequence
        dropout:        dropout rate (0.1 theo paper)
        pad_idx:        id của padding token
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:        int   = 512,
        num_layers:     int   = 6,
        num_heads:      int   = 8,
        d_ff:           int   = 2048,
        max_len:        int   = 5000,
        dropout:        float = 0.1,
        pad_idx:        int   = 0,
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
        )

    def forward(self, src, tgt):
        """
        Dùng trong lúc training (teacher forcing — truyền tgt thật vào decoder).

        Args:
            src: (batch, src_len) — token id văn bản gốc
            tgt: (batch, tgt_len) — token id tóm tắt (shifted right, bỏ token cuối)
        Returns:
            logits: (batch, tgt_len, tgt_vocab_size) — chưa qua softmax
        """
        src_mask = make_src_mask(src, self.pad_idx)   # (B, 1, 1, src_len)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)   # (B, 1, tgt_len, tgt_len)

        enc_out = self.encoder(src, src_mask)          # (B, src_len, d_model)
        logits  = self.decoder(tgt, enc_out, tgt_mask, src_mask)  # (B, tgt_len, V)

        return logits

    @torch.no_grad()
    def generate(
        self,
        src,
        bos_idx:   int = 1,
        eos_idx:   int = 2,
        max_new:   int = 128,
        strategy:  str = "greedy",
        num_beams: int = 4,
    ):
        """
        Sinh tóm tắt từ văn bản nguồn (inference only, không dùng teacher forcing).

        Args:
            src:       (batch, src_len) — token id văn bản gốc
            bos_idx:   id token bắt đầu <bos>
            eos_idx:   id token kết thúc <eos>
            max_new:   số token tối đa sinh ra
            strategy:  "greedy" hoặc "beam"
            num_beams: số beam (chỉ dùng khi strategy="beam")
        Returns:
            generated: (batch, seq_len) — token id đã sinh
        """
        self.eval()
        device = src.device

        src_mask = make_src_mask(src, self.pad_idx)
        enc_out  = self.encoder(src, src_mask)     # chạy encoder 1 lần

        if strategy == "greedy":
            return self._greedy(enc_out, src_mask, bos_idx, eos_idx, max_new, device)
        elif strategy == "beam":
            return self._beam_search(enc_out, src_mask, bos_idx, eos_idx, max_new, num_beams, device)
        else:
            raise ValueError(f"strategy phải là 'greedy' hoặc 'beam', nhận được: {strategy}")

    # ------------------------------------------------------------------
    # Greedy decoding
    # ------------------------------------------------------------------
    def _greedy(self, enc_out, src_mask, bos_idx, eos_idx, max_new, device):
        """
        Mỗi bước chọn token có xác suất cao nhất.
        Nhanh nhưng chất lượng thường thấp hơn beam search.

        Args:
            enc_out:  (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len)
        Returns:
            generated: (batch, seq_len)
        """
        batch = enc_out.size(0)

        # Khởi đầu bằng token <bos>
        generated = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)
        # Track xem batch nào đã sinh xong
        finished  = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_new):
            tgt_mask = make_tgt_mask(generated, self.pad_idx)
            logits   = self.decoder(generated, enc_out, tgt_mask, src_mask)
            # Lấy token cuối cùng
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch, 1)

            generated = torch.cat([generated, next_token], dim=1)

            # Đánh dấu batch đã kết thúc
            finished |= (next_token.squeeze(-1) == eos_idx)
            if finished.all():
                break

        return generated  # (batch, seq_len)

    # ------------------------------------------------------------------
    # Beam Search decoding
    # ------------------------------------------------------------------
    def _beam_search(self, enc_out, src_mask, bos_idx, eos_idx, max_new, num_beams, device):
        """
        Beam search: giữ num_beams hypothesis tốt nhất ở mỗi bước.
        Chất lượng tốt hơn greedy, hay dùng trong các bài tóm tắt.

        Giới hạn: hiện chỉ hỗ trợ batch_size=1 cho đơn giản.
        """
        assert enc_out.size(0) == 1, "Beam search hiện chỉ hỗ trợ batch_size=1"

        # Mở rộng enc_out cho num_beams
        enc_out  = enc_out.expand(num_beams, -1, -1)       # (B, src_len, d_model)
        src_mask = src_mask.expand(num_beams, -1, -1, -1)  # (B, 1, 1, src_len)

        # Mỗi beam: (sequence, log_prob)
        beams = [(torch.tensor([[bos_idx]], device=device), 0.0)]
        completed = []

        for _ in range(max_new):
            all_candidates = []

            # Stack tất cả beam hiện tại thành 1 batch để forward 1 lần
            seqs      = torch.cat([b[0] for b in beams], dim=0)  # (num_beams, t)
            tgt_mask  = make_tgt_mask(seqs, self.pad_idx)
            logits    = self.decoder(seqs, enc_out[:len(beams)], tgt_mask, src_mask[:len(beams)])

            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (num_beams, vocab)

            for i, (seq, score) in enumerate(beams):
                top_log_probs, top_tokens = log_probs[i].topk(num_beams)

                for log_p, tok in zip(top_log_probs, top_tokens):
                    new_seq   = torch.cat([seq, tok.view(1, 1)], dim=1)
                    new_score = score + log_p.item()
                    all_candidates.append((new_seq, new_score))

            # Sắp xếp theo score và giữ top num_beams
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = []

            for seq, score in all_candidates[:num_beams * 2]:
                if seq[0, -1].item() == eos_idx:
                    # Normalize theo độ dài để tránh ưu tiên câu ngắn
                    length_penalty = len(seq[0]) ** 0.6
                    completed.append((seq, score / length_penalty))
                else:
                    beams.append((seq, score))
                if len(beams) == num_beams:
                    break

            if len(completed) >= num_beams or len(beams) == 0:
                break

        if not completed:
            completed = beams  # fallback nếu không có beam nào kết thúc

        # Trả về beam có score cao nhất
        best_seq = max(completed, key=lambda x: x[1])[0]
        return best_seq  # (1, seq_len)
