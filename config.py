from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Đường dẫn dữ liệu
    # ------------------------------------------------------------------
    train_file:  str = "data/train.parquet"       # hoặc .json tùy dataset
    valid_file:  str = "data/valid.parquet"
    output_dir:  str = "checkpoints"          # lưu checkpoint
    log_dir:     str = "logs"                 # lưu loss log

    # Tên cột trong file dữ liệu — đổi cho đúng với dataset thực tế
    src_col:     str = "article"              # cột văn bản gốc
    tgt_col:     str = "summary"           # cột tóm tắt

    # ------------------------------------------------------------------
    # Tokenizer / Vocabulary
    # ------------------------------------------------------------------
    vocab_size:  int = 16000      # kích thước vocab, tăng nếu dataset lớn
    min_freq:    int = 2          # từ xuất hiện ít hơn min_freq bị bỏ qua
    pad_token:   str = "<pad>"
    bos_token:   str = "<bos>"
    eos_token:   str = "<eos>"
    unk_token:   str = "<unk>"

    # id tương ứng (phải khớp với thứ tự thêm vào vocab trong tokenizer.py)
    pad_idx:     int = 0
    bos_idx:     int = 1
    eos_idx:     int = 2
    unk_idx:     int = 3

    # ------------------------------------------------------------------
    # Độ dài sequence
    # ------------------------------------------------------------------
    src_max_len: int = 512        # độ dài tối đa văn bản nguồn (token)
    tgt_max_len: int = 128        # độ dài tối đa tóm tắt (token)

    # ------------------------------------------------------------------
    # Kiến trúc Transformer (theo paper gốc)
    # ------------------------------------------------------------------
    d_model:     int = 512        # chiều embedding
    num_layers:  int = 6          # số tầng encoder và decoder
    num_heads:   int = 8          # số attention head
    d_ff:        int = 2048       # chiều ẩn Feed-Forward (thường = 4 * d_model)
    dropout:     float = 0.1      # dropout rate

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    batch_size:  int = 32
    num_epochs:  int = 20
    grad_clip:   float = 1.0      # gradient clipping để tránh exploding gradient

    # Learning rate schedule (warmup theo paper gốc)
    # lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    warmup_steps: int = 4000
    # Adam betas và epsilon theo paper
    adam_beta1:  float = 0.9
    adam_beta2:  float = 0.98
    adam_eps:    float = 1e-9

    # Label smoothing (cải tiến so với paper gốc, giúp tránh overfit)
    label_smoothing: float = 0.1

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    decode_strategy: str = "beam"   # "greedy" hoặc "beam"
    num_beams:       int = 4
    max_gen_len:     int = 128

    # ------------------------------------------------------------------
    # Hệ thống
    # ------------------------------------------------------------------
    seed:        int = 42
    num_workers: int = 4          # DataLoader workers
    device:      str = "cuda"     # "cuda" hoặc "cpu", tự detect bên dưới

    def __post_init__(self):
        import torch
        # Tự động fallback sang CPU nếu không có GPU
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("[Config] Không tìm thấy GPU, chuyển sang CPU.")

        # Tạo thư mục nếu chưa có
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def model_kwargs(self):
        """Trả về dict truyền thẳng vào Transformer.__init__()"""
        return dict(
            d_model    = self.d_model,
            num_layers = self.num_layers,
            num_heads  = self.num_heads,
            d_ff       = self.d_ff,
            dropout    = self.dropout,
            pad_idx    = self.pad_idx,
        )

    def summary(self):
        """In tóm tắt config ra màn hình."""
        print("=" * 50)
        print("CONFIG")
        print("=" * 50)
        for k, v in self.__dict__.items():
            print(f"  {k:<20} = {v}")
        print("=" * 50)


# Dùng ngay: from config import CFG
CFG = Config()