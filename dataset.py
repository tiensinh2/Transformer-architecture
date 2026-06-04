"""
Dataset và DataLoader cho bài toán tóm tắt văn bản tiếng Việt.
Đọc file .parquet, encode bằng Tokenizer tự xây, trả về tensor.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CFG
from tokenizer import Tokenizer, build_tokenizer_from_dataset


class SummarizationDataset(Dataset):
    """
    Dataset cho bài tóm tắt văn bản.
    Mỗi mẫu gồm:
      - src:      (src_max_len,)  token id văn bản gốc
      - tgt_in:   (tgt_max_len,)  token id tóm tắt, bắt đầu bằng <bos>, bỏ token cuối
                                  → dùng làm input decoder (teacher forcing)
      - tgt_out:  (tgt_max_len,)  token id tóm tắt, bỏ <bos>, kết thúc bằng <eos>
                                  → dùng làm ground truth để tính loss
    """

    def __init__(self, df: pd.DataFrame, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        # Lọc bỏ dòng rỗng nếu có
        df = df.dropna(subset=[CFG.src_col, CFG.tgt_col]).reset_index(drop=True)
        self.articles  = df[CFG.src_col].tolist()
        self.summaries = df[CFG.tgt_col].tolist()

        print(f"[Dataset] Loaded {len(self.articles):,} mẫu")

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_text = str(self.articles[idx])
        tgt_text = str(self.summaries[idx])

        # Encode văn bản nguồn: <bos> + tokens + <eos> + padding
        src = self.tokenizer.encode(
            src_text,
            max_len=CFG.src_max_len,
            add_bos=True,
            add_eos=True,
        )

        # Encode tóm tắt đầy đủ: <bos> + tokens + <eos> + padding
        tgt_full = self.tokenizer.encode(
            tgt_text,
            max_len=CFG.tgt_max_len + 1,   # +1 để sau khi shift vẫn đủ tgt_max_len
            add_bos=True,
            add_eos=True,
        )

        # Teacher forcing:
        #   tgt_in  = <bos> t1 t2 ... t_{n-1}   (input decoder)
        #   tgt_out = t1 t2 ... t_{n-1} <eos>    (target để tính loss)
        tgt_in  = tgt_full[:CFG.tgt_max_len]       # bỏ token cuối
        tgt_out = tgt_full[1:CFG.tgt_max_len + 1]  # bỏ <bos>

        return (
            torch.tensor(src,     dtype=torch.long),
            torch.tensor(tgt_in,  dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long),
        )


def load_data(
    tokenizer: Tokenizer,
    train_file: str = CFG.train_file,
    valid_file: str = CFG.valid_file,
) -> Tuple[DataLoader, DataLoader]:
    """
    Đọc file parquet, tạo Dataset và DataLoader cho train và valid.

    Args:
        tokenizer:   Tokenizer đã được build vocab
        train_file:  đường dẫn file train.parquet
        valid_file:  đường dẫn file valid.parquet
    Returns:
        train_loader, valid_loader
    """
    print(f"[Data] Đọc train: {train_file}")
    train_df = pd.read_parquet(train_file)

    print(f"[Data] Đọc valid: {valid_file}")
    valid_df = pd.read_parquet(valid_file)

    train_dataset = SummarizationDataset(train_df, tokenizer)
    valid_dataset = SummarizationDataset(valid_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=(CFG.device == "cuda"),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(CFG.device == "cuda"),
    )

    print(f"[Data] Train batches: {len(train_loader):,} | Valid batches: {len(valid_loader):,}")
    return train_loader, valid_loader


def setup_data(
    train_file:  str = CFG.train_file,
    valid_file:  str = CFG.valid_file,
    vocab_path:  str = "data/vocab.json",
) -> Tuple[Tokenizer, DataLoader, DataLoader]:
    """
    Hàm tiện ích gọi từ train.py:
      1. Nếu vocab.json đã tồn tại → load luôn
      2. Nếu chưa có → build từ train set rồi lưu
      3. Trả về tokenizer + 2 DataLoader

    Args:
        train_file: đường dẫn train.parquet
        valid_file: đường dẫn valid.parquet
        vocab_path: đường dẫn lưu/load vocab
    Returns:
        tokenizer, train_loader, valid_loader
    """
    import os
    if os.path.exists(vocab_path):
        print(f"[Data] Load vocab từ {vocab_path}")
        tokenizer = Tokenizer.load(vocab_path)
    else:
        print(f"[Data] Chưa có vocab, đang build từ {train_file} ...")
        tokenizer = build_tokenizer_from_dataset(train_file, vocab_path)

    train_loader, valid_loader = load_data(tokenizer, train_file, valid_file)
    return tokenizer, train_loader, valid_loader
