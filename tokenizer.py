"""
Tokenizer đơn giản tự xây từ đầu, không dùng thư viện ngoài.
Chiến lược: tách từ theo khoảng trắng và dấu câu (word-level tokenization).
Đủ dùng cho Bài 1 theo yêu cầu đề.
"""

import re
import json
from pathlib import Path
from collections import Counter
from typing import List, Optional

from config import CFG


def basic_tokenize(text: str) -> List[str]:
    """
    Tách text thành list token đơn giản:
      - lowercase
      - tách dấu câu ra khỏi từ
      - split theo khoảng trắng
    """
    text = text.lower().strip()
    # Thêm khoảng trắng quanh dấu câu để tách chúng thành token riêng
    text = re.sub(r"([.,!?;:\"\'()\[\]{}<>])", r" \1 ", text)
    # Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


class Vocabulary:
    """
    Quản lý ánh xạ token ↔ id.
    Các special token luôn ở đầu với id cố định (theo CFG):
      0: <pad>
      1: <bos>
      2: <eos>
      3: <unk>
    """

    SPECIALS = [
        CFG.pad_token,   # 0
        CFG.bos_token,   # 1
        CFG.eos_token,   # 2
        CFG.unk_token,   # 3
    ]

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self._build_specials()

    def _build_specials(self):
        for idx, tok in enumerate(self.SPECIALS):
            self.token2id[tok] = idx
            self.id2token[idx] = tok

    def build_from_texts(self, texts: List[str], vocab_size: int, min_freq: int):
        """
        Xây vocab từ danh sách văn bản thô.

        Args:
            texts:      list các string (toàn bộ train set)
            vocab_size: giới hạn tổng số token (bao gồm cả special)
            min_freq:   bỏ qua từ xuất hiện ít hơn min_freq lần
        """
        counter = Counter()
        for text in texts:
            counter.update(basic_tokenize(text))

        # Sắp xếp theo tần suất giảm dần, bỏ qua từ hiếm
        most_common = [
            tok for tok, freq in counter.most_common()
            if freq >= min_freq
        ]

        # Giới hạn vocab_size (trừ đi số special token đã có)
        max_tokens = vocab_size - len(self.SPECIALS)
        most_common = most_common[:max_tokens]

        for tok in most_common:
            if tok not in self.token2id:   # không ghi đè special token
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok

        print(f"[Vocab] Kích thước vocab: {len(self.token2id):,} token")

    def __len__(self):
        return len(self.token2id)

    def __getitem__(self, token: str) -> int:
        return self.token2id.get(token, CFG.unk_idx)

    def decode_id(self, idx: int) -> str:
        return self.id2token.get(idx, CFG.unk_token)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
        print(f"[Vocab] Đã lưu vocab vào {path}")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            token2id = json.load(f)
        vocab.token2id = {k: int(v) for k, v in token2id.items()}
        vocab.id2token = {int(v): k for k, v in token2id.items()}
        print(f"[Vocab] Đã load vocab từ {path} — {len(vocab):,} token")
        return vocab


class Tokenizer:
    """
    Tokenizer hoàn chỉnh: text → list[int] và ngược lại.
    Dùng chung 1 vocab cho cả nguồn lẫn đích (shared vocab).
    """

    def __init__(self, vocab: Optional[Vocabulary] = None):
        self.vocab = vocab or Vocabulary()

    def build(self, texts: List[str]):
        """Xây vocab từ danh sách text."""
        self.vocab.build_from_texts(texts, CFG.vocab_size, CFG.min_freq)

    def encode(
        self,
        text: str,
        max_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Text → list token id, có cắt/pad đến max_len.

        Args:
            text:    chuỗi văn bản thô
            max_len: độ dài tối đa (bao gồm cả <bos> và <eos>)
            add_bos: thêm <bos> ở đầu
            add_eos: thêm <eos> ở cuối
        Returns:
            ids: list[int] độ dài đúng max_len (đã pad nếu thiếu)
        """
        tokens = basic_tokenize(text)

        # Tính số token nội dung còn lại sau khi trừ special token
        budget = max_len - int(add_bos) - int(add_eos)
        tokens = tokens[:budget]   # cắt nếu quá dài

        ids = []
        if add_bos:
            ids.append(CFG.bos_idx)
        ids += [self.vocab[tok] for tok in tokens]
        if add_eos:
            ids.append(CFG.eos_idx)

        # Pad đến max_len
        ids += [CFG.pad_idx] * (max_len - len(ids))
        return ids

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
    ) -> str:
        """
        List token id → chuỗi văn bản.

        Args:
            ids:          list[int] token id
            skip_special: bỏ qua <pad>, <bos>, <eos>
        Returns:
            text: chuỗi văn bản đã decode
        """
        special_ids = {CFG.pad_idx, CFG.bos_idx, CFG.eos_idx}
        tokens = []
        for idx in ids:
            if skip_special and idx in special_ids:
                continue
            if idx == CFG.eos_idx:
                break   # dừng tại <eos>
            tokens.append(self.vocab.decode_id(idx))
        return " ".join(tokens)

    def save(self, path: str):
        self.vocab.save(path)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        vocab = Vocabulary.load(path)
        return cls(vocab)

    def __len__(self):
        return len(self.vocab)


# ------------------------------------------------------------------
# Hàm tiện ích: xây và lưu tokenizer từ file dữ liệu
# ------------------------------------------------------------------
def build_tokenizer_from_dataset(
    train_file: str = CFG.train_file,
    save_path:  str = "data/vocab.json",
) -> Tokenizer:
    """
    Đọc file train, xây vocab từ cả cột src lẫn tgt, lưu vocab ra file.
    Gọi hàm này 1 lần trước khi train.

    Hỗ trợ file CSV và JSON.
    """
    import pandas as pd

    print(f"[Tokenizer] Đọc dữ liệu từ {train_file} ...")
    if train_file.endswith(".parquet"):
        df = pd.read_parquet(train_file)
    elif train_file.endswith(".csv"):
        df = pd.read_csv(train_file)
    elif train_file.endswith(".json") or train_file.endswith(".jsonl"):
        df = pd.read_json(train_file, lines=train_file.endswith(".jsonl"))
    else:
        raise ValueError("Chỉ hỗ trợ .parquet, .csv, .json/.jsonl")

    # Gộp cả src và tgt để xây vocab chung
    all_texts = df[CFG.src_col].tolist() + df[CFG.tgt_col].tolist()
    all_texts = [str(t) for t in all_texts if isinstance(t, str)]

    tokenizer = Tokenizer()
    tokenizer.build(all_texts)
    tokenizer.save(save_path)
    return tokenizer