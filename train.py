"""
train.py — Huấn luyện Transformer từ đầu cho bài toán tóm tắt văn bản tiếng Việt.
Chạy: python train.py
"""

import os
import math
import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import CFG
from data.dataset import setup_data
from model import Transformer


# ------------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# 2. Learning Rate Schedule (theo paper gốc)
#    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
# ------------------------------------------------------------------
class WarmupScheduler:
    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self.step_num     = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _get_lr(self) -> float:
        s = self.step_num
        w = self.warmup_steps
        return self.d_model ** (-0.5) * min(s ** (-0.5), s * w ** (-1.5))

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, state):
        self.step_num = state["step_num"]


# ------------------------------------------------------------------
# 3. Loss với label smoothing
# ------------------------------------------------------------------
class LabelSmoothingLoss(nn.Module):
    """
    CrossEntropy + Label Smoothing.
    Thay vì one-hot [0,0,1,0,...], target trở thành:
      - đúng class:   1 - smoothing
      - các class khác: smoothing / (V - 1)
    Giúp model không overfit, tổng quát hóa tốt hơn.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch * tgt_len, vocab_size) — chưa qua softmax
            target: (batch * tgt_len,)
        Returns:
            loss: scalar
        """
        log_probs = torch.log_softmax(logits, dim=-1)  # (N, V)

        # Tạo smooth distribution
        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
            # Padding token không tính loss
            smooth_target[target == self.pad_idx] = 0.0

        loss = -(smooth_target * log_probs).sum(dim=-1)  # (N,)

        # Chỉ tính trung bình trên token thật (không phải padding)
        non_pad = (target != self.pad_idx).sum()
        return loss.sum() / non_pad.clamp(min=1)


# ------------------------------------------------------------------
# 4. Train 1 epoch
# ------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (src, tgt_in, tgt_out) in enumerate(loader):
        src     = src.to(device)
        tgt_in  = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        # Forward
        logits = model(src, tgt_in)   # (batch, tgt_len, vocab_size)

        # Flatten để tính loss
        B, T, V = logits.shape
        loss = criterion(
            logits.reshape(B * T, V),
            tgt_out.reshape(B * T),
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — tránh exploding gradient
        nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)

        # Update weights + lr schedule
        optimizer.step()
        scheduler.step()

        # Đếm token thật (không phải padding)
        non_pad = (tgt_out != CFG.pad_idx).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad

        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx+1}/{len(loader)} | "
                  f"loss {loss.item():.4f} | "
                  f"lr {scheduler._get_lr():.2e}")

    return total_loss / max(total_tokens, 1)


# ------------------------------------------------------------------
# 5. Validate 1 epoch
# ------------------------------------------------------------------
@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for src, tgt_in, tgt_out in loader:
        src     = src.to(device)
        tgt_in  = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        logits = model(src, tgt_in)
        B, T, V = logits.shape
        loss = criterion(
            logits.reshape(B * T, V),
            tgt_out.reshape(B * T),
        )

        non_pad = (tgt_out != CFG.pad_idx).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


# ------------------------------------------------------------------
# 6. Lưu / Load checkpoint
# ------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss":       val_loss,
    }, path)
    print(f"  [Checkpoint] Đã lưu → {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"[Checkpoint] Load từ {path} | epoch {ckpt['epoch']} | val_loss {ckpt['val_loss']:.4f}")
    return ckpt["epoch"], ckpt["val_loss"]


# ------------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------------
def main():
    set_seed(CFG.seed)
    device = torch.device(CFG.device)
    print(f"[Train] Device: {device}")
    CFG.summary()

    # --- Data ---
    tokenizer, train_loader, valid_loader = setup_data()
    vocab_size = len(tokenizer)
    print(f"[Train] Vocab size: {vocab_size:,}")

    # --- Model ---
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,   # shared vocab
        **CFG.model_kwargs(),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Số tham số: {n_params:,}")

    # --- Loss, Optimizer, Scheduler ---
    criterion = LabelSmoothingLoss(vocab_size, CFG.pad_idx, CFG.label_smoothing)

    optimizer = Adam(
        model.parameters(),
        betas=(CFG.adam_beta1, CFG.adam_beta2),
        eps=CFG.adam_eps,
    )

    scheduler = WarmupScheduler(optimizer, CFG.d_model, CFG.warmup_steps)

    # --- Resume nếu có checkpoint ---
    start_epoch = 0
    best_val_loss = float("inf")
    best_ckpt = os.path.join(CFG.output_dir, "best_model.pt")
    last_ckpt = os.path.join(CFG.output_dir, "last_model.pt")

    if os.path.exists(last_ckpt):
        start_epoch, best_val_loss = load_checkpoint(last_ckpt, model, optimizer, scheduler)
        start_epoch += 1

    # --- Log ---
    log = {"train_loss": [], "val_loss": []}
    log_path = os.path.join(CFG.log_dir, "loss_log.json")

    # --- Training Loop ---
    print(f"\n[Train] Bắt đầu train từ epoch {start_epoch+1}/{CFG.num_epochs}")
    print("=" * 60)

    for epoch in range(start_epoch, CFG.num_epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss   = validate_epoch(model, valid_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{CFG.num_epochs} | "
              f"train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | "
              f"time {elapsed:.1f}s")

        # Lưu log
        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        # Lưu best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_ckpt)
            print(f"  → Best model cập nhật (val_loss={val_loss:.4f})")

        # Lưu last checkpoint để resume
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, last_ckpt)

    print("=" * 60)
    print(f"[Train] Hoàn thành! Best val_loss: {best_val_loss:.4f}")
    print(f"[Train] Best model: {best_ckpt}")
    print(f"[Train] Loss log:   {log_path}")


if __name__ == "__main__":
    main()