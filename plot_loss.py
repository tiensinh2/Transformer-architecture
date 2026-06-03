"""
plot_loss.py — Vẽ biểu đồ train loss và validation loss.
Đọc từ logs/loss_log.json được lưu bởi train.py.
Chạy: python plot_loss.py
"""

import os
import json
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ------------------------------------------------------------------
# 1. Đọc log
# ------------------------------------------------------------------
def load_log(log_path: str = "logs/loss_log.json") -> dict:
    assert os.path.exists(log_path), (
        f"Không tìm thấy file log tại {log_path}. "
        "Hãy chạy train.py trước."
    )
    with open(log_path, "r") as f:
        log = json.load(f)
    assert "train_loss" in log and "val_loss" in log, \
        "File log phải có key 'train_loss' và 'val_loss'."
    return log


# ------------------------------------------------------------------
# 2. Vẽ biểu đồ
# ------------------------------------------------------------------
def plot_loss(
    log: dict,
    save_path: str = "logs/loss_curve.png",
    title: str = "Training & Validation Loss",
):
    train_loss = log["train_loss"]
    val_loss   = log["val_loss"]
    epochs     = list(range(1, len(train_loss) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, train_loss, marker="o", linewidth=2,
            color="#2563EB", label="Train Loss")
    ax.plot(epochs, val_loss,   marker="s", linewidth=2,
            color="#DC2626", label="Validation Loss", linestyle="--")

    # Đánh dấu điểm val_loss thấp nhất
    best_epoch = val_loss.index(min(val_loss))
    ax.annotate(
        f"Best val: {val_loss[best_epoch]:.4f}\n(epoch {best_epoch+1})",
        xy=(best_epoch + 1, val_loss[best_epoch]),
        xytext=(best_epoch + 1 + max(1, len(epochs) * 0.05),
                val_loss[best_epoch] + (max(val_loss) - min(val_loss)) * 0.1),
        arrowprops=dict(arrowstyle="->", color="#15803D"),
        fontsize=9, color="#15803D",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Đã lưu biểu đồ → {save_path}")
    plt.show()


# ------------------------------------------------------------------
# 3. Main
# ------------------------------------------------------------------
def main():
    log_path  = "logs/loss_log.json"
    save_path = "logs/loss_curve.png"

    log = load_log(log_path)

    n = len(log["train_loss"])
    print(f"[Plot] Số epoch đã train: {n}")
    print(f"[Plot] Train loss: {log['train_loss'][0]:.4f} → {log['train_loss'][-1]:.4f}")
    print(f"[Plot] Val   loss: {log['val_loss'][0]:.4f} → {log['val_loss'][-1]:.4f}")
    print(f"[Plot] Best val loss: {min(log['val_loss']):.4f} tại epoch {log['val_loss'].index(min(log['val_loss']))+1}")

    plot_loss(log, save_path)


if __name__ == "__main__":
    main()