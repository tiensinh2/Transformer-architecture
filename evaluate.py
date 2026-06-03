"""
evaluate.py — Đánh giá model Transformer trên valid set.
Tính ROUGE-1, ROUGE-2, ROUGE-L và in ví dụ thực tế.
Chạy: python evaluate.py
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import CFG
from data.tokenizer import Tokenizer
from model import Transformer


# ------------------------------------------------------------------
# 1. Cài rouge_score nếu chưa có
# ------------------------------------------------------------------
try:
    from rouge_score import rouge_scorer
except ImportError:
    os.system("pip install rouge-score -q")
    from rouge_score import rouge_scorer


# ------------------------------------------------------------------
# 2. Load model từ checkpoint
# ------------------------------------------------------------------
def load_model(checkpoint_path: str, vocab_size: int, device) -> Transformer:
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        **CFG.model_kwargs(),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Evaluate] Load model từ {checkpoint_path} | epoch {ckpt['epoch']+1} | val_loss {ckpt['val_loss']:.4f}")
    return model


# ------------------------------------------------------------------
# 3. Sinh tóm tắt cho toàn bộ valid set
# ------------------------------------------------------------------
@torch.no_grad()
def generate_summaries(
    model: Transformer,
    tokenizer: Tokenizer,
    articles: list,
    device,
    strategy: str = CFG.decode_strategy,
    num_beams: int = CFG.num_beams,
    max_gen_len: int = CFG.max_gen_len,
) -> list:
    """
    Sinh tóm tắt cho danh sách bài báo.

    Args:
        model:      Transformer đã load checkpoint
        tokenizer:  Tokenizer đã build vocab
        articles:   list[str] văn bản gốc
        device:     cpu hoặc cuda
        strategy:   "greedy" hoặc "beam"
        num_beams:  số beam (chỉ dùng khi strategy="beam")
        max_gen_len: số token tối đa sinh ra
    Returns:
        summaries: list[str] tóm tắt đã decode
    """
    summaries = []

    for article in tqdm(articles, desc="Generating"):
        # Encode văn bản nguồn
        src_ids = tokenizer.encode(
            article,
            max_len=CFG.src_max_len,
            add_bos=True,
            add_eos=True,
        )
        src = torch.tensor([src_ids], dtype=torch.long).to(device)  # (1, src_len)

        # Sinh tóm tắt
        generated = model.generate(
            src,
            bos_idx=CFG.bos_idx,
            eos_idx=CFG.eos_idx,
            max_new=max_gen_len,
            strategy=strategy,
            num_beams=num_beams,
        )

        # Decode token id → text
        token_ids = generated[0].cpu().tolist()
        summary = tokenizer.decode(token_ids, skip_special=True)
        summaries.append(summary)

    return summaries


# ------------------------------------------------------------------
# 4. Tính ROUGE
# ------------------------------------------------------------------
def compute_rouge(predictions: list, references: list) -> dict:
    """
    Tính ROUGE-1, ROUGE-2, ROUGE-L cho toàn bộ valid set.

    Args:
        predictions: list[str] tóm tắt do model sinh ra
        references:  list[str] tóm tắt ground truth
    Returns:
        scores: dict với các key rouge1, rouge2, rougeL
                mỗi key có f1, precision, recall (trung bình toàn dataset)
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,   # tiếng Việt không cần stemmer
    )

    totals = {
        "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    }

    n = len(predictions)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key]["precision"] += scores[key].precision
            totals[key]["recall"]    += scores[key].recall
            totals[key]["f1"]        += scores[key].fmeasure

    # Tính trung bình
    for key in totals:
        for metric in totals[key]:
            totals[key][metric] /= n

    return totals


# ------------------------------------------------------------------
# 5. In kết quả đẹp
# ------------------------------------------------------------------
def print_rouge(scores: dict):
    print("\n" + "=" * 55)
    print(f"{'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("=" * 55)
    for key, vals in scores.items():
        print(f"{key.upper():<12} "
              f"{vals['precision']*100:>9.2f}% "
              f"{vals['recall']*100:>9.2f}% "
              f"{vals['f1']*100:>9.2f}%")
    print("=" * 55)


def print_examples(articles, references, predictions, n: int = 5):
    """In n ví dụ thực tế để đánh giá định tính."""
    print(f"\n{'='*60}")
    print("VÍ DỤ ĐỊNH TÍNH (model sinh ra)")
    print(f"{'='*60}")
    for i in range(min(n, len(articles))):
        print(f"\n[{i+1}] BÀI BÁO (đầu):")
        print(f"  {articles[i][:200]}...")
        print(f"\n  GROUND TRUTH : {references[i]}")
        print(f"  MODEL OUTPUT : {predictions[i]}")
        print("-" * 60)


# ------------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------------
def main():
    device = torch.device(CFG.device)
    print(f"[Evaluate] Device: {device}")

    # --- Load tokenizer ---
    vocab_path = "data/vocab.json"
    assert os.path.exists(vocab_path), \
        f"Chưa có vocab tại {vocab_path}. Hãy chạy train.py trước."
    tokenizer = Tokenizer.load(vocab_path)

    # --- Load valid set ---
    print(f"[Evaluate] Đọc valid set: {CFG.valid_file}")
    valid_df   = pd.read_parquet(CFG.valid_file)
    articles   = valid_df[CFG.src_col].tolist()
    references = valid_df[CFG.tgt_col].tolist()
    print(f"[Evaluate] Số mẫu valid: {len(articles):,}")

    # --- Load model ---
    best_ckpt = os.path.join(CFG.output_dir, "best_model.pt")
    assert os.path.exists(best_ckpt), \
        f"Chưa có checkpoint tại {best_ckpt}. Hãy chạy train.py trước."
    model = load_model(best_ckpt, vocab_size=len(tokenizer), device=device)

    # --- Greedy decode ---
    print(f"\n[Evaluate] Sinh tóm tắt (strategy=greedy)...")
    preds_greedy = generate_summaries(
        model, tokenizer, articles, device,
        strategy="greedy",
    )
    scores_greedy = compute_rouge(preds_greedy, references)
    print("\n[Greedy Decoding]")
    print_rouge(scores_greedy)

    # --- Beam search decode ---
    print(f"\n[Evaluate] Sinh tóm tắt (strategy=beam, beams={CFG.num_beams})...")
    preds_beam = generate_summaries(
        model, tokenizer, articles, device,
        strategy="beam", num_beams=CFG.num_beams,
    )
    scores_beam = compute_rouge(preds_beam, references)
    print(f"\n[Beam Search (beams={CFG.num_beams})]")
    print_rouge(scores_beam)

    # --- In ví dụ định tính ---
    print_examples(articles, references, preds_beam, n=5)

    # --- Lưu kết quả ---
    results = {
        "greedy": scores_greedy,
        "beam":   scores_beam,
    }
    result_path = os.path.join(CFG.log_dir, "rouge_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Evaluate] Đã lưu kết quả ROUGE → {result_path}")


if __name__ == "__main__":
    main()