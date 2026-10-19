"""Baseline vs. fine-tuned pairwise F1 comparison on synthetic mock data.

IMPORTANT — SYNTHETIC DEMO ONLY:
    The results produced by this script are from synthetic mock data, not from
    the proprietary 8M-statute corpus. The +0.20 F1 improvement reported in the
    README (0.61 → 0.81) was measured on the proprietary corpus using the full
    pipeline and cannot be reproduced from this public repository.

    This script exists to verify that:
      (a) The mock data supports non-zero triplet construction
      (b) Fine-tuning updates model weights (non-zero gradient flow)
      (c) The evaluation metric functions correctly
    It does NOT attempt to reproduce the historical research result.

Usage:
    python src/experiment.py \\
        --pairs data/mock_pairs.jsonl \\
        --output runs/experiment \\
        --epochs 2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run_experiment(
    pairs_path: str,
    output_dir: str,
    base_model: str = "Qwen/Qwen3-Embedding-0.6B",
    epochs: int = 2,
    batch_size: int = 4,
    seed: int = 42,
) -> dict:
    """Run baseline → fine-tune → evaluate on synthetic mock data.

    Returns a results dict with baseline and finetuned F1 scores.
    Clearly marked as synthetic — do not compare to the README's historical numbers.
    """
    from sentence_transformers import SentenceTransformer
    from src.train import TripletDatasetFromPairs, TrainConfig, train
    from src.evaluate import evaluate_model

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs_path = Path(pairs_path)
    if pairs_path.suffix == ".jsonl":
        pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
        pairs_df = pd.DataFrame(pairs)
    else:
        pairs_df = pd.read_parquet(pairs_path)

    logger.info("=== SYNTHETIC DEMO — results are NOT comparable to README numbers ===")

    # Verify triplets are constructable before loading the full model
    ds = TripletDatasetFromPairs(str(pairs_path), seed=seed)
    if len(ds) == 0:
        raise RuntimeError(
            "Zero triplets — regenerate mock data with: python src/generate_mock_data.py"
        )
    logger.info("Triplet check passed: %d triplets available", len(ds))

    # --- Baseline ---
    logger.info("Evaluating baseline model: %s", base_model)
    baseline_model = SentenceTransformer(base_model, trust_remote_code=True)
    baseline_result = evaluate_model(
        baseline_model, pairs_df,
        k_range=range(5, 31, 5),
        seed=seed,
    )
    logger.info(
        "Baseline — best_K=%d  train_F1=%.4f  test_F1=%.4f",
        baseline_result.best_k, baseline_result.train_f1, baseline_result.test_f1,
    )
    del baseline_model

    # --- Fine-tune ---
    ft_output = str(out / "finetuned")
    cfg = TrainConfig(
        model_name=base_model,
        triplets_path=str(pairs_path),
        output_dir=ft_output,
        epochs=epochs,
        contrastive_batch_size=batch_size,
        triplet_margin=0.3,
        seed=seed,
    )
    logger.info("Fine-tuning for %d epoch(s)...", epochs)
    train(cfg)

    # --- Fine-tuned eval ---
    task_dir = Path(ft_output) / f"task_{cfg.task_id:02d}" / "final"
    logger.info("Evaluating fine-tuned model from: %s", task_dir)
    ft_model = SentenceTransformer(str(task_dir), trust_remote_code=True)
    ft_result = evaluate_model(
        ft_model, pairs_df,
        k_range=range(5, 31, 5),
        seed=seed,
    )
    logger.info(
        "Fine-tuned — best_K=%d  train_F1=%.4f  test_F1=%.4f",
        ft_result.best_k, ft_result.train_f1, ft_result.test_f1,
    )

    results = {
        "note": "SYNTHETIC DEMO — not comparable to historical research results",
        "seed": seed,
        "epochs": epochs,
        "baseline": {
            "best_k": baseline_result.best_k,
            "train_f1": round(baseline_result.train_f1, 4),
            "test_f1": round(baseline_result.test_f1, 4),
            "n_train_pairs": baseline_result.n_train_pairs,
            "n_test_pairs": baseline_result.n_test_pairs,
        },
        "finetuned": {
            "best_k": ft_result.best_k,
            "train_f1": round(ft_result.train_f1, 4),
            "test_f1": round(ft_result.test_f1, 4),
            "n_train_pairs": ft_result.n_train_pairs,
            "n_test_pairs": ft_result.n_test_pairs,
        },
        "delta_test_f1": round(ft_result.test_f1 - baseline_result.test_f1, 4),
    }

    results_path = out / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", results_path)

    print("\n" + "=" * 60)
    print("SYNTHETIC DEMO RESULTS (mock data only)")
    print("=" * 60)
    print(f"  Baseline  test F1: {baseline_result.test_f1:.4f}")
    print(f"  Fine-tuned test F1: {ft_result.test_f1:.4f}")
    print(f"  Δ test F1:          {results['delta_test_f1']:+.4f}")
    print("=" * 60)
    print("  (Historical research result: +0.20 on proprietary corpus)")
    print("  See README for full context.\n")

    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline vs fine-tuned experiment on mock data")
    p.add_argument("--pairs", default="data/mock_pairs.jsonl")
    p.add_argument("--output", default="runs/experiment")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_experiment(
        pairs_path=args.pairs,
        output_dir=args.output,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
