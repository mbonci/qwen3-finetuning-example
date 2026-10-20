"""Pairwise classification F1 evaluation via cluster membership.

Implements the validation methodology used to produce the +0.20 F1 improvement
reported in the README. The key design decisions:

  - Positive = same cluster label, ground-truth score ∈ {1, 2, 3}
  - Negative = different cluster label, ground-truth score ∈ {5, 6, 7}
  - Scores 4 and 8 are excluded from evaluation (boundary / N/A)
  - Best-K is selected per topic on the TRAIN split only; the TEST split is
    held out until after K is chosen to prevent label leakage
  - Split is stratified within each topic (matching the private pipeline)
  - K is swept over a configurable range; ties broken by preferring larger K

Usage:
    python src/evaluate.py \\
        --pairs data/mock_pairs.jsonl \\
        --model Qwen/Qwen3-Embedding-0.6B \\
        --k-range 5 50

Usage (library):
    from src.evaluate import evaluate_model
    result = evaluate_model(model, pairs_df, k_range=range(5, 151, 5))
    print(result)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

POSITIVE_SCORES = {1, 2, 3}
NEGATIVE_SCORES = {5, 6, 7}
EVAL_SCORES = POSITIVE_SCORES | NEGATIVE_SCORES


@dataclass
class EvalResult:
    best_k: int
    train_f1: float
    train_precision: float
    train_recall: float
    test_f1: float
    test_precision: float
    test_recall: float
    n_train_pairs: int
    n_test_pairs: int
    k_sweep: dict[int, float]  # {K: train_f1}


def _pairwise_f1(
    labels: np.ndarray,
    doc_ids: list[str],
    pairs_df: pd.DataFrame,
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for pairwise classification via cluster membership."""
    id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    tp = fp = fn = 0
    for _, row in pairs_df.iterrows():
        a_idx = id_to_idx.get(row["anchor_id"])
        b_idx = id_to_idx.get(row["candidate_id"])
        if a_idx is None or b_idx is None:
            continue
        true_pos = int(row["similarity_score"]) in POSITIVE_SCORES
        pred_pos = labels[a_idx] == labels[b_idx] and labels[a_idx] != -1
        if pred_pos and true_pos:
            tp += 1
        elif pred_pos and not true_pos:
            fp += 1
        elif not pred_pos and true_pos:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def _kmeans_labels(embeddings: np.ndarray, k: int) -> np.ndarray:
    try:
        import faiss
        kmeans = faiss.Kmeans(embeddings.shape[1], k, niter=20, nredo=1, verbose=False)
        kmeans.train(np.ascontiguousarray(embeddings, dtype=np.float32))
        _, labels = kmeans.index.search(np.ascontiguousarray(embeddings, dtype=np.float32), 1)
        return labels.ravel()
    except ImportError:
        from sklearn.cluster import MiniBatchKMeans
        return MiniBatchKMeans(n_clusters=k, random_state=42, n_init=1).fit_predict(embeddings)


def evaluate_model(
    model,
    pairs_df: pd.DataFrame,
    k_range: range | None = None,
    train_frac: float = 0.8,
    seed: int = 42,
) -> EvalResult:
    """Evaluate a SentenceTransformer model with pairwise classification F1.

    Parameters
    ----------
    model:      SentenceTransformer instance (base or fine-tuned)
    pairs_df:   DataFrame with anchor_id, candidate_id, text_a, text_b, similarity_score
    k_range:    K values to sweep for K-Means; defaults to range(5, 51, 5)
    train_frac: Fraction of pairs used for best-K selection; rest is held out
    seed:       Random seed for train/test split
    """
    if k_range is None:
        k_range = range(5, 51, 5)

    eval_df = pairs_df[pairs_df["similarity_score"].isin(EVAL_SCORES)].copy()
    eval_df["similarity_score"] = eval_df["similarity_score"].astype(int)

    # Stratified split within each topic — matches private c00_split_random_pairs.py
    topic_col = "primary_topic_a" if "primary_topic_a" in eval_df.columns else None
    rng = np.random.default_rng(seed)
    train_chunks, test_chunks = [], []
    if topic_col is not None:
        for _, group in eval_df.groupby(topic_col):
            group = group.sample(frac=1, random_state=seed)
            n_test = max(1, int(len(group) * (1 - train_frac)))
            test_chunks.append(group.iloc[:n_test])
            train_chunks.append(group.iloc[n_test:])
        train_df = pd.concat(train_chunks).reset_index(drop=True)
        test_df = pd.concat(test_chunks).reset_index(drop=True)
    else:
        idx = rng.permutation(len(eval_df))
        n_train = int(len(eval_df) * train_frac)
        train_df = eval_df.iloc[idx[:n_train]]
        test_df = eval_df.iloc[idx[n_train:]]

    # Collect unique texts for embedding
    all_texts: dict[str, str] = {}
    for _, row in eval_df.iterrows():
        all_texts.setdefault(row["anchor_id"], row["text_a"])
        all_texts.setdefault(row["candidate_id"], row["text_b"])
    doc_ids = list(all_texts.keys())
    texts = list(all_texts.values())

    logger.info("Encoding %d unique documents...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 200,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Sweep K on TRAIN only
    k_sweep: dict[int, float] = {}
    for k in k_range:
        effective_k = min(k, len(doc_ids))
        labels = _kmeans_labels(embeddings, effective_k)
        _, _, f1 = _pairwise_f1(labels, doc_ids, train_df)
        k_sweep[k] = f1
        logger.info("K=%d  train_F1=%.4f", k, f1)

    # Best K: highest train F1, ties broken by larger K
    best_k = max(k_sweep, key=lambda k: (k_sweep[k], k))
    logger.info("Best K=%d (train F1=%.4f)", best_k, k_sweep[best_k])

    # Re-cluster with best K and evaluate both splits
    final_labels = _kmeans_labels(embeddings, min(best_k, len(doc_ids)))
    tr_prec, tr_rec, tr_f1 = _pairwise_f1(final_labels, doc_ids, train_df)
    te_prec, te_rec, te_f1 = _pairwise_f1(final_labels, doc_ids, test_df)

    logger.info(
        "Final — K=%d | train F1=%.4f | test F1=%.4f", best_k, tr_f1, te_f1
    )
    return EvalResult(
        best_k=best_k,
        train_f1=tr_f1, train_precision=tr_prec, train_recall=tr_rec,
        test_f1=te_f1, test_precision=te_prec, test_recall=te_rec,
        n_train_pairs=len(train_df),
        n_test_pairs=len(test_df),
        k_sweep=k_sweep,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate embedding model with pairwise cluster F1")
    p.add_argument("--pairs", default="data/mock_pairs.jsonl")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--k-range", nargs=2, type=int, default=[5, 50], metavar=("K_MIN", "K_MAX"))
    p.add_argument("--k-step", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer

    pairs_path = Path(args.pairs)
    if pairs_path.suffix == ".jsonl":
        pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
        pairs_df = pd.DataFrame(pairs)
    else:
        pairs_df = pd.read_parquet(pairs_path)

    model = SentenceTransformer(args.model, trust_remote_code=True)
    k_range = range(args.k_range[0], args.k_range[1] + 1, args.k_step)
    result = evaluate_model(model, pairs_df, k_range=k_range, seed=args.seed)

    print(f"\n{'='*50}")
    print(f"Best K: {result.best_k}")
    print(f"Train  — F1={result.train_f1:.4f}  P={result.train_precision:.4f}  R={result.train_recall:.4f}  n={result.n_train_pairs}")
    print(f"Test   — F1={result.test_f1:.4f}  P={result.test_precision:.4f}  R={result.test_recall:.4f}  n={result.n_test_pairs}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
