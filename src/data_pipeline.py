"""Contrastive data pipeline: BM25 candidate mining and LLM-as-a-judge scoring.

Stage 1 — mine_candidates():
  Builds a BM25 index per topic and retrieves the top-k lexically similar statutes
  for each sampled anchor. Merges BM25 hits (80%) with random within-topic pairs (20%),
  computes cosine similarity, bins by quantile, and stratified-samples to the target quota.

Stage 2 — score_pairs_async():
  Sends each pair to gpt-4.1-mini with an 8-point ordinal rubric via an async worker
  pool. Returns similarity_score ∈ {1..8}; scores 1-3 map to positives, 4-7 to negatives,
  8 = not applicable.

Stage 3 — build_triplets():
  Converts the scored pair table into (anchor, positive, negative) triplets using
  transitive inference and cosine-rank-based hard-negative selection.

Usage (mock / offline):
    from src.data_pipeline import mine_candidates, build_triplets
    candidates = mine_candidates(texts, topic_map)
    # Score with a real OpenAI key: scored = await score_pairs_async(candidates)
    # Or attach mock scores and go straight to triplets:
    candidates["similarity_score"] = 2
    triplets = build_triplets(candidates)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COSINE_BINS = ["q40_60", "q60_80", "q80_90", "q90_95", "q95_100"]

# Stratified sampling weights: (candidate_source, cosine_bin) → unnormalized weight.
# BM25 pairs total ~75% of the final sample; random pairs ~25%.
# Higher bins (q80_90 and above) receive more weight — these are the hard cases
# near the positive/negative decision boundary that carry the most training signal.
STRATIFIED_WEIGHTS: dict[tuple[str, str], float] = {
    ("bm25",   "q40_60"):  0.06,
    ("bm25",   "q60_80"):  0.08,
    ("bm25",   "q80_90"):  0.18,
    ("bm25",   "q90_95"):  0.16,
    ("bm25",   "q95_100"): 0.12,
    ("random", "q40_60"):  0.02,
    ("random", "q60_80"):  0.02,
    ("random", "q80_90"):  0.06,
    ("random", "q90_95"):  0.06,
    ("random", "q95_100"): 0.04,
}

POSITIVE_SCORES = {1, 2, 3}
NEGATIVE_SCORES = {4, 5, 6, 7}

JUDGE_SYSTEM_PROMPT = (
    "You are an expert legal analyst. Rate how related two statutory policy provisions are "
    "on a scale from 1 to 8, where:\n"
    "  1 = Definitely related (same policy area, close regulatory content)\n"
    "  2 = Related (same policy area, somewhat different scope)\n"
    "  3 = Probably related (overlapping policy concerns)\n"
    "  4 = Probably not related (surface similarity only)\n"
    "  5 = Not related (different regulatory domains)\n"
    "  6 = Clearly not related\n"
    "  7 = Definitely not related\n"
    "  8 = Cannot determine / not applicable\n\n"
    "Respond with a single JSON object: {\"similarity_score\": <integer 1-8>}"
)

JUDGE_USER_TEMPLATE = (
    "Statute A:\n{text_a}\n\nStatute B:\n{text_b}\n\n"
    "Provide your similarity_score as JSON."
)


# ---------------------------------------------------------------------------
# Stage 1: Candidate Mining
# ---------------------------------------------------------------------------


def _assign_per_topic_cosine_bins(
    df: pd.DataFrame,
    cosim_col: str = "cosine_similarity",
    topic_col: str = "topic_a",
    quantile_edges: list[float] | None = None,
) -> np.ndarray:
    """Assign cosine similarity values to percentile-based bins computed within each topic.

    Thresholds are computed per topic so that the bin boundaries reflect the
    distribution of each topic rather than a global absolute threshold. Pairs
    below the 40th percentile of their topic are assigned None and excluded
    from stratified sampling.
    """
    if quantile_edges is None:
        quantile_edges = [0.40, 0.60, 0.80, 0.90, 0.95]

    result = pd.Series(index=df.index, dtype=object)
    for _, group in df.groupby(topic_col):
        thresholds = np.quantile(group[cosim_col].values, quantile_edges)
        bin_ids = np.digitize(group[cosim_col].values, bins=thresholds)
        # bin_id 0 → below 40th percentile → excluded (None)
        result.loc[group.index] = [
            COSINE_BINS[b - 1] if b > 0 else None for b in bin_ids
        ]
    return result.values


def _cosine_similarity_batched(
    embeddings: np.ndarray, idx_a: list[int], idx_b: list[int], batch_size: int = 50_000
) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-9, None)
    scores = []
    for start in range(0, len(idx_a), batch_size):
        sl = slice(start, start + batch_size)
        scores.append(
            np.einsum("ij,ij->i", normed[idx_a[sl]], normed[idx_b[sl]])
        )
    return np.concatenate(scores)


def mine_candidates(
    texts: dict[str, str],
    topic_map: dict[str, str],
    embeddings: dict[str, np.ndarray] | None = None,
    n_random: int = 2_500_000,
    bm25_anchor_frac: float = 0.05,
    min_anchors_per_topic: int = 1_000,
    bm25_top_k: int = 100,
    max_total_pairs: int = 100_000,
    state_prefix_fn: Any | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Mine hard-negative candidates via BM25 and random within-topic sampling.

    Candidate generation:
    - BM25 (bm25s): one index per topic, top-k retrieval per sampled anchor
    - Random within-topic: enforces different-state pairs when state_prefix_fn is set
    - Merges both, deduplicates (BM25 takes priority)

    Cosine binning (requires embeddings):
    - Per-topic percentile bins: q40_60, q60_80, q80_90, q90_95, q95_100
    - Pairs below the 40th percentile of their topic are excluded
    - Stratified sampling from 10 (source × bin) strata using STRATIFIED_WEIGHTS

    Parameters
    ----------
    texts:                {doc_id: text}
    topic_map:            {doc_id: topic_label}
    embeddings:           Optional {doc_id: vector} for cosine similarity + bin assignment
    n_random:             Random within-topic candidate pairs to generate
    bm25_anchor_frac:     Fraction of docs per topic used as BM25 query anchors
    min_anchors_per_topic Minimum anchor count per topic regardless of frac
    bm25_top_k:           Top-k BM25 hits per anchor
    max_total_pairs:      Final stratified-sample target
    state_prefix_fn:      Optional callable (doc_id -> state_key); pairs with the same
                          state key are excluded. None = no exclusion.
    seed:                 Random seed
    """
    try:
        import bm25s
    except ImportError as exc:
        raise ImportError("bm25s is required: pip install bm25s") from exc

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    doc_ids = list(texts.keys())
    topic_groups: dict[str, list[str]] = {}
    for doc_id in doc_ids:
        t = topic_map.get(doc_id, "Unknown")
        topic_groups.setdefault(t, []).append(doc_id)

    # --- BM25 candidates ---
    bm25_pairs: list[tuple[str, str, str]] = []  # (anchor, candidate, topic)
    for topic, members in topic_groups.items():
        n_anchors = min(len(members), max(min_anchors_per_topic, int(len(members) * bm25_anchor_frac)))
        anchors = rng.sample(members, min(n_anchors, len(members)))

        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize([texts[m] for m in members]))

        # Filter anchors with non-empty text, then batch-retrieve all at once
        valid_anchors = [(a, texts[a]) for a in anchors if texts.get(a)]
        if not valid_anchors:
            continue
        valid_ids, valid_queries = zip(*valid_anchors)

        k = min(bm25_top_k + 1, len(members))
        batch_results, _ = retriever.retrieve(bm25s.tokenize(list(valid_queries)), k=k)

        for i, anchor in enumerate(valid_ids):
            for idx in batch_results[i]:
                candidate = members[int(idx)]
                if candidate == anchor:
                    continue
                if state_prefix_fn is not None and state_prefix_fn(anchor) == state_prefix_fn(candidate):
                    continue
                bm25_pairs.append((anchor, candidate, topic))

    # --- Random within-topic candidates ---
    random_pairs: list[tuple[str, str, str]] = []
    topic_list = list(topic_groups.keys())
    attempts = 0
    seen_random: set[tuple[str, str]] = set()
    while len(random_pairs) < n_random and attempts < n_random * 10:
        attempts += 1
        topic = rng.choice(topic_list)
        members = topic_groups[topic]
        if len(members) < 2:
            continue
        a, b = rng.sample(members, 2)
        if state_prefix_fn is not None and state_prefix_fn(a) == state_prefix_fn(b):
            continue
        key = (min(a, b), max(a, b))
        if key in seen_random:
            continue
        seen_random.add(key)
        random_pairs.append((a, b, topic))

    # --- Merge and deduplicate (BM25 priority) ---
    seen_global: set[tuple[str, str]] = set()
    rows: list[dict] = []

    for a, b, topic in bm25_pairs:
        key = (min(a, b), max(a, b))
        if key in seen_global:
            continue
        seen_global.add(key)
        rows.append({"anchor_id": a, "candidate_id": b, "topic_a": topic,
                     "text_a": texts[a], "text_b": texts[b],
                     "candidate_source": "bm25"})

    for a, b, topic in random_pairs:
        key = (min(a, b), max(a, b))
        if key in seen_global:
            continue
        seen_global.add(key)
        rows.append({"anchor_id": a, "candidate_id": b, "topic_a": topic,
                     "text_a": texts[a], "text_b": texts[b],
                     "candidate_source": "random"})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # --- Cosine similarity + per-topic percentile bins ---
    df["cosine_similarity"] = None
    df["bin"] = None

    if embeddings is not None:
        emb_matrix = np.array([embeddings[d] for d in doc_ids], dtype=np.float32)
        id_to_idx = {d: i for i, d in enumerate(doc_ids)}
        idx_a = [id_to_idx[r] for r in df["anchor_id"]]
        idx_b = [id_to_idx[r] for r in df["candidate_id"]]
        cosines = _cosine_similarity_batched(emb_matrix, idx_a, idx_b)
        df["cosine_similarity"] = cosines
        df["bin"] = _assign_per_topic_cosine_bins(df)

        # Exclude pairs below the 40th percentile of their topic
        df = df[df["bin"].notna()].reset_index(drop=True)

        # --- Stratified sampling from 10 strata ---
        total_weight = sum(STRATIFIED_WEIGHTS.values())
        norm_weights = {k: v / total_weight for k, v in STRATIFIED_WEIGHTS.items()}
        targets = {k: int(v * max_total_pairs) for k, v in norm_weights.items()}

        chunks: list[pd.DataFrame] = []
        for (source, cbin), target in targets.items():
            if target == 0:
                continue
            pool = df[(df["candidate_source"] == source) & (df["bin"] == cbin)]
            if pool.empty:
                continue
            n_sample = min(target, len(pool))
            chunks.append(pool.sample(n=n_sample, random_state=seed))

        if chunks:
            df = pd.concat(chunks).reset_index(drop=True)
            # Top up if under target
            if len(df) < max_total_pairs:
                sampled_keys = set(zip(df["anchor_id"], df["candidate_id"]))
                remaining = df.iloc[0:0]  # empty placeholder — not actually used
                # Re-sample from full pool excluding already-sampled keys
                # (simple approach: trim or leave as-is)
        if len(df) > max_total_pairs:
            df = df.sample(n=max_total_pairs, random_state=seed).reset_index(drop=True)
    else:
        # No embeddings: simple source-proportional sampling without binning
        n_bm25 = int(max_total_pairs * 0.75)
        n_rand = max_total_pairs - n_bm25
        bm25_df = df[df["candidate_source"] == "bm25"]
        rand_df = df[df["candidate_source"] == "random"]
        df = pd.concat([
            bm25_df.sample(n=min(n_bm25, len(bm25_df)), random_state=seed),
            rand_df.sample(n=min(n_rand, len(rand_df)), random_state=seed),
        ]).reset_index(drop=True)

    logger.info(
        "Candidate mining complete: %d pairs (bm25=%d, random=%d)",
        len(df),
        (df["candidate_source"] == "bm25").sum(),
        (df["candidate_source"] == "random").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Stage 2: LLM-as-a-Judge Scoring
# ---------------------------------------------------------------------------


async def _score_one(
    client: Any,
    model: str,
    row: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> dict:
    import json as _json

    user_msg = JUDGE_USER_TEMPLATE.format(text_a=row["text_a"], text_b=row["text_b"])
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                )
            content = response.choices[0].message.content
            parsed = _json.loads(content)
            score = int(parsed["similarity_score"])
            assert 1 <= score <= 8
            return {**row, "similarity_score": score}
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                logger.warning("Failed to score pair after %d retries: %s", max_retries, exc)
                return {**row, "similarity_score": None}
            wait = min(2 ** attempt + random.random(), 60.0)
            await asyncio.sleep(wait)
    return {**row, "similarity_score": None}


async def _score_pairs_async_inner(
    pairs_df: pd.DataFrame,
    model: str,
    concurrency: int,
    checkpoint_path: str | None,
    checkpoint_every: int,
) -> pd.DataFrame:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("openai is required: pip install openai") from exc

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    rows = pairs_df.to_dict("records")
    results: list[dict] = []

    # Process in checkpoint-sized batches; within each batch all requests fire
    # concurrently (bounded by the semaphore), not sequentially.
    for batch_start in range(0, len(rows), checkpoint_every):
        batch = rows[batch_start : batch_start + checkpoint_every]
        batch_results = await asyncio.gather(
            *[_score_one(client, model, row, semaphore) for row in batch]
        )
        results.extend(batch_results)
        if checkpoint_path:
            pd.DataFrame(results).to_parquet(checkpoint_path, index=False)
            logger.info("Checkpoint saved at row %d", batch_start + len(batch))

    return pd.DataFrame(results)


def score_pairs_async(
    pairs_df: pd.DataFrame,
    model: str = "gpt-4.1-mini",
    concurrency: int = 35,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 20_000,
) -> pd.DataFrame:
    """Score pairs using an async OpenAI worker pool.

    Returns the input DataFrame with an added `similarity_score` column (int 1–8).
    Requires OPENAI_API_KEY to be set in the environment.
    """
    return asyncio.run(
        _score_pairs_async_inner(pairs_df, model, concurrency, checkpoint_path, checkpoint_every)
    )


# ---------------------------------------------------------------------------
# Stage 3: Triplet Construction
# ---------------------------------------------------------------------------


def build_triplets(
    scored_df: pd.DataFrame,
    max_positives_per_anchor: int = 2,
    max_negatives_per_positive: int = 3,
    target_per_topic: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Build (anchor, positive, negative) triplets from scored pairwise labels.

    Uses transitive inference: if (A, B) is positive and (A, C) is negative,
    the triplet (A, B, C) is valid without requiring a direct (B, C) label.
    Hard negatives are selected by cosine rank when embeddings are available.

    Returns DataFrame with columns: anchor_id, positive_id, negative_id,
    anchor_text, positive_text, negative_text, topic.
    """
    rng = random.Random(seed)

    df = scored_df.copy()
    if "similarity_score" not in df.columns:
        raise ValueError("scored_df must contain a similarity_score column")

    df_valid = df[df["similarity_score"].notna()].copy()
    df_valid["similarity_score"] = df_valid["similarity_score"].astype(int)

    positives: dict[str, list[str]] = {}
    negatives: dict[str, list[str]] = {}

    for _, row in df_valid.iterrows():
        a, b, score = row["anchor_id"], row["candidate_id"], row["similarity_score"]
        if score in POSITIVE_SCORES:
            positives.setdefault(a, []).append(b)
            positives.setdefault(b, []).append(a)
        elif score in NEGATIVE_SCORES:
            negatives.setdefault(a, []).append(b)
            negatives.setdefault(b, []).append(a)

    text_lookup: dict[str, str] = {}
    for _, row in df_valid.iterrows():
        text_lookup[row["anchor_id"]] = row["text_a"]
        text_lookup[row["candidate_id"]] = row["text_b"]

    topic_lookup: dict[str, str] = {}
    if "primary_topic_a" in df_valid.columns:
        for _, row in df_valid.iterrows():
            topic_lookup.setdefault(row["anchor_id"], row["primary_topic_a"])

    triplets = []
    anchors = list(set(positives.keys()) & set(negatives.keys()))
    rng.shuffle(anchors)

    for anchor in anchors:
        pos_list = positives[anchor]
        neg_list = negatives[anchor]

        if "cosine_similarity" in df_valid.columns:
            neg_cosines = {}
            for _, row in df_valid[
                (df_valid["anchor_id"] == anchor) & (df_valid["similarity_score"].isin(NEGATIVE_SCORES))
            ].iterrows():
                neg_cosines[row["candidate_id"]] = float(row["cosine_similarity"] or 0)
            neg_list = sorted(neg_list, key=lambda x: neg_cosines.get(x, 0), reverse=True)

        sampled_pos = pos_list[: max_positives_per_anchor]
        for positive in sampled_pos:
            sampled_neg = neg_list[: max_negatives_per_positive]
            for negative in sampled_neg:
                triplets.append({
                    "anchor_id": anchor,
                    "positive_id": positive,
                    "negative_id": negative,
                    "anchor_text": text_lookup.get(anchor, ""),
                    "positive_text": text_lookup.get(positive, ""),
                    "negative_text": text_lookup.get(negative, ""),
                    "topic": topic_lookup.get(anchor, "Unknown"),
                })

    result = pd.DataFrame(triplets)
    logger.info("Built %d triplets from %d scored pairs", len(result), len(df_valid))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Data pipeline: mine or score pairs")
    p.add_argument("--mode", choices=["mine", "score"], default="mine")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--n-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--concurrency", type=int, default=35)
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "score":
        if Path(args.input).suffix == ".jsonl":
            with open(args.input) as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            input_df = pd.DataFrame(rows)
        else:
            input_df = pd.read_parquet(args.input)

        scored = score_pairs_async(
            input_df,
            concurrency=args.concurrency,
            checkpoint_path=args.checkpoint,
        )
        out_path = out_dir / f"scored_task_{args.task_id:02d}.parquet"
        scored.to_parquet(out_path, index=False)
        logger.info("Scored %d pairs → %s", len(scored), out_path)

    elif args.mode == "mine":
        logger.info("Mine mode: loading texts from %s (task %d/%d)", args.input, args.task_id, args.n_tasks)
        if Path(args.input).suffix == ".jsonl":
            with open(args.input) as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        else:
            rows = pd.read_parquet(args.input).to_dict("records")

        texts: dict[str, str] = {}
        topic_map: dict[str, str] = {}
        for row in rows:
            texts.setdefault(row["anchor_id"], row.get("text_a", ""))
            texts.setdefault(row["candidate_id"], row.get("text_b", ""))
            if "primary_topic_a" in row:
                topic_map.setdefault(row["anchor_id"], row["primary_topic_a"])

        # Shard the doc list by task_id for array parallelism
        all_ids = sorted(texts.keys())
        shard_ids = all_ids[args.task_id :: args.n_tasks]
        shard_texts = {k: texts[k] for k in shard_ids}
        shard_topics = {k: topic_map.get(k, "Unknown") for k in shard_ids}

        candidates = mine_candidates(shard_texts, shard_topics)
        out_path = out_dir / f"pairs_task_{args.task_id:02d}.parquet"
        candidates.to_parquet(out_path, index=False)
        logger.info("Mined %d pairs → %s", len(candidates), out_path)
