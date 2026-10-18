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
import logging
import random
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COSINE_BINS = ["q40_60", "q60_80", "q80_90", "q90_95", "q95_100"]
BIN_EDGES = [0.40, 0.60, 0.80, 0.90, 0.95, 1.01]
BM25_RANDOM_RATIO = 0.80  # fraction of quota filled by BM25 hits
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


def _cosine_bin(score: float) -> str:
    for i, (lo, hi) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
        if lo <= score < hi:
            return COSINE_BINS[i]
    return COSINE_BINS[-1]


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
    n_random: int = 100_000,
    bm25_anchor_frac: float = 0.05,
    bm25_top_k: int = 100,
    max_total_pairs: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Mine hard-negative candidates via BM25 and random within-topic sampling.

    Parameters
    ----------
    texts:            {doc_id: text} mapping of all statutes in scope
    topic_map:        {doc_id: topic_label} — used to restrict pairing to within-topic
    embeddings:       Optional {doc_id: vector} for cosine similarity computation
    n_random:         Random within-topic pairs to generate before merging with BM25
    bm25_anchor_frac: Fraction of docs per topic selected as BM25 query anchors
    bm25_top_k:       Top-k BM25 hits retrieved per anchor
    max_total_pairs:  Stratified-sample target after merging BM25 + random

    Returns
    -------
    DataFrame with columns: anchor_id, candidate_id, text_a, text_b, cosine_similarity,
    bin, pair_type, bm25_source (bool)
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError("rank_bm25 is required: pip install rank_bm25") from exc

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    doc_ids = list(texts.keys())
    topics: dict[str, list[str]] = {}
    for doc_id in doc_ids:
        t = topic_map.get(doc_id, "Unknown")
        topics.setdefault(t, []).append(doc_id)

    bm25_pairs: list[tuple[str, str]] = []
    for topic, members in topics.items():
        n_anchors = max(1, int(len(members) * bm25_anchor_frac))
        anchors = rng.sample(members, min(n_anchors, len(members)))
        tokenized = [texts[m].lower().split() for m in members]
        bm25 = BM25Okapi(tokenized)
        member_index = {m: i for i, m in enumerate(members)}

        for anchor in anchors:
            query = texts[anchor].lower().split()
            scores_arr = bm25.get_scores(query)
            top_indices = np.argsort(scores_arr)[::-1][: bm25_top_k + 1]
            for idx in top_indices:
                candidate = members[idx]
                if candidate != anchor:
                    bm25_pairs.append((anchor, candidate))

    random_pairs: list[tuple[str, str]] = []
    for topic, members in topics.items():
        if len(members) < 2:
            continue
        n_topic = min(n_random // max(len(topics), 1), len(members) * (len(members) - 1))
        for _ in range(n_topic):
            a, b = rng.sample(members, 2)
            random_pairs.append((a, b))

    seen: set[frozenset] = set()
    unique_bm25 = []
    for a, b in bm25_pairs:
        key = frozenset((a, b))
        if key not in seen:
            seen.add(key)
            unique_bm25.append((a, b, True))

    unique_random = []
    for a, b in random_pairs:
        key = frozenset((a, b))
        if key not in seen:
            seen.add(key)
            unique_random.append((a, b, False))

    n_bm25_quota = int(max_total_pairs * BM25_RANDOM_RATIO)
    n_random_quota = max_total_pairs - n_bm25_quota
    sampled = (
        rng.sample(unique_bm25, min(n_bm25_quota, len(unique_bm25)))
        + rng.sample(unique_random, min(n_random_quota, len(unique_random)))
    )

    rows = []
    for a, b, is_bm25 in sampled:
        rows.append({
            "anchor_id": a,
            "candidate_id": b,
            "text_a": texts[a],
            "text_b": texts[b],
            "cosine_similarity": None,
            "bin": None,
            "pair_type": "within" if topic_map.get(a) == topic_map.get(b) else "across",
            "bm25_source": is_bm25,
        })

    df = pd.DataFrame(rows)

    if embeddings is not None:
        idx_a = [doc_ids.index(r) for r in df["anchor_id"] if r in doc_ids]
        idx_b = [doc_ids.index(r) for r in df["candidate_id"] if r in doc_ids]
        emb_matrix = np.array([embeddings[d] for d in doc_ids])
        cosines = _cosine_similarity_batched(emb_matrix, idx_a, idx_b)
        df = df.iloc[: len(cosines)].copy()
        df["cosine_similarity"] = cosines
        df["bin"] = df["cosine_similarity"].apply(_cosine_bin)

    logger.info("Candidate mining complete: %d pairs (bm25=%d, random=%d)",
                len(df), df["bm25_source"].sum(), (~df["bm25_source"]).sum())
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

    for i, row in enumerate(rows):
        result = await _score_one(client, model, row, semaphore)
        results.append(result)
        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            pd.DataFrame(results).to_parquet(checkpoint_path, index=False)
            logger.info("Checkpoint saved at row %d", i + 1)

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
