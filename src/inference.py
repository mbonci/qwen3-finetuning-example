"""Fast shard-based embedding extraction for large-scale corpora.

Designed for HPC: the corpus is divided into N shards and this script processes
shard SLURM_ARRAY_TASK_ID / N. Each shard is read in row groups and written to
a separate zstd-compressed parquet file with embeddings stored as Arrow
fixed-size lists (float32).

Usage (local, single shard):
    python src/inference.py \\
        --input data/mock_pairs.jsonl \\
        --output embeddings/ \\
        --model Qwen/Qwen3-Embedding-0.6B \\
        --shard 0 --n-shards 1

Usage (HPC array, 24 shards):
    sbatch run_hpc_job.slurm  # dispatches SLURM_ARRAY_TASK_ID=0..23
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2; use 1024 for Qwen3-Embedding-0.6B


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load Qwen3-Embedding model from a local path or HuggingFace hub.

    If model_path points to a local directory that exists, loads from there
    (e.g. a fine-tuned checkpoint). Otherwise falls back to HuggingFace.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers is required: pip install sentence-transformers") from exc

    local = Path(model_path)
    source = str(local) if local.exists() else model_path
    logger.info("Loading model from: %s", source)
    model = SentenceTransformer(source, trust_remote_code=True)
    model.max_seq_length = 512
    return model


# ---------------------------------------------------------------------------
# Safe encoding with CUDA OOM retry
# ---------------------------------------------------------------------------

def safe_encode(
    model,
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = True,
    max_retries: int = 4,
) -> np.ndarray:
    """Encode texts with automatic batch-size halving on CUDA OOM.

    Retries up to max_retries times, halving batch_size on each OOM error.
    Returns a float32 numpy array of shape (len(texts), embedding_dim).
    """
    import torch

    current_batch = batch_size
    for attempt in range(max_retries):
        try:
            embeddings = model.encode(
                texts,
                batch_size=current_batch,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 1000,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc) and current_batch > 1:
                current_batch = max(1, current_batch // 2)
                logger.warning("CUDA OOM — retrying with batch_size=%d", current_batch)
                torch.cuda.empty_cache()
            else:
                raise
    raise RuntimeError(f"safe_encode failed after {max_retries} attempts")


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------

def _texts_from_jsonl(path: Path) -> list[tuple[str, str]]:
    import json

    records = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            for id_col, text_col in [("anchor_id", "text_a"), ("candidate_id", "text_b")]:
                records.append((row[id_col], row[text_col]))
    seen: dict[str, str] = {}
    for doc_id, text in records:
        seen.setdefault(doc_id, text)
    return list(seen.items())


def _texts_from_parquet(path: Path) -> list[tuple[str, str]]:
    import pandas as pd

    df = pd.read_parquet(path)
    seen: dict[str, str] = {}
    for _, row in df.iterrows():
        seen.setdefault(row["anchor_id"], row["text_a"])
        seen.setdefault(row["candidate_id"], row["text_b"])
    return list(seen.items())


def embed_shards(
    input_path: str | Path,
    output_dir: str | Path,
    shard_id: int,
    n_shards: int,
    model_path: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    """Embed the shard_id-th slice of input_path and write to output_dir.

    Output filename: shard_{shard_id:04d}_of_{n_shards:04d}.parquet
    Schema: doc_id (string), embedding (fixed_size_list<float32>[EMBEDDING_DIM])

    Parameters
    ----------
    input_path:  JSONL or parquet file with text columns
    output_dir:  Directory to write output parquet files
    shard_id:    Zero-based shard index (from SLURM_ARRAY_TASK_ID)
    n_shards:    Total number of shards
    model_path:  Local checkpoint path or HuggingFace model ID
    batch_size:  Initial encoding batch size (halved on CUDA OOM)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix == ".jsonl":
        all_docs = _texts_from_jsonl(input_path)
    else:
        all_docs = _texts_from_parquet(input_path)

    shard_docs = all_docs[shard_id::n_shards]
    logger.info("Shard %d/%d: %d documents", shard_id, n_shards, len(shard_docs))

    if not shard_docs:
        logger.warning("Empty shard — nothing to encode")
        return output_dir

    model = load_model(model_path)

    t0 = time.time()
    doc_ids = [d[0] for d in shard_docs]
    texts = [d[1] for d in shard_docs]
    embeddings = safe_encode(model, texts, batch_size=batch_size)
    elapsed = time.time() - t0
    logger.info(
        "Encoded %d docs in %.1fs (%.0f docs/s)", len(doc_ids), elapsed, len(doc_ids) / elapsed
    )

    out_path = output_dir / f"shard_{shard_id:04d}_of_{n_shards:04d}.parquet"
    _write_embeddings_parquet(doc_ids, embeddings, out_path)
    logger.info("Wrote %s", out_path)
    return out_path


def _write_embeddings_parquet(
    doc_ids: list[str], embeddings: np.ndarray, out_path: Path
) -> None:
    dim = embeddings.shape[1]
    list_type = pa.list_(pa.float32(), dim)
    emb_arrays = [pa.array(embeddings[i].tolist(), type=pa.float32()) for i in range(len(embeddings))]
    emb_col = pa.chunked_array(
        [pa.FixedSizeListArray.from_arrays(
            pa.concat_arrays(emb_arrays), type=list_type
        )]
    )
    table = pa.table({"doc_id": doc_ids, "embedding": emb_col})
    pq.write_table(table, out_path, compression="zstd")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shard-based embedding extraction for Qwen3")
    p.add_argument("--input", default="data/mock_pairs.jsonl")
    p.add_argument("--output", default="embeddings/")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--shard",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
    )
    p.add_argument("--n-shards", type=int, default=1)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    embed_shards(
        input_path=args.input,
        output_dir=args.output,
        shard_id=args.shard,
        n_shards=args.n_shards,
        model_path=args.model,
        batch_size=args.batch_size,
    )
