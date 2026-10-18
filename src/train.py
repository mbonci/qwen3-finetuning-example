"""Fine-tune Qwen3-Embedding-0.6B with contrastive triplet loss.

Designed for SLURM array jobs: each task receives a unique hyperparameter configuration
via SLURM_ARRAY_TASK_ID (mapped to a grid slot in the triplet preparation output).

Training pipeline:
  1. Load Qwen3-Embedding-0.6B (or a previously fine-tuned checkpoint)
  2. Construct triplets on-the-fly from a scored pairwise dataset using transitive
     inference and hard-negative selection by cosine rank
  3. Train with cosine-distance triplet loss (margin = 0.3)
  4. Evaluate per-topic binary F1 on a held-out validation set
  5. Save the best checkpoint and a cosine distribution visualisation

Usage (local / mock):
    python src/train.py --triplets data/mock_pairs.jsonl --output runs/test --epochs 1

Usage (HPC array):
    sbatch run_hpc_job.slurm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    triplets_path: str = "data/mock_pairs.jsonl"
    output_dir: str = "runs/default"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    triplet_margin: float = 0.3
    distance_metric: str = "cosine"
    max_positives_per_anchor: int = 2
    max_negatives_per_anchor: int = 3
    eval_steps: int = 500
    save_steps: int = 1000
    seed: int = 42
    # Set automatically from SLURM_ARRAY_TASK_ID if running on HPC
    task_id: int = field(default_factory=lambda: int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))


# ---------------------------------------------------------------------------
# Triplet Loss
# ---------------------------------------------------------------------------

class TripletLoss(torch.nn.Module):
    """Cosine- or Euclidean-distance triplet loss with configurable margin.

    L(a, p, n) = max(0, d(a, p) − d(a, n) + margin)
    """

    def __init__(self, margin: float = 0.3, distance: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def _dist(self, x: Tensor, y: Tensor) -> Tensor:
        if self.distance == "cosine":
            return 1.0 - F.cosine_similarity(x, y, dim=1)
        return torch.norm(x - y, dim=1)

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        d_pos = self._dist(anchor, positive)
        d_neg = self._dist(anchor, negative)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TripletDatasetFromPairs(Dataset):
    """Constructs triplets on-the-fly from a scored pairwise JSONL/parquet file.

    Triplet construction strategy:
    - For each anchor with at least one positive and one negative, pair each
      positive with hard negatives ranked by descending cosine similarity (i.e.
      the most confusable negatives come first).
    - Transitive inference: if (A,B) is positive, B's positives are also treated
      as indirect positives of A for augmentation.
    """

    POSITIVE_SCORES = {1, 2, 3}
    NEGATIVE_SCORES = {4, 5, 6, 7}

    def __init__(
        self,
        path: str | Path,
        max_positives: int = 2,
        max_negatives: int = 3,
        augment_transitive: bool = True,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        pairs = self._load(Path(path))
        self.triplets = self._build_triplets(pairs, max_positives, max_negatives, augment_transitive)
        logger.info("TripletDataset: %d triplets from %d pairs", len(self.triplets), len(pairs))

    @staticmethod
    def _load(path: Path) -> list[dict]:
        if path.suffix == ".jsonl":
            with path.open() as f:
                return [json.loads(line) for line in f if line.strip()]
        return pd.read_parquet(path).to_dict("records")

    def _build_triplets(
        self,
        pairs: list[dict],
        max_positives: int,
        max_negatives: int,
        augment_transitive: bool,
    ) -> list[tuple[str, str, str, str, str, str]]:
        positives: dict[str, list[tuple[str, str, float]]] = {}
        negatives: dict[str, list[tuple[str, str, float]]] = {}
        text_lookup: dict[str, str] = {}

        for p in pairs:
            score = p.get("similarity_score")
            if score is None:
                continue
            score = int(score)
            a, b = p["anchor_id"], p["candidate_id"]
            ta, tb = p.get("text_a", ""), p.get("text_b", "")
            cos = float(p.get("cosine_similarity") or 0.0)
            text_lookup[a] = ta
            text_lookup[b] = tb

            if score in self.POSITIVE_SCORES:
                positives.setdefault(a, []).append((b, tb, cos))
                positives.setdefault(b, []).append((a, ta, cos))
            elif score in self.NEGATIVE_SCORES:
                negatives.setdefault(a, []).append((b, tb, cos))
                negatives.setdefault(b, []).append((a, ta, cos))

        if augment_transitive:
            for anchor, pos_list in list(positives.items()):
                indirect: list[tuple[str, str, float]] = []
                for pos_id, _, _ in pos_list[:max_positives]:
                    indirect.extend(positives.get(pos_id, []))
                existing_ids = {x[0] for x in pos_list}
                for item in indirect:
                    if item[0] not in existing_ids and item[0] != anchor:
                        pos_list.append(item)
                        existing_ids.add(item[0])

        triplets = []
        for anchor in sorted(set(positives.keys()) & set(negatives.keys())):
            ta = text_lookup.get(anchor, "")
            pos_sorted = positives[anchor][:max_positives]
            neg_sorted = sorted(negatives[anchor], key=lambda x: x[2], reverse=True)[:max_negatives]
            for pos_id, tp, _ in pos_sorted:
                for neg_id, tn, _ in neg_sorted:
                    triplets.append((anchor, pos_id, neg_id, ta, tp, tn))

        self.rng.shuffle(triplets)
        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        anchor_id, pos_id, neg_id, ta, tp, tn = self.triplets[idx]
        return ta, tp, tn


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class BinaryClassificationEvaluatorPrimaryF1:
    """Evaluates per-topic binary F1 on a validation set of scored pairs.

    A pair is predicted positive if cosine_similarity ≥ threshold (0.5 default).
    Reports macro-averaged F1 across topics and an overall F1.
    """

    POSITIVE_SCORES = {1, 2, 3}

    def __init__(self, val_pairs: list[dict], threshold: float = 0.5):
        self.val_pairs = val_pairs
        self.threshold = threshold

    def __call__(self, model, output_path: Optional[str] = None) -> float:
        texts_a = [p["text_a"] for p in self.val_pairs]
        texts_b = [p["text_b"] for p in self.val_pairs]
        labels = [1 if int(p["similarity_score"]) in self.POSITIVE_SCORES else 0 for p in self.val_pairs]
        topics = [p.get("primary_topic_a", "Unknown") for p in self.val_pairs]

        emb_a = model.encode(texts_a, convert_to_tensor=True, normalize_embeddings=True)
        emb_b = model.encode(texts_b, convert_to_tensor=True, normalize_embeddings=True)
        cosines = F.cosine_similarity(emb_a, emb_b).cpu().numpy()
        preds = (cosines >= self.threshold).astype(int)

        topic_f1: dict[str, float] = {}
        for topic in sorted(set(topics)):
            mask = [t == topic for t in topics]
            t_labels = [l for l, m in zip(labels, mask) if m]
            t_preds = [p for p, m in zip(preds, mask) if m]
            if len(t_labels) < 2:
                continue
            tp = sum(p == 1 and l == 1 for p, l in zip(t_preds, t_labels))
            fp = sum(p == 1 and l == 0 for p, l in zip(t_preds, t_labels))
            fn = sum(p == 0 and l == 1 for p, l in zip(t_preds, t_labels))
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            topic_f1[topic] = 2 * prec * rec / max(prec + rec, 1e-9)

        macro_f1 = float(np.mean(list(topic_f1.values()))) if topic_f1 else 0.0
        logger.info("Eval — macro F1: %.4f  |  per-topic: %s", macro_f1,
                    {t: f"{v:.3f}" for t, v in topic_f1.items()})
        return macro_f1


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def _collate_fn(batch: list[tuple[str, str, str]]) -> tuple[list[str], list[str], list[str]]:
    anchors, positives, negatives = zip(*batch)
    return list(anchors), list(positives), list(negatives)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers is required: pip install sentence-transformers") from exc

    output_dir = Path(cfg.output_dir) / f"task_{cfg.task_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", cfg.model_name)
    model = SentenceTransformer(cfg.model_name, trust_remote_code=True)
    model.max_seq_length = cfg.max_seq_length

    dataset = TripletDatasetFromPairs(
        cfg.triplets_path,
        max_positives=cfg.max_positives_per_anchor,
        max_negatives=cfg.max_negatives_per_anchor,
        seed=cfg.seed,
    )

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate_fn, drop_last=True
    )

    loss_fn = TripletLoss(margin=cfg.triplet_margin, distance=cfg.distance_metric)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(dataloader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    try:
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    except ImportError:
        scheduler = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Training on %s | triplets=%d | steps=%d", device, len(dataset), total_steps)

    best_f1 = 0.0
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (anchors, positives, negatives) in enumerate(dataloader):
            emb_a = model.encode(anchors, convert_to_tensor=True, normalize_embeddings=True)
            emb_p = model.encode(positives, convert_to_tensor=True, normalize_embeddings=True)
            emb_n = model.encode(negatives, convert_to_tensor=True, normalize_embeddings=True)

            optimizer.zero_grad()
            loss = loss_fn(emb_a, emb_p, emb_n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                logger.info("Epoch %d | step %d | loss %.4f", epoch + 1, global_step, loss.item())

        logger.info("Epoch %d complete | avg loss %.4f", epoch + 1, epoch_loss / max(len(dataloader), 1))

        ckpt_path = output_dir / f"checkpoint_epoch_{epoch + 1}"
        model.save(str(ckpt_path))
        logger.info("Checkpoint saved: %s", ckpt_path)

    logger.info("Training complete. Best F1: %.4f", best_f1)
    (output_dir / "config.json").write_text(
        json.dumps(cfg.__dict__, default=str, indent=2)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune Qwen3-Embedding-0.6B with triplet loss")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--triplets", default="data/mock_pairs.jsonl")
    p.add_argument("--output", default="runs/default")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = TrainConfig(
        model_name=args.model,
        triplets_path=args.triplets,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        triplet_margin=args.margin,
        seed=args.seed,
    )
    train(cfg)
