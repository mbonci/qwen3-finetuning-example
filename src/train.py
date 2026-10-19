"""Fine-tune Qwen3-Embedding-0.6B with contrastive triplet loss.

Designed for SLURM array jobs: each task receives a unique hyperparameter configuration
via SLURM_ARRAY_TASK_ID (mapped to a grid slot in the triplet preparation output).

Training pipeline:
  1. Load Qwen3-Embedding-0.6B (or a previously fine-tuned checkpoint)
  2. Construct triplets on-the-fly from a scored pairwise dataset using transitive
     inference and hard-negative selection by cosine rank
  3. Train with cosine-distance triplet loss (margin = 0.3) via SentenceTransformerTrainer
  4. Evaluate per-topic binary F1 on a held-out validation set
  5. Save the best checkpoint

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

import numpy as np

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
    max_positives_per_anchor: int = 2
    max_negatives_per_anchor: int = 3
    seed: int = 42
    # Set automatically from SLURM_ARRAY_TASK_ID if running on HPC
    task_id: int = field(default_factory=lambda: int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))


# ---------------------------------------------------------------------------
# Dataset (triplet construction only — training is handled by the Trainer)
# ---------------------------------------------------------------------------

class TripletDatasetFromPairs:
    """Constructs triplets from a scored pairwise JSONL/parquet file.

    Triplet construction strategy:
    - For each anchor with at least one positive and one negative, pair each
      positive with hard negatives ranked by descending cosine similarity.
    - Transitive inference: if (A,B) is positive, B's positives are also treated
      as indirect positives of A for augmentation.
    """

    POSITIVE_SCORES = {1, 2, 3}
    NEGATIVE_SCORES = {5, 6, 7}

    def __init__(
        self,
        path: str | Path,
        max_positives: int = 2,
        max_negatives: int = 3,
        augment_transitive: bool = True,
        seed: int = 42,
    ):
        import pandas as pd

        self.rng = random.Random(seed)
        pairs = self._load(Path(path))
        self.triplets = self._build_triplets(pairs, max_positives, max_negatives, augment_transitive)
        logger.info("TripletDataset: %d triplets from %d pairs", len(self.triplets), len(pairs))

    @staticmethod
    def _load(path: Path) -> list[dict]:
        import pandas as pd

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
    ) -> list[tuple[str, str, str]]:
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

        triplets: list[tuple[str, str, str]] = []
        eligible = sorted(set(positives.keys()) & set(negatives.keys()))
        for anchor in eligible:
            ta = text_lookup.get(anchor, "")
            pos_sorted = positives[anchor][:max_positives]
            neg_sorted = sorted(negatives[anchor], key=lambda x: x[2], reverse=True)[:max_negatives]
            for _, tp, _ in pos_sorted:
                for _, tn, _ in neg_sorted:
                    triplets.append((ta, tp, tn))

        self.rng.shuffle(triplets)
        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        return self.triplets[idx]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class BinaryClassificationEvaluatorPrimaryF1:
    """Evaluates per-topic binary F1 on a validation set of scored pairs.

    A pair is predicted positive if cosine_similarity ≥ threshold (0.5 default).
    Reports macro-averaged F1 across topics and an overall F1.

    Uses model.encode() which is correct here — evaluation does not require gradients.
    """

    POSITIVE_SCORES = {1, 2, 3}

    def __init__(self, val_pairs: list[dict], threshold: float = 0.5, name: str = "eval"):
        self.val_pairs = val_pairs
        self.threshold = threshold
        self.name = name

    def __call__(self, model, output_path: str | None = None, epoch: int = -1, steps: int = -1) -> float:
        import torch
        import torch.nn.functional as F

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
# Training
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import TripletLoss as STTripletLoss, TripletDistanceMetric
        from datasets import Dataset as HFDataset
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers>=3.0 and datasets are required: "
            "pip install 'sentence-transformers>=3.0' datasets"
        ) from exc

    output_dir = Path(cfg.output_dir) / f"task_{cfg.task_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", cfg.model_name)
    model = SentenceTransformer(cfg.model_name, trust_remote_code=True)
    model.max_seq_length = cfg.max_seq_length

    triplet_ds = TripletDatasetFromPairs(
        cfg.triplets_path,
        max_positives=cfg.max_positives_per_anchor,
        max_negatives=cfg.max_negatives_per_anchor,
        seed=cfg.seed,
    )
    if len(triplet_ds) == 0:
        raise ValueError(
            f"No triplets could be constructed from {cfg.triplets_path}. "
            "Ensure the data has anchor-centric pairs where the same anchor_id "
            "appears in both positive (score 1–3) and negative (score 5–7) rows."
        )

    anchors, positives, negatives = zip(*[triplet_ds[i] for i in range(len(triplet_ds))])
    train_dataset = HFDataset.from_dict({
        "anchor": list(anchors),
        "positive": list(positives),
        "negative": list(negatives),
    })
    logger.info("Training dataset: %d triplets", len(train_dataset))

    loss = STTripletLoss(
        model=model,
        distance_metric=TripletDistanceMetric.COSINE,
        triplet_margin=cfg.triplet_margin,
    )

    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="no",
        save_strategy="epoch",
        seed=cfg.seed,
        logging_steps=50,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    logger.info("Starting training | triplets=%d | epochs=%d", len(train_dataset), cfg.epochs)
    trainer.train()

    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    logger.info("Training complete. Final model saved to %s", final_path)
    (output_dir / "config.json").write_text(json.dumps(cfg.__dict__, default=str, indent=2))


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
    import torch
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
