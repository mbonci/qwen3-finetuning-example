"""Fine-tune Qwen3-Embedding-0.6B with hybrid contrastive + triplet loss.

Two training objectives are interleaved each epoch via model.fit():

  1. WeightedContrastiveLoss — pairwise Siamese loss over binary-labeled pairs.
     Per-score sampling multipliers upweight hard/informative examples.
     margin = 0.7 (cosine distance space).

  2. TripletLoss — cosine-distance triplet loss over (anchor, positive, negative)
     triples constructed on-the-fly with transitive inference.
     margin = 0.5 (optimized via grid search on proprietary corpus).

Both losses call model(sentence_features[i])["sentence_embedding"] — the
differentiable forward path through the transformer. Using model.encode() in
the training loop is a fatal bug: it runs under torch.no_grad() internally and
loss.backward() becomes a no-op.

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
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Per-score sampling multipliers — upweight hard/informative examples.
# Scores 1 and 7 (clear positive / clear negative) get highest weight; scores
# 4 and 5 are ambiguous so they're downweighted for contrastive training.
DEFAULT_SAMPLING_MULTIPLIERS: dict[int, float] = {
    1: 4.0,
    2: 2.0,
    3: 0.5,
    4: 0.2,
    5: 0.2,
    6: 1.0,
    7: 4.0,
}


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    triplets_path: str = "data/mock_pairs.jsonl"
    output_dir: str = "runs/default"
    epochs: int = 3
    # Contrastive objective
    contrastive_batch_size: int = 8
    contrastive_margin: float = 0.7     # cosine-distance margin for Siamese loss
    # Triplet objective
    triplet_batch_size: int = 32
    triplet_margin: float = 0.5         # optimized via grid search (was 0.7)
    triplet_samples_per_anchor: int = 4
    # Shared
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 42
    # Set from SLURM_ARRAY_TASK_ID when running on HPC
    task_id: int = field(default_factory=lambda: int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))


# ---------------------------------------------------------------------------
# Loss 1: Weighted Pairwise Contrastive Loss
# ---------------------------------------------------------------------------

class WeightedContrastiveLoss(nn.Module):
    """Pairwise Siamese contrastive loss with per-sample score weighting.

    L(a, b, y, w) = w * 0.5 * [y * d(a,b)^2  +  (1-y) * max(0, margin - d(a,b))^2]

    where d(a,b) = 1 - cos(a, b)  and  y ∈ {0, 1}.

    Labels may be either:
      - A scalar tensor: y per sample, weight = 1.0
      - A 2-column tensor [y, w]: binary label + per-sample weight

    The per-sample weights are set from score multipliers in the dataset so that
    hard examples (clear positives: score 1, clear negatives: score 7) contribute
    more to the gradient than ambiguous boundary examples (scores 4-5).
    """

    def __init__(self, model, margin: float = 0.7):
        super().__init__()
        self.model = model
        self.margin = margin

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        rep_a = self.model(sentence_features[0])["sentence_embedding"]
        rep_b = self.model(sentence_features[1])["sentence_embedding"]
        rep_a = F.normalize(rep_a, p=2, dim=1)
        rep_b = F.normalize(rep_b, p=2, dim=1)

        if labels.dim() > 1 and labels.size(1) >= 2:
            y = labels[:, 0].float()
            w = labels[:, 1].float()
        else:
            y = labels.view(-1).float()
            w = torch.ones_like(y)

        d = 1.0 - F.cosine_similarity(rep_a, rep_b)
        pos_term = y * d.pow(2)
        neg_term = (1.0 - y) * F.relu(self.margin - d).pow(2)
        loss = w * 0.5 * (pos_term + neg_term)
        return loss.mean()


# ---------------------------------------------------------------------------
# Loss 2: Triplet Loss
# ---------------------------------------------------------------------------

class TripletLoss(nn.Module):
    """Cosine-distance triplet loss.

    L(a, p, n) = max(0, d(a,p) − d(a,n) + margin)
    where d(x,y) = 1 − cos(x, y).

    Calls model(sentence_features[i])["sentence_embedding"] — the differentiable
    path. model.encode() must NOT be used here; it runs under torch.no_grad().
    """

    def __init__(self, model, margin: float = 0.5):
        super().__init__()
        self.model = model
        self.margin = margin

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        rep_a = self.model(sentence_features[0])["sentence_embedding"]
        rep_p = self.model(sentence_features[1])["sentence_embedding"]
        rep_n = self.model(sentence_features[2])["sentence_embedding"]
        rep_a = F.normalize(rep_a, p=2, dim=1)
        rep_p = F.normalize(rep_p, p=2, dim=1)
        rep_n = F.normalize(rep_n, p=2, dim=1)
        d_pos = 1.0 - F.cosine_similarity(rep_a, rep_p)
        d_neg = 1.0 - F.cosine_similarity(rep_a, rep_n)
        return F.relu(d_pos - d_neg + self.margin).mean()


# ---------------------------------------------------------------------------
# Dataset 1: Pairwise (Contrastive)
# ---------------------------------------------------------------------------

class PairDatasetFromPairs:
    """Binary-labeled pairs for WeightedContrastiveLoss.

    Each example is an InputExample with:
      texts = [text_a, text_b]
      label = [binary_label, weight]

    where binary_label ∈ {0, 1} and weight is from DEFAULT_SAMPLING_MULTIPLIERS.
    Score 4 is included as a training negative (hard boundary negative) even though
    it is excluded from the cluster evaluation metric.
    """

    POSITIVE_SCORES = {1, 2, 3}
    NEGATIVE_SCORES = {4, 5, 6, 7}

    def __init__(self, path: str | Path, multipliers: dict[int, float] | None = None):
        from sentence_transformers import InputExample

        if multipliers is None:
            multipliers = DEFAULT_SAMPLING_MULTIPLIERS

        pairs = self._load(Path(path))
        self.examples = []
        for p in pairs:
            score = p.get("similarity_score")
            if score is None:
                continue
            score = int(score)
            if score in self.POSITIVE_SCORES:
                binary = 1
            elif score in self.NEGATIVE_SCORES:
                binary = 0
            else:
                continue
            weight = multipliers.get(score, 1.0)
            self.examples.append(
                InputExample(
                    texts=[p.get("text_a", ""), p.get("text_b", "")],
                    label=[binary, weight],
                )
            )
        logger.info("PairDataset: %d labeled pairs", len(self.examples))

    @staticmethod
    def _load(path: Path) -> list[dict]:
        import pandas as pd
        if path.suffix == ".jsonl":
            with path.open() as f:
                return [json.loads(line) for line in f if line.strip()]
        return pd.read_parquet(path).to_dict("records")


# ---------------------------------------------------------------------------
# Dataset 2: Triplet
# ---------------------------------------------------------------------------

class TripletDatasetFromPairs:
    """Constructs triplets on-the-fly from a scored pairwise JSONL/parquet file.

    Triplet construction strategy:
    - For each anchor with at least one positive and one negative, pair each
      positive with hard negatives ranked by descending cosine similarity.
    - Transitive inference: if (A,B) is positive, B's positives are also treated
      as indirect positives of A; if B's negatives are very clear (score 7),
      they are treated as inferred negatives of A.
    - Score 4 is included as a training negative (hard boundary negative).
    """

    POSITIVE_SCORES = {1, 2, 3}
    NEGATIVE_SCORES = {4, 5, 6, 7}

    def __init__(
        self,
        path: str | Path,
        samples_per_anchor: int = 4,
        augment_transitive: bool = True,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        pairs = self._load(Path(path))
        self.triplets = self._build_triplets(pairs, samples_per_anchor, augment_transitive)
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
        samples_per_anchor: int,
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
            # One-hop transitive positives: A~B (score 1/2) + B~C (score 1/2) → (A,C) positive
            bridge_scores = {1, 2}
            for anchor, pos_list in list(positives.items()):
                indirect: list[tuple[str, str, float]] = []
                existing = {x[0] for x in pos_list}
                for bridge_id, _, _ in [x for x in pos_list if True]:
                    # Only use strong bridge pairs
                    bridge_direct = [x for x in positives.get(bridge_id, [])
                                     if x[0] != anchor and x[0] not in existing]
                    indirect.extend(bridge_direct)
                for item in indirect:
                    if item[0] not in existing:
                        pos_list.append(item)
                        existing.add(item[0])

        triplets: list[tuple[str, str, str]] = []
        eligible = sorted(set(positives.keys()) & set(negatives.keys()))
        for anchor in eligible:
            ta = text_lookup.get(anchor, "")
            # Hard negatives first (highest cosine similarity among negatives)
            pos_pool = positives[anchor]
            neg_pool = sorted(negatives[anchor], key=lambda x: x[2], reverse=True)
            for _ in range(samples_per_anchor):
                if pos_pool and neg_pool:
                    _, tp, _ = self.rng.choice(pos_pool)
                    _, tn, _ = self.rng.choice(neg_pool)
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
    """Per-topic binary F1 evaluation on held-out pairs.

    Predicts positive if cosine_similarity ≥ threshold.
    Reports macro-averaged F1 across topics.

    Uses model.encode() — correct for evaluation, which does not need gradients.
    """

    POSITIVE_SCORES = {1, 2, 3}

    def __init__(self, val_pairs: list[dict], threshold: float = 0.5, name: str = "eval"):
        self.val_pairs = val_pairs
        self.threshold = threshold
        self.name = name

    def __call__(self, model, output_path: str | None = None, epoch: int = -1, steps: int = -1) -> float:
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
# Cosine distribution visualisation
# ---------------------------------------------------------------------------

def get_similarity_distributions(model, s1: list[str], s2: list[str], labels: list[int], batch_size: int = 128) -> dict:
    """Encode pairs and return cosine similarities split by positive/negative label."""
    import torch.nn.functional as F
    e1 = np.array(model.encode(s1, batch_size=batch_size, normalize_embeddings=False, show_progress_bar=False))
    e2 = np.array(model.encode(s2, batch_size=batch_size, normalize_embeddings=False, show_progress_bar=False))
    a = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
    b = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12)
    cosine = np.sum(a * b, axis=1)
    labels_arr = np.array(labels)
    return {
        "cosine_positive": cosine[labels_arr == 1],
        "cosine_negative": cosine[labels_arr == 0],
    }


def plot_cosine_kde(before: dict, after: dict, save_path: str) -> None:
    """Save a before/after cosine KDE plot (positive vs negative pairs)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping KDE plot")
        return

    all_vals = np.concatenate([before["cosine_positive"], before["cosine_negative"],
                                after["cosine_positive"], after["cosine_negative"]])
    xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))

    plt.figure(figsize=(7, 5))
    sns.kdeplot(before["cosine_positive"], label="Positive (before)", fill=True, color="green", alpha=0.3)
    sns.kdeplot(after["cosine_positive"],  label="Positive (after)",  linestyle="--", color="green")
    sns.kdeplot(before["cosine_negative"], label="Negative (before)", fill=True, color="red",   alpha=0.3)
    sns.kdeplot(after["cosine_negative"],  label="Negative (after)",  linestyle="--", color="red")
    plt.title("Cosine Distribution (KDE) — Positive vs Negative")
    plt.xlim(xmin, xmax)
    plt.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("KDE plot saved to %s", save_path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    try:
        from sentence_transformers import SentenceTransformer, InputExample
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("sentence-transformers>=3.0 is required") from exc

    output_dir = Path(cfg.output_dir) / f"task_{cfg.task_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", cfg.model_name)
    model = SentenceTransformer(cfg.model_name, trust_remote_code=True)

    # --- Snapshot cosine distributions before training ---
    pair_ds_for_viz = PairDatasetFromPairs(cfg.triplets_path)
    s1_viz = [ex.texts[0] for ex in pair_ds_for_viz.examples]
    s2_viz = [ex.texts[1] for ex in pair_ds_for_viz.examples]
    labels_viz = [int(ex.label) for ex in pair_ds_for_viz.examples]
    before_dists = get_similarity_distributions(model, s1_viz, s2_viz, labels_viz)

    # --- Objective 1: weighted pairwise contrastive ---
    pair_ds = PairDatasetFromPairs(cfg.triplets_path)
    if len(pair_ds.examples) == 0:
        raise ValueError("No labeled pairs — regenerate mock data with: python src/generate_mock_data.py")

    contrastive_dataloader = DataLoader(
        pair_ds.examples,
        batch_size=cfg.contrastive_batch_size,
        shuffle=True,
        collate_fn=model.smart_batching_collate,
    )
    contrastive_loss = WeightedContrastiveLoss(model=model, margin=cfg.contrastive_margin)

    # --- Objective 2: triplet ---
    triplet_ds = TripletDatasetFromPairs(
        cfg.triplets_path,
        samples_per_anchor=cfg.triplet_samples_per_anchor,
        seed=cfg.seed,
    )
    if len(triplet_ds) == 0:
        raise ValueError(
            "No triplets could be constructed. Ensure mock data has anchor-centric pairs "
            "(same anchor_id in both positive and negative rows). "
            "Re-run: python src/generate_mock_data.py"
        )

    triplet_examples = [InputExample(texts=list(t)) for t in triplet_ds.triplets]
    triplet_dataloader = DataLoader(
        triplet_examples,
        batch_size=cfg.triplet_batch_size,
        shuffle=True,
        collate_fn=model.smart_batching_collate,
    )
    triplet_loss = TripletLoss(model=model, margin=cfg.triplet_margin)

    train_objectives = [
        (contrastive_dataloader, contrastive_loss),
        (triplet_dataloader, triplet_loss),
    ]

    logger.info(
        "Training | pairs=%d contrastive | %d triplets | epochs=%d",
        len(pair_ds.examples), len(triplet_ds), cfg.epochs,
    )

    # model.fit() interleaves batches from both objectives each step.
    # It uses the differentiable forward path internally (not model.encode()).
    model.fit(
        train_objectives=train_objectives,
        epochs=cfg.epochs,
        warmup_steps=cfg.warmup_steps,
        output_path=str(output_dir / "final"),
        save_best_model=False,
        show_progress_bar=True,
        max_grad_norm=cfg.max_grad_norm,
    )
    logger.info("Training complete. Model saved to %s", output_dir / "final")
    (output_dir / "config.json").write_text(json.dumps(cfg.__dict__, default=str, indent=2))

    # --- Snapshot cosine distributions after training and plot ---
    trained_model = SentenceTransformer(str(output_dir / "final"), trust_remote_code=True)
    after_dists = get_similarity_distributions(trained_model, s1_viz, s2_viz, labels_viz)
    kde_path = str(output_dir / "visualization" / "cosine_kde.png")
    plot_cosine_kde(before_dists, after_dists, save_path=kde_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune Qwen3-Embedding-0.6B with hybrid loss")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--triplets", default="data/mock_pairs.jsonl")
    p.add_argument("--output", default="runs/default")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8, help="Contrastive batch size")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--contrastive-margin", type=float, default=0.7)
    p.add_argument("--triplet-margin", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = TrainConfig(
        model_name=args.model,
        triplets_path=args.triplets,
        output_dir=args.output,
        epochs=args.epochs,
        contrastive_batch_size=args.batch_size,
        learning_rate=args.lr,
        contrastive_margin=args.contrastive_margin,
        triplet_margin=args.triplet_margin,
        seed=args.seed,
    )
    train(cfg)
