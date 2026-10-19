"""Gradient flow test — verifies parameters actually update during training.

This is the key regression test for the model.encode() bug: that call runs
internally under torch.no_grad() so loss.backward() is a no-op. The correct
fix (SentenceTransformerTrainer with sentence_transformers.losses.TripletLoss)
must show non-zero parameter updates after one training step.
"""

import json
import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


def _write_mock_pairs(path: Path, n_anchors: int = 10) -> None:
    from src.generate_mock_data import generate_pairs
    pairs = generate_pairs(n_anchors=n_anchors, seed=99)
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")


@pytest.mark.slow
class TestGradientFlow:
    """Marked slow because they load a real transformer model (~600MB)."""

    def test_parameters_update_after_one_epoch(self, tmp_path):
        """After one training epoch, model parameters must have changed."""
        from sentence_transformers import SentenceTransformer
        from src.train import TripletDatasetFromPairs, TrainConfig, train

        pairs_path = tmp_path / "pairs.jsonl"
        _write_mock_pairs(pairs_path, n_anchors=15)

        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
        initial_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        cfg = TrainConfig(
            triplets_path=str(pairs_path),
            output_dir=str(tmp_path / "run"),
            epochs=1,
            contrastive_batch_size=4,
            triplet_batch_size=4,
            seed=42,
        )
        train(cfg)

        updated_model = SentenceTransformer(
            str(tmp_path / "run" / "task_00" / "final"),
            trust_remote_code=True,
        )
        updated_params = dict(updated_model.named_parameters())

        changed = []
        for name, init_p in initial_params.items():
            if name in updated_params:
                if not torch.allclose(init_p, updated_params[name].detach(), atol=1e-7):
                    changed.append(name)

        assert len(changed) > 0, (
            "No parameters changed after training — gradient flow is broken. "
            "Ensure SentenceTransformerTrainer is used (not model.encode() in the loop)."
        )

    def test_triplets_are_nonzero_before_training(self, tmp_path):
        """Prerequisite: triplet construction must succeed before testing gradient flow."""
        from src.train import TripletDatasetFromPairs

        pairs_path = tmp_path / "pairs.jsonl"
        _write_mock_pairs(pairs_path, n_anchors=10)
        ds = TripletDatasetFromPairs(str(pairs_path), seed=42)
        assert len(ds) > 0, "Need non-zero triplets to test gradient flow"
