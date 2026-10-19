"""Tests for triplet construction — verifies the zero-triplets bug is fixed."""

import json
import tempfile
from pathlib import Path

import pytest

from src.train import TripletDatasetFromPairs
from src.generate_mock_data import generate_pairs


def _write_jsonl(pairs: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")


class TestTripletConstruction:
    def test_anchor_centric_mock_data_produces_nonzero_triplets(self, tmp_path):
        """Anchor-centric pairs (each anchor has pos + neg) must yield triplets."""
        pairs = generate_pairs(n_anchors=10, seed=0)
        pairs_path = tmp_path / "pairs.jsonl"
        _write_jsonl(pairs, pairs_path)
        ds = TripletDatasetFromPairs(str(pairs_path), seed=0)
        assert len(ds) > 0, (
            "Expected non-zero triplets from anchor-centric mock data. "
            "Check that generate_pairs() produces anchors with both positive and negative rows."
        )

    def test_unique_pair_data_produces_zero_triplets(self, tmp_path):
        """Unique pairs with no shared anchors cannot form triplets."""
        pairs = [
            {"anchor_id": "a-0", "candidate_id": "b-0", "text_a": "x", "text_b": "y",
             "similarity_score": 2, "cosine_similarity": 0.95},
            {"anchor_id": "a-1", "candidate_id": "b-1", "text_a": "p", "text_b": "q",
             "similarity_score": 6, "cosine_similarity": 0.45},
        ]
        pairs_path = tmp_path / "unique_pairs.jsonl"
        _write_jsonl(pairs, pairs_path)
        ds = TripletDatasetFromPairs(str(pairs_path), seed=0)
        assert len(ds) == 0, "Unique pairs with no shared anchors cannot form triplets"

    def test_triplet_texts_are_strings(self, tmp_path):
        pairs = generate_pairs(n_anchors=5, seed=1)
        pairs_path = tmp_path / "pairs.jsonl"
        _write_jsonl(pairs, pairs_path)
        ds = TripletDatasetFromPairs(str(pairs_path), seed=0)
        assert len(ds) > 0
        anchor, positive, negative = ds[0]
        assert isinstance(anchor, str) and len(anchor) > 0
        assert isinstance(positive, str) and len(positive) > 0
        assert isinstance(negative, str) and len(negative) > 0
        assert anchor != negative, "Anchor and negative should not be identical texts"

    def test_all_anchors_have_both_pos_and_neg(self):
        pairs = generate_pairs(n_anchors=20, seed=42)
        anchor_ids_pos = {p["anchor_id"] for p in pairs if p["similarity_score"] <= 3}
        anchor_ids_neg = {p["anchor_id"] for p in pairs if p["similarity_score"] >= 5}
        shared = anchor_ids_pos & anchor_ids_neg
        assert len(shared) == 20, (
            f"Expected all 20 anchors to have pos+neg pairs; got {len(shared)}"
        )

    def test_transitive_augmentation_increases_triplets(self, tmp_path):
        """Transitive inference should produce at least as many triplets as without it."""
        pairs = generate_pairs(n_anchors=15, seed=5)
        pairs_path = tmp_path / "pairs.jsonl"
        _write_jsonl(pairs, pairs_path)
        ds_no_transitive = TripletDatasetFromPairs(str(pairs_path), augment_transitive=False, seed=0)
        ds_with_transitive = TripletDatasetFromPairs(str(pairs_path), augment_transitive=True, seed=0)
        assert len(ds_with_transitive) >= len(ds_no_transitive)
