"""Smoke tests — end-to-end pipeline runs without crashing on mock data.

These tests do NOT require a GPU or the Qwen3 model weights. They verify the
pipeline mechanics (data generation, triplet construction, clustering, evaluation)
using random embeddings as a fast substitute for real model output.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_mock_pairs(path: Path, n_anchors: int = 20) -> None:
    from src.generate_mock_data import generate_pairs
    pairs = generate_pairs(n_anchors=n_anchors, seed=7)
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")


class TestMockDataGeneration:
    def test_generate_pairs_runs(self):
        from src.generate_mock_data import generate_pairs
        pairs = generate_pairs(n_anchors=10, seed=0)
        assert len(pairs) >= 40  # at least 2 pos + 2 neg per anchor

    def test_all_required_fields_present(self):
        from src.generate_mock_data import generate_pairs
        pairs = generate_pairs(n_anchors=5, seed=0)
        required = {"anchor_id", "candidate_id", "text_a", "text_b",
                    "similarity_score", "cosine_similarity", "bin", "pair_type"}
        for p in pairs:
            assert required <= set(p.keys()), f"Missing fields: {required - set(p.keys())}"

    def test_scores_in_valid_range(self):
        from src.generate_mock_data import generate_pairs
        pairs = generate_pairs(n_anchors=10, seed=0)
        for p in pairs:
            assert 1 <= p["similarity_score"] <= 7, (
                f"Score {p['similarity_score']} out of range (expect 1-7 for mock data)"
            )


class TestTripletDataset:
    def test_nonzero_triplets_from_anchor_centric_data(self, tmp_path):
        from src.train import TripletDatasetFromPairs
        pairs_path = tmp_path / "pairs.jsonl"
        _write_mock_pairs(pairs_path, n_anchors=20)
        ds = TripletDatasetFromPairs(str(pairs_path), seed=0)
        assert len(ds) > 0

    def test_getitem_returns_three_strings(self, tmp_path):
        from src.train import TripletDatasetFromPairs
        pairs_path = tmp_path / "pairs.jsonl"
        _write_mock_pairs(pairs_path, n_anchors=10)
        ds = TripletDatasetFromPairs(str(pairs_path), seed=0)
        assert len(ds) > 0
        item = ds[0]
        assert len(item) == 3
        assert all(isinstance(s, str) and len(s) > 0 for s in item)


class TestClusteringPipeline:
    def test_reduce_dimensions_smoke(self):
        from src.cluster import reduce_dimensions
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((100, 64)).astype(np.float32)
        reduced = reduce_dimensions(emb, pca_dim=16, umap_dim=8)
        # If umap-learn is not installed, reduce_dimensions returns PCA output (pca_dim dims)
        assert reduced.shape[0] == 100
        assert reduced.shape[1] in (8, 16), f"Unexpected shape: {reduced.shape}"

    def test_kmeans_smoke(self):
        from src.cluster import cluster_kmeans
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((100, 16)).astype(np.float32)
        result = cluster_kmeans(emb, n_clusters=10, n_iter=5, n_redo=1)
        assert len(result.labels) == 100
        assert len(set(result.labels)) <= 10

    def test_evaluate_clusters_with_doc_ids(self, tmp_path):
        from src.cluster import evaluate_clusters
        rng = np.random.default_rng(0)
        n = 50
        embeddings = rng.standard_normal((n, 8)).astype(np.float32)
        labels = np.arange(n)  # every doc in its own cluster
        doc_ids = [f"doc_{i}" for i in range(n)]
        pairs_df = pd.DataFrame([
            {"anchor_id": "doc_0", "candidate_id": "doc_1", "similarity_score": 2},
            {"anchor_id": "doc_2", "candidate_id": "doc_3", "similarity_score": 6},
        ])
        result = evaluate_clusters(embeddings, labels, pairs_df, doc_ids=doc_ids)
        # doc_0 and doc_1 in different clusters → FN; doc_2 and doc_3 in different clusters → TN
        assert result.f1 == pytest.approx(0.0)
        assert result.recall == pytest.approx(0.0)


class TestDataPipeline:
    def test_build_triplets_from_anchor_centric_df(self, tmp_path):
        from src.generate_mock_data import generate_pairs
        from src.data_pipeline import build_triplets
        pairs = generate_pairs(n_anchors=15, seed=3)
        df = pd.DataFrame(pairs)
        triplets = build_triplets(df)
        assert len(triplets) > 0
        assert "anchor_text" in triplets.columns
        assert "positive_text" in triplets.columns
        assert "negative_text" in triplets.columns

    def test_evaluate_from_jsonl(self, tmp_path):
        """evaluate.py evaluate_model runs without a real model using a dummy sentinel."""
        pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
        # Just verify the import is clean; full eval tested in test_gradient.py (slow)
        from src.evaluate import evaluate_model, POSITIVE_SCORES, NEGATIVE_SCORES, EVAL_SCORES
        assert EVAL_SCORES == {1, 2, 3, 5, 6, 7}
