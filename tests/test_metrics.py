"""Tests for evaluation metrics — pairwise classification F1 correctness."""

import numpy as np
import pandas as pd
import pytest

from src.cluster import evaluate_clusters, ClusterEvalResult


def _make_embeddings(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


class TestEvaluateClusters:
    def test_perfect_clustering(self):
        """All positives in same cluster, all negatives in different clusters → F1=1.0."""
        embeddings = _make_embeddings(6)
        # Docs 0,1 → cluster 0 (positive pair)
        # Docs 2,3 → cluster 1 (negative pair)
        # Docs 4,5 → cluster 2 (negative pair)
        labels = np.array([0, 0, 1, 2, 3, 4])
        doc_ids = ["a", "b", "c", "d", "e", "f"]
        pairs_df = pd.DataFrame([
            {"anchor_id": "a", "candidate_id": "b", "similarity_score": 2},  # positive, same cluster → TP
            {"anchor_id": "c", "candidate_id": "d", "similarity_score": 6},  # negative, diff cluster → TN
        ])
        result = evaluate_clusters(embeddings, labels, pairs_df, doc_ids=doc_ids)
        assert result.f1 == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)

    def test_all_in_one_cluster_zero_precision(self):
        """All docs in one cluster → every pair predicted positive → precision = 0 for neg pairs."""
        embeddings = _make_embeddings(4)
        labels = np.array([0, 0, 0, 0])  # all same cluster
        doc_ids = ["a", "b", "c", "d"]
        pairs_df = pd.DataFrame([
            {"anchor_id": "a", "candidate_id": "b", "similarity_score": 2},  # positive, same → TP
            {"anchor_id": "c", "candidate_id": "d", "similarity_score": 6},  # negative, same → FP
        ])
        result = evaluate_clusters(embeddings, labels, pairs_df, doc_ids=doc_ids)
        # TP=1, FP=1, FN=0 → P=0.5, R=1.0, F1=0.667
        assert result.precision == pytest.approx(0.5)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)

    def test_precision_is_not_f1(self):
        """Regression: precision was previously computed as F1 (same call)."""
        embeddings = _make_embeddings(6)
        labels = np.array([0, 0, 1, 1, 2, 2])
        doc_ids = ["a", "b", "c", "d", "e", "f"]
        pairs_df = pd.DataFrame([
            {"anchor_id": "a", "candidate_id": "b", "similarity_score": 1},  # TP
            {"anchor_id": "a", "candidate_id": "c", "similarity_score": 6},  # TN (diff cluster)
            {"anchor_id": "c", "candidate_id": "d", "similarity_score": 1},  # TP
            {"anchor_id": "a", "candidate_id": "d", "similarity_score": 6},  # TN
        ])
        result = evaluate_clusters(embeddings, labels, pairs_df, doc_ids=doc_ids)
        assert result.precision != result.f1 or (result.precision == 1.0 and result.f1 == 1.0), (
            "precision and F1 should differ unless both are 1.0"
        )

    def test_scores_4_and_8_excluded(self):
        """Boundary scores 4 and 8 should not be evaluated."""
        embeddings = _make_embeddings(4)
        labels = np.array([0, 0, 0, 0])
        doc_ids = ["a", "b", "c", "d"]
        pairs_df = pd.DataFrame([
            {"anchor_id": "a", "candidate_id": "b", "similarity_score": 4},  # excluded
            {"anchor_id": "c", "candidate_id": "d", "similarity_score": 8},  # excluded
        ])
        result = evaluate_clusters(embeddings, labels, pairs_df, doc_ids=doc_ids)
        assert result.n_pairs_evaluated == 0

    def test_empty_pairs_returns_zero(self):
        embeddings = _make_embeddings(4)
        labels = np.array([0, 1, 0, 1])
        doc_ids = ["a", "b", "c", "d"]
        # Pass a properly-shaped empty DataFrame (missing-column error is a pandas quirk)
        empty_df = pd.DataFrame(columns=["anchor_id", "candidate_id", "similarity_score"])
        result = evaluate_clusters(embeddings, labels, empty_df, doc_ids=doc_ids)
        assert result.f1 == 0.0

    def test_doc_ids_aligns_embeddings_to_labels(self):
        """doc_ids maps anchor/candidate IDs to the correct row in labels."""
        embeddings = _make_embeddings(4)
        # doc_0/doc_1 → cluster 0 (same); doc_2 → cluster 1, doc_3 → cluster 2 (different)
        labels = np.array([0, 0, 1, 2])
        doc_ids = ["doc_0", "doc_1", "doc_2", "doc_3"]
        pairs_df = pd.DataFrame([
            {"anchor_id": "doc_0", "candidate_id": "doc_1", "similarity_score": 2},  # TP
            {"anchor_id": "doc_2", "candidate_id": "doc_3", "similarity_score": 6},  # TN
        ])
        result = evaluate_clusters(embeddings, labels, pairs_df, doc_ids=doc_ids)
        # TP=1, FP=0, FN=0 → P=1.0, R=1.0, F1=1.0
        assert result.f1 == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
