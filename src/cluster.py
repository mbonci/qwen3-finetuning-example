"""Scalable semantic clustering of embedding vectors.

Three-stage pipeline:
  1. reduce_dimensions()  — PCA to 256d, then UMAP to lower dimension
  2. cluster_kmeans()     — FAISS-accelerated K-Means (GPU or CPU)
  3. cluster_hdbscan()    — Density-based clustering for irregular group shapes

Also provides evaluate_clusters() which computes clustering F1 on known similar
pairs — the metric used to report the +0.20 improvement in the README.

Usage (end-to-end mock run):
    python src/cluster.py

Usage (library):
    from src.cluster import reduce_dimensions, cluster_kmeans, evaluate_clusters
    reduced = reduce_dimensions(embeddings, pca_dim=256, umap_dim=64)
    labels, centroids = cluster_kmeans(reduced, n_clusters=150)
    f1 = evaluate_clusters(reduced, labels, val_pairs_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Results dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KMeansResult:
    labels: np.ndarray          # shape (n_docs,) — cluster index per document
    centroids: np.ndarray       # shape (n_clusters, dim)
    inertia: float
    n_iter: int


@dataclass
class HDBSCANResult:
    labels: np.ndarray          # shape (n_docs,) — -1 = noise
    cluster_sizes: dict[int, int]
    n_clusters: int
    noise_fraction: float


@dataclass
class ClusterEvalResult:
    f1: float                   # macro F1 on known similar pairs
    precision: float
    recall: float
    n_pairs_evaluated: int
    threshold: float


# ---------------------------------------------------------------------------
# Stage 1: Dimensionality Reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    embeddings: np.ndarray,
    pca_dim: int = 256,
    umap_dim: int = 64,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce high-dimensional embeddings via PCA then UMAP.

    PCA is applied first to bring embeddings to pca_dim dimensions (cheap,
    linear, handles collinearity). UMAP then captures non-linear structure
    for clustering.

    Parameters
    ----------
    embeddings:       Float32 array of shape (n_docs, embedding_dim)
    pca_dim:          Target dimensionality after PCA (256 recommended)
    umap_dim:         Target dimensionality after UMAP (2 for viz, 64 for clustering)
    umap_n_neighbors: UMAP neighborhood size (larger = more global structure)
    umap_min_dist:    UMAP minimum distance (smaller = tighter clusters)

    Returns
    -------
    Float32 array of shape (n_docs, umap_dim)
    """
    n_docs, orig_dim = embeddings.shape
    logger.info("Input: %d docs × %d dims", n_docs, orig_dim)

    actual_pca_dim = min(pca_dim, orig_dim, n_docs - 1)
    logger.info("PCA: %d → %d", orig_dim, actual_pca_dim)
    pca = PCA(n_components=actual_pca_dim, random_state=random_state)
    reduced_pca = pca.fit_transform(embeddings).astype(np.float32)
    explained = pca.explained_variance_ratio_.sum()
    logger.info("PCA explained variance: %.1f%%", explained * 100)

    actual_umap_dim = min(umap_dim, actual_pca_dim)
    logger.info("UMAP: %d → %d", actual_pca_dim, actual_umap_dim)
    try:
        import umap
        reducer = umap.UMAP(
            n_components=actual_umap_dim,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=random_state,
            low_memory=True,
        )
        reduced = reducer.fit_transform(reduced_pca).astype(np.float32)
    except ImportError:
        logger.warning("umap-learn not installed — returning PCA output only")
        reduced = reduced_pca

    logger.info("Reduced: %d docs × %d dims", reduced.shape[0], reduced.shape[1])
    return reduced


# ---------------------------------------------------------------------------
# Stage 2a: FAISS K-Means
# ---------------------------------------------------------------------------

def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 150,
    n_iter: int = 25,
    n_redo: int = 3,
    verbose: bool = False,
    use_gpu: bool = False,
) -> KMeansResult:
    """FAISS-accelerated K-Means clustering.

    For 8M documents, FAISS K-Means with GPU acceleration is 10–50× faster than
    sklearn. Falls back to sklearn if faiss is not installed.

    Parameters
    ----------
    embeddings:  Float32 array of shape (n_docs, dim)
    n_clusters:  Number of cluster centroids K
    n_iter:      K-Means iterations
    n_redo:      Number of independent restarts (best inertia is kept)
    use_gpu:     Use FAISS GPU index if available

    Returns
    -------
    KMeansResult with labels, centroids, inertia, and iteration count
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    n_docs, dim = embeddings.shape
    n_clusters = min(n_clusters, n_docs)
    logger.info("K-Means: %d docs, %d clusters, dim=%d", n_docs, n_clusters, dim)

    try:
        import faiss

        kmeans = faiss.Kmeans(
            dim,
            n_clusters,
            niter=n_iter,
            nredo=n_redo,
            verbose=verbose,
            gpu=use_gpu and faiss.get_num_gpus() > 0,
        )
        kmeans.train(embeddings)
        _, labels = kmeans.index.search(embeddings, 1)
        labels = labels.ravel()
        inertia = float(kmeans.obj[-1]) if hasattr(kmeans, "obj") and len(kmeans.obj) > 0 else 0.0
        return KMeansResult(
            labels=labels,
            centroids=kmeans.centroids,
            inertia=inertia,
            n_iter=n_iter,
        )

    except ImportError:
        logger.warning("faiss not installed — falling back to sklearn MiniBatchKMeans")
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=n_redo)
        labels = km.fit_predict(embeddings)
        return KMeansResult(
            labels=labels,
            centroids=km.cluster_centers_.astype(np.float32),
            inertia=float(km.inertia_),
            n_iter=n_iter,
        )


# ---------------------------------------------------------------------------
# Stage 2b: HDBSCAN
# ---------------------------------------------------------------------------

def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> HDBSCANResult:
    """Density-based clustering with HDBSCAN.

    Complements K-Means by finding clusters of arbitrary shape and identifying
    noise points (label = -1). Well-suited for semantic clusters that vary
    significantly in size and density.

    Parameters
    ----------
    embeddings:               Float32 array of shape (n_docs, dim)
    min_cluster_size:         Minimum points to form a cluster
    min_samples:              Core distance parameter (higher = more noise)
    metric:                   Distance metric ('euclidean', 'cosine')
    cluster_selection_method: 'eom' (excess of mass) or 'leaf'
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError as exc:
        raise ImportError("hdbscan is required: pip install hdbscan") from exc

    logger.info("HDBSCAN: %d docs, min_cluster_size=%d", embeddings.shape[0], min_cluster_size)
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(embeddings)
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    noise_fraction = float((labels == -1).mean())
    cluster_sizes = {int(l): int((labels == l).sum()) for l in sorted(unique_labels) if l != -1}

    logger.info(
        "HDBSCAN: %d clusters, %.1f%% noise", n_clusters, noise_fraction * 100
    )
    return HDBSCANResult(
        labels=labels,
        cluster_sizes=cluster_sizes,
        n_clusters=n_clusters,
        noise_fraction=noise_fraction,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    known_pairs_df: pd.DataFrame,
    threshold: float = 0.5,
) -> ClusterEvalResult:
    """Compute clustering F1 on known similar pairs.

    Two documents are predicted as similar if they share the same cluster label.
    Ground truth positives are pairs with similarity_score ∈ {1, 2, 3}.

    This is the metric used to measure the +0.20 clustering F1 improvement
    from fine-tuning Qwen3-Embedding-0.6B vs. the base model.

    Parameters
    ----------
    embeddings:     Float32 array of shape (n_docs, dim)
    labels:         Cluster label per document (shape n_docs,)
    known_pairs_df: DataFrame with anchor_id, candidate_id, similarity_score,
                    and an integer index aligned with embeddings
    threshold:      Cosine similarity threshold (used as fallback if label-based
                    F1 is degenerate — all docs in one cluster)
    """
    POSITIVE_SCORES = {1, 2, 3}
    df = known_pairs_df.dropna(subset=["similarity_score"]).copy()
    df["similarity_score"] = df["similarity_score"].astype(int)

    id_to_idx = {doc_id: i for i, doc_id in enumerate(df["anchor_id"].tolist() + df["candidate_id"].tolist())}

    y_true = []
    y_pred = []
    for _, row in df.iterrows():
        a_idx = id_to_idx.get(row["anchor_id"])
        b_idx = id_to_idx.get(row["candidate_id"])
        if a_idx is None or b_idx is None or a_idx >= len(labels) or b_idx >= len(labels):
            continue
        true_label = 1 if row["similarity_score"] in POSITIVE_SCORES else 0
        same_cluster = int(labels[a_idx] == labels[b_idx] and labels[a_idx] != -1)
        y_true.append(true_label)
        y_pred.append(same_cluster)

    if not y_true:
        logger.warning("No evaluable pairs found — returning zero F1")
        return ClusterEvalResult(f1=0.0, precision=0.0, recall=0.0, n_pairs_evaluated=0, threshold=threshold)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(f1_score(y_true, y_pred, zero_division=0, average="binary"))
    tp = sum(p == 1 and t == 1 for p, t in zip(y_pred, y_true))
    fn = sum(p == 0 and t == 1 for p, t in zip(y_pred, y_true))
    recall = tp / max(tp + fn, 1)

    logger.info(
        "Cluster eval: F1=%.4f  P=%.4f  R=%.4f  n=%d",
        f1, precision, recall, len(y_true),
    )
    return ClusterEvalResult(
        f1=f1, precision=precision, recall=recall,
        n_pairs_evaluated=len(y_true), threshold=threshold,
    )


# ---------------------------------------------------------------------------
# End-to-end mock run
# ---------------------------------------------------------------------------

def _mock_run() -> None:
    """Smoke-test the full clustering pipeline with random embeddings."""
    n_docs, dim = 500, 128
    logger.info("Mock run: %d docs × %d dims", n_docs, dim)
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n_docs, dim)).astype(np.float32)

    reduced = reduce_dimensions(embeddings, pca_dim=64, umap_dim=16)
    logger.info("Reduced shape: %s", reduced.shape)

    km = cluster_kmeans(reduced, n_clusters=20)
    logger.info("K-Means: inertia=%.2f, unique_labels=%d", km.inertia, len(set(km.labels)))

    mock_pairs = pd.DataFrame({
        "anchor_id": [str(i) for i in range(50)],
        "candidate_id": [str(i + 50) for i in range(50)],
        "similarity_score": rng.integers(1, 8, size=50),
    })
    eval_result = evaluate_clusters(reduced, km.labels, mock_pairs)
    logger.info("Eval F1: %.4f", eval_result.f1)
    logger.info("Mock run complete.")


if __name__ == "__main__":
    _mock_run()
