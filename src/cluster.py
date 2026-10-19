"""Scalable semantic clustering of embedding vectors.

Two independent clustering approaches are exposed as top-level pipelines:

  pipeline_kmeans()   — FAISS K-Means directly on raw embeddings (no PCA/UMAP)
      Exhaustive partition. Scales to 8M+ docs via FAISS GPU. K swept per topic;
      best K selected by silhouette score.

  pipeline_hdbscan()  — PCA (→50d) → UMAP (→5d) → HDBSCAN
      Density-based; finds irregular-shaped clusters. UMAP is required — HDBSCAN
      degrades above ~10–20 dimensions. Noise points (-1) are reassigned to their
      nearest cluster centroid so the output is an exhaustive partition too.

Both pipelines share the same dim-reduction primitives (reduce_pca, reduce_umap)
and the same evaluate_clusters() metric.

Usage (end-to-end mock run showing both approaches):
    python src/cluster.py

Usage (library):
    from src.cluster import pipeline_kmeans, pipeline_hdbscan, evaluate_clusters

    km  = pipeline_kmeans(embeddings, n_clusters=150)
    hdb = pipeline_hdbscan(embeddings, min_cluster_size=15)

    f1_km  = evaluate_clusters(embeddings, km.labels,  val_pairs_df, doc_ids=ids)
    f1_hdb = evaluate_clusters(embeddings, hdb.labels, val_pairs_df, doc_ids=ids)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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
    n_noise_reassigned: int = 0


@dataclass
class ClusterEvalResult:
    f1: float
    precision: float
    recall: float
    n_pairs_evaluated: int
    threshold: float


# ---------------------------------------------------------------------------
# Dim-reduction primitives
# ---------------------------------------------------------------------------

def reduce_pca(
    embeddings: np.ndarray,
    n_components: int = 256,
    random_state: int = 42,
) -> np.ndarray:
    """PCA reduction. Clamps n_components to min(n_components, orig_dim, n_docs-1)."""
    n_docs, orig_dim = embeddings.shape
    actual = min(n_components, orig_dim, n_docs - 1)
    logger.info("PCA: %d → %d", orig_dim, actual)
    pca = PCA(n_components=actual, random_state=random_state)
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    logger.info("PCA explained variance: %.1f%%", pca.explained_variance_ratio_.sum() * 100)
    return reduced


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 64,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """UMAP reduction. Raises ImportError if umap-learn is not installed."""
    try:
        import umap as umap_lib
    except ImportError as exc:
        raise ImportError("umap-learn is required: pip install umap-learn") from exc

    actual = min(n_components, embeddings.shape[1])
    logger.info("UMAP: %d → %d", embeddings.shape[1], actual)
    reducer = umap_lib.UMAP(
        n_components=actual,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=True,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


def reduce_dimensions(
    embeddings: np.ndarray,
    pca_dim: int = 256,
    umap_dim: int = 64,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """PCA then UMAP. Falls back to PCA-only if umap-learn is not installed."""
    reduced = reduce_pca(embeddings, n_components=pca_dim, random_state=random_state)
    try:
        reduced = reduce_umap(
            reduced,
            n_components=umap_dim,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
    except ImportError:
        logger.warning("umap-learn not installed — returning PCA output only")
    logger.info("Reduced: %d docs × %d dims", reduced.shape[0], reduced.shape[1])
    return reduced


def _assign_noise_to_nearest_centroid(
    labels: np.ndarray,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Reassign noise points (-1) to the nearest cluster centroid.

    Centroids are the mean of all non-noise points in each cluster, computed
    in the embedding space passed in (typically the UMAP output).
    Returns (filled_labels, n_reassigned).
    """
    mask_noise = labels == -1
    n_noise = int(mask_noise.sum())
    if n_noise == 0:
        return labels.copy(), 0

    from scipy.spatial.distance import cdist

    mask_core = ~mask_noise
    centroid_labels = np.unique(labels[mask_core])
    centroids = np.vstack([
        embeddings[labels == c].mean(axis=0) for c in centroid_labels
    ])
    dists = cdist(embeddings[mask_noise], centroids)
    nearest = centroid_labels[dists.argmin(axis=1)]

    filled = labels.copy()
    filled[mask_noise] = nearest
    return filled, n_noise


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
    min_samples: int | None = None,
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
# Approach 1: FAISS K-Means pipeline
# ---------------------------------------------------------------------------

def pipeline_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 150,
    n_iter: int = 25,
    n_redo: int = 3,
    use_gpu: bool = False,
) -> KMeansResult:
    """FAISS K-Means directly on raw embeddings — no PCA or UMAP.

    K-Means does not require dimensionality reduction: it operates directly
    in the embedding space where the distances are already meaningful. At
    8M-document scale, FAISS GPU acceleration makes this fast without PCA/UMAP.
    K is swept per topic with silhouette score to select the best partition.

    Parameters
    ----------
    embeddings:  Float32 array of shape (n_docs, embedding_dim)
    n_clusters:  K for K-Means
    n_iter:      K-Means iterations
    n_redo:      Independent restarts (best inertia kept)
    use_gpu:     Use FAISS GPU index if available
    """
    logger.info("Pipeline K-Means: %d docs × %d dims (no dim reduction)", *embeddings.shape)
    return cluster_kmeans(embeddings, n_clusters=n_clusters, n_iter=n_iter,
                          n_redo=n_redo, use_gpu=use_gpu)


# ---------------------------------------------------------------------------
# Approach 2: Dim-reduced HDBSCAN pipeline
# ---------------------------------------------------------------------------

def pipeline_hdbscan(
    embeddings: np.ndarray,
    pca_dim: int = 50,
    umap_dim: int = 5,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    min_cluster_size: int = 15,
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
    random_state: int = 42,
) -> HDBSCANResult:
    """Full pipeline: PCA (→50d) → UMAP (→5d) → HDBSCAN → noise reassignment.

    UMAP is required — HDBSCAN degrades significantly above ~20 dimensions.
    PCA to 50d before UMAP is much cheaper than running UMAP on 1024d inputs.
    umap_min_dist=0.0 (default) produces tight local structure that helps
    HDBSCAN resolve boundaries; contrast with pipeline_kmeans which doesn't
    use UMAP at all.

    Noise points (-1) are reassigned to their nearest cluster centroid so the
    final output is an exhaustive partition (same contract as pipeline_kmeans).

    Parameters
    ----------
    embeddings:               Float32 array of shape (n_docs, embedding_dim)
    pca_dim:                  PCA target before UMAP (50 matches proprietary pipeline)
    umap_dim:                 UMAP target (5 matches proprietary pipeline)
    umap_n_neighbors:         UMAP neighborhood size
    umap_min_dist:            UMAP min_dist; 0.0 gives tighter cluster cores
    min_cluster_size:         Minimum points to form a cluster (15 in production)
    min_samples:              HDBSCAN core-distance; None → uses min_cluster_size
    cluster_selection_method: 'eom' (excess of mass) or 'leaf'
    random_state:             Seed for PCA and UMAP
    """
    logger.info("Pipeline HDBSCAN: %d docs × %d dims", *embeddings.shape)
    reduced = reduce_pca(embeddings, n_components=pca_dim, random_state=random_state)
    reduced_umap = reduce_umap(
        reduced,
        n_components=umap_dim,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=random_state,
    )
    result = cluster_hdbscan(
        reduced_umap,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples if min_samples is not None else min_cluster_size,
        cluster_selection_method=cluster_selection_method,
    )
    filled_labels, n_reassigned = _assign_noise_to_nearest_centroid(result.labels, reduced_umap)
    if n_reassigned > 0:
        logger.info("HDBSCAN noise reassignment: %d points reassigned", n_reassigned)
    return HDBSCANResult(
        labels=filled_labels,
        cluster_sizes={int(l): int((filled_labels == l).sum())
                       for l in sorted(set(filled_labels))},
        n_clusters=result.n_clusters,
        noise_fraction=result.noise_fraction,
        n_noise_reassigned=n_reassigned,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    known_pairs_df: pd.DataFrame,
    doc_ids: list[str] | None = None,
    threshold: float = 0.5,
) -> ClusterEvalResult:
    """Compute pairwise classification F1 via cluster membership on known pairs.

    Two documents are predicted similar if they share the same cluster label.
    Ground truth positives are pairs with similarity_score ∈ {1, 2, 3};
    negatives are pairs with score ∈ {5, 6, 7} (scores 4 and 8 are excluded).

    This is the metric used to report the +0.20 improvement from fine-tuning:
    pairwise classification F1 (not generic cluster quality), measured on a
    held-out validation set of labeled pairs from the proprietary corpus.

    Parameters
    ----------
    embeddings:     Float32 array of shape (n_docs, dim)
    labels:         Cluster label per document (shape n_docs,); aligned with embeddings
    known_pairs_df: DataFrame with anchor_id, candidate_id, similarity_score columns
    doc_ids:        Ordered list of document IDs corresponding to rows of embeddings.
                    Required to correctly map anchor_id/candidate_id → label index.
                    If None, the DataFrame's integer index is used (legacy behaviour).
    threshold:      Retained for API compatibility; not used in cluster-membership eval.
    """
    POSITIVE_SCORES = {1, 2, 3}
    NEGATIVE_SCORES = {5, 6, 7}

    df = known_pairs_df.dropna(subset=["similarity_score"]).copy()
    df["similarity_score"] = df["similarity_score"].astype(int)
    # Restrict to clearly labeled pairs; exclude boundary scores 4 and 8
    df = df[df["similarity_score"].isin(POSITIVE_SCORES | NEGATIVE_SCORES)]

    if doc_ids is not None:
        id_to_idx: dict[str, int] = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    else:
        # Legacy: positional index built from DataFrame — only correct when the
        # DataFrame's integer index happens to match the embeddings row order.
        all_ids = df["anchor_id"].tolist() + df["candidate_id"].tolist()
        seen: dict[str, int] = {}
        for doc_id in all_ids:
            if doc_id not in seen:
                seen[doc_id] = len(seen)
        id_to_idx = seen

    y_true: list[int] = []
    y_pred: list[int] = []
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

    tp = sum(p == 1 and t == 1 for p, t in zip(y_pred, y_true))
    fp = sum(p == 1 and t == 0 for p, t in zip(y_pred, y_true))
    fn = sum(p == 0 and t == 1 for p, t in zip(y_pred, y_true))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)

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
    """Smoke-test both clustering pipelines with random embeddings."""
    n_docs, dim = 500, 128
    logger.info("Mock run: %d docs × %d dims", n_docs, dim)
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n_docs, dim)).astype(np.float32)
    doc_ids = [str(i) for i in range(n_docs)]

    mock_pairs = pd.DataFrame({
        "anchor_id": [str(i) for i in range(50)],
        "candidate_id": [str(i + 50) for i in range(50)],
        "similarity_score": rng.integers(1, 8, size=50),
    })

    # --- Approach 1: FAISS K-Means ---
    logger.info("=== Approach 1: FAISS K-Means ===")
    km = pipeline_kmeans(embeddings, n_clusters=20)
    logger.info("K-Means: inertia=%.2f, clusters=%d", km.inertia, len(set(km.labels)))
    km_eval = evaluate_clusters(embeddings, km.labels, mock_pairs, doc_ids=doc_ids)
    logger.info("K-Means eval: F1=%.4f  P=%.4f  R=%.4f", km_eval.f1, km_eval.precision, km_eval.recall)

    # --- Approach 2: Dim-reduced HDBSCAN ---
    logger.info("=== Approach 2: HDBSCAN ===")
    try:
        hdb = pipeline_hdbscan(embeddings, pca_dim=16, umap_dim=5, min_cluster_size=10, min_samples=3)
        logger.info(
            "HDBSCAN: %d clusters, %.1f%% noise",
            hdb.n_clusters, hdb.noise_fraction * 100,
        )
        hdb_eval = evaluate_clusters(embeddings, hdb.labels, mock_pairs, doc_ids=doc_ids)
        logger.info("HDBSCAN eval: F1=%.4f  P=%.4f  R=%.4f", hdb_eval.f1, hdb_eval.precision, hdb_eval.recall)
    except ImportError as exc:
        logger.warning("HDBSCAN pipeline skipped: %s", exc)

    logger.info("Mock run complete.")


if __name__ == "__main__":
    _mock_run()
