# 📄 Document Intelligence & Representation Learning Pipeline

![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![HuggingFace](https://img.shields.io/badge/🤗-Qwen3--Embedding--0.6B-orange)
![SLURM](https://img.shields.io/badge/HPC-SLURM%20Array-purple?logo=linux&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-K--Means%20%7C%20HDBSCAN-red)

**Fine-tuned `Qwen3-Embedding-0.6B` on 8 million proprietary legal statutes using contrastive triplet loss, yielding a +0.20 improvement in clustering F1 on semantically similar document pairs. Scaled inference across the full corpus to produce actionable semantic groups via FAISS K-Means and HDBSCAN.**

---

## 🔒 Data Privacy & Compliance Notice

> The training corpus consists of **8 million proprietary legal statutes** sourced from a state-law research archive. Raw text cannot be shared publicly due to data-use agreements and compliance requirements.

This repository implements a **mock data strategy** that faithfully mimics the pipeline mechanics without exposing any real corpus content:

- `src/generate_mock_data.py` produces synthetic statute-like text pairs using topic-aligned templates drawn from real legislative domains (Health, Education, Government Operations, Environment, etc.)
- Mock pairs reproduce the **boundary-sampling mechanics** of the real pipeline: cosine similarity bins (q40–q100), BM25-retrieved hard negatives, and ordinal judge scores calibrated to mirror the real label distribution
- All training, inference, and clustering scripts run end-to-end on the mock dataset with no modification
- Real results reported in this README were produced by running the full pipeline on the proprietary corpus on an institutional HPC cluster

---

## 🏗️ Architecture & Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Document Intelligence Pipeline                           │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐
  │  Raw Corpus      │  8M legal statutes across 50 US states
  │  (proprietary)   │  Deduplicated, topic-classified (8 domains)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Stage 1 — Contrastive Pair Mining                  src/data_pipeline.py │
  │                                                                          │
  │  ┌─────────────────────────┐   ┌──────────────────────────────────────┐  │
  │  │  BM25 Hard-Negative     │   │  Random Within-Topic Sampling        │  │
  │  │  Mining                 │   │                                      │  │
  │  │  • Build BM25 index     │   │  • 2.5M random within-topic pairs    │  │
  │  │    per topic            │   │  • Enforce different-state pairs     │  │
  │  │  • Sample 5% anchors    │   │  • Deduplication (order-invariant)   │  │
  │  │  • Retrieve top-100     │   │                                      │  │
  │  │    per anchor           │   └──────────────────────────────────────┘  │
  │  └────────────┬────────────┘                         │                   │
  │               └───────────────── 80% / 20% ─────────┘                   │
  │                                        │                                 │
  │                                        ▼                                 │
  │                        Cosine Similarity Binning                         │
  │                   q40-60 | q60-80 | q80-90 | q90-95 | q95-100           │
  │                        Stratified sample → 100K pairs                    │
  └──────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Stage 2 — LLM-as-a-Judge Scoring                   src/data_pipeline.py │
  │                                                                          │
  │  Model: gpt-4.1-mini  |  Concurrency: 35 workers  |  Scores: 1–8        │
  │                                                                          │
  │   1-3 → Positive (related)    4-7 → Negative (unrelated)   8 → N/A      │
  │                                                                          │
  │  Exponential backoff retry  |  Checkpoint every 20K rows                 │
  └──────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Stage 3 — Hybrid Contrastive + Triplet Fine-Tuning         src/train.py │
  │                                                                          │
  │  Base model: Qwen3-Embedding-0.6B (595M params)                          │
  │  Objective 1: WeightedContrastiveLoss  |  margin=0.7  |  batch=8        │
  │  Objective 2: TripletLoss (cosine)     |  margin=0.5  |  batch=32       │
  │  Two objectives interleaved via model.fit(train_objectives=[...])        │
  │  Triplets: on-the-fly construction with transitive inference             │
  │  Hard negatives selected by descending cosine rank                       │
  │                                                                          │
  │  ┌──────────────────────────┐   ┌──────────────────────────────────────┐  │
  │  │  SLURM Array Job         │   │  Evaluation                          │  │
  │  │  24 tasks × 1 GPU        │   │  Per-topic binary F1 on held-out     │  │
  │  │  RTX 4090 × 32GB         │   │  validation pairs                    │  │
  │  │  Institutional HPC       │   │  Best checkpoint selection           │  │
  │  └──────────────────────────┘   └──────────────────────────────────────┘  │
  └──────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Stage 4 — Corpus-Scale Inference                       src/inference.py │
  │                                                                          │
  │  8M statutes processed in 24 SLURM shards                                │
  │  CUDA OOM retry: halves batch size on failure                            │
  │  Output: zstd-compressed parquet, Arrow fixed-size-list embeddings       │
  └──────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Stage 5 — Semantic Clustering                           src/cluster.py  │
  │                                                                          │
  │  Approach 1 — FAISS K-Means                                              │
  │    PCA (1024d → 256d)  →  UMAP (256d → 64d)  →  FAISS K-Means (K=150)  │
  │    Exhaustive partition; every document assigned to a cluster             │
  │                                                                           │
  │  Approach 2 — Dim-reduced HDBSCAN                                        │
  │    PCA (1024d → 256d)  →  UMAP (256d → 64d, min_dist=0.0)               │
  │    → HDBSCAN (min_cluster=50)  — noise points labelled -1                │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Architecture Deep Dive

### 1. Hard-Negative Mining with BM25

Naïve random sampling for contrastive learning produces easy negatives that carry
weak training signal. The pipeline mines **hard negatives** — statutes that are
lexically similar but semantically distinct — by building a BM25 index per topic
and retrieving the top-100 candidates for each sampled anchor.

```
For each topic domain T:
  anchors ← sample(members_T, frac=0.05)
  For each anchor a:
    candidates ← bm25_index_T.top_k(a, k=100)
    cosine_scores ← embed(candidates) · embed(a)
    assign to bin: q40-60 | q60-80 | q80-90 | q90-95 | q95-100

Merge BM25 (80%) + random (20%) → stratified sample → 100K pairs
```

Cosine quantile bins ensure the training set is **balanced across the decision
boundary**: the model sees pairs at every difficulty level, not just easy
positives and random negatives.

---

### 2. Hybrid Contrastive + Triplet Loss

Training uses two interleaved objectives via `model.fit(train_objectives=[...])`:

**Objective 1 — WeightedContrastiveLoss (margin = 0.7)**
```
L(a, b, y, w) = w × 0.5 × [y × d(a,b)²  +  (1-y) × max(0, margin − d(a,b))²]

where  d(a, b) = 1 − cos(a, b)   and   y ∈ {0, 1}
```
Per-score sampling multipliers `{1: 4.0, 2: 2.0, 3: 0.5, 4: 0.2, 5: 0.2, 6: 1.0, 7: 4.0}`
upweight hard/informative examples. Score 4 (boundary) is included as a hard
training negative with low weight; it is excluded from the evaluation metric.

**Objective 2 — TripletLoss (cosine distance, margin = 0.5)**
```
L(a, p, n) = max(0,  d(a, p)  −  d(a, n)  +  margin)

where  d(x, y) = 1 − cos(x, y)   and   margin = 0.5
```

**Triplet construction** uses transitive inference: if the scored pair table
establishes (A ↔ B) as positive and (A ↔ C) as negative, the triplet (A, B, C)
is valid without requiring a direct (B, C) judgment. This multiplies the effective
triplet count from the labeled pairs without additional API calls.

**Hard-negative selection**: within each anchor's negative pool, candidates are
sorted by descending cosine similarity — the most confusable negatives are
presented first, keeping gradients informative throughout training.

Both losses call `model(sentence_features[i])["sentence_embedding"]` — the
differentiable forward path through the transformer. Using `model.encode()` in
a training loop is a fatal bug: it runs under `torch.no_grad()` and
`loss.backward()` becomes a no-op.

---

### 3. Two-Approach Semantic Clustering

After fine-tuning, the full 8M-statute corpus is embedded in parallel across
24 SLURM shards. Two independent clustering pipelines are run on the resulting
embedding matrix:

**Approach 1 — FAISS K-Means** (`pipeline_kmeans`):
```
1024d (Qwen3)  →[PCA]→  256d  →[UMAP]→  64d  →[FAISS K-Means]→  K=150 groups
```
- Produces an **exhaustive partition**: every document is assigned exactly one cluster
- UMAP is optional — can be skipped when embeddings are already low-dimensional
- FAISS K-Means with GPU acceleration runs on the full 8M matrix in minutes vs. hours for sklearn
- Best K is selected on the training split only (see `src/evaluate.py`)

**Approach 2 — Dim-reduced HDBSCAN** (`pipeline_hdbscan`):
```
1024d (Qwen3)  →[PCA]→  256d  →[UMAP, min_dist=0.0]→  64d  →[HDBSCAN]→  variable groups + noise
```
- UMAP is **required** — HDBSCAN degrades significantly above ~64 dimensions (curse of dimensionality)
- `min_dist=0.0` produces tighter local structure than the K-Means pipeline, which helps
  HDBSCAN resolve cluster boundaries at low dimensionality
- Documents that don't fit any cluster are labelled `-1` (noise), surfacing coherent
  semantic groups without forcing every document into one
- Complements K-Means by finding variable-density and irregular-shaped clusters

**Shared components:**
- **PCA** preserves 85–90% of variance while eliminating collinear dimensions
- **UMAP** captures non-linear manifold structure that linear PCA misses
- `evaluate_clusters()` — same pairwise classification F1 metric for both approaches

---

## 📊 Results

### Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Embedding dimension | 1024 |
| Max sequence length | 512 tokens |
| Loss 1 | WeightedContrastiveLoss (cosine distance) |
| Contrastive margin | 0.7 |
| Contrastive batch size | 8 |
| Loss 2 | TripletLoss (cosine distance) |
| Triplet margin | 0.5 |
| Triplet batch size | 32 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Warmup steps | 100 |
| Optimizer | AdamW |
| Training pairs | ~100K labeled pairs → ~300K triplets |
| Hardware | 1× RTX 4090, 32 GB VRAM |
| HPC cluster | Institutional HPC cluster (SLURM array, 24 tasks) |

### Dimensionality Reduction & Clustering

**Approach 1 — FAISS K-Means (`pipeline_kmeans`)**

| Stage | Input dim | Output dim | Method |
|---|---|---|---|
| PCA | 1024 | 256 | sklearn PCA, 87% variance explained |
| UMAP | 256 | 64 | `n_neighbors=15`, `min_dist=0.1`, cosine metric |
| FAISS K-Means | 64 | K=150 clusters | FAISS `Kmeans`, 25 iterations, 3 restarts |

**Approach 2 — Dim-reduced HDBSCAN (`pipeline_hdbscan`)**

| Stage | Input dim | Output dim | Method |
|---|---|---|---|
| PCA | 1024 | 256 | sklearn PCA |
| UMAP | 256 | 64 | `n_neighbors=15`, `min_dist=0.0` (tighter local structure) |
| HDBSCAN | 64 | variable | `min_cluster_size=50`, `min_samples=10`, `cluster_selection=eom` |

### Performance

> **Historical research results — proprietary corpus only.** The numbers below
> were produced by running the full pipeline on the institutional HPC cluster
> against the 8M-statute corpus. They cannot be independently reproduced from
> this public repository because the training corpus is not included. `src/experiment.py`
> provides a synthetic demo that verifies gradient flow and metric correctness on
> mock data; its output should not be compared to these figures.

| Model | Pairwise Classification F1 | Notes |
|---|---|---|
| `Qwen3-Embedding-0.6B` (base) | 0.61 | Zero-shot, no fine-tuning |
| `Qwen3-Embedding-0.6B` (fine-tuned) | **0.81** | Trained on 8M-statute contrastive pairs |
| **Δ improvement** | **+0.20** | On held-out labeled pairs, proprietary corpus |

**Metric definition** — pairwise classification F1 via cluster membership: two documents
are predicted similar if they share the same K-Means cluster label. Ground-truth positives
are pairs with LLM-judge score ∈ {1, 2, 3}; negatives use scores ∈ {5, 6, 7} (scores 4
and 8 are excluded). Best K is selected on the training split only; the test split is held
out until after K is chosen to prevent label leakage. See `src/evaluate.py`.

---

## 🚀 Quick Start

**1. Generate mock data** (no external dependencies required):

```bash
pip install -r requirements.txt
python src/generate_mock_data.py
# → data/mock_pairs.jsonl (~200 anchor-centric statute pairs)
# Each anchor has both positives (score 1–3) and negatives (score 5–7)
# so that triplet construction produces non-zero training examples.
```

**2. Run the smoke tests** (no GPU or model download required):

```bash
pytest tests/ -v -k "not slow"
```

**3. Fine-tune on mock triplets**:

```bash
python src/train.py \
  --triplets data/mock_pairs.jsonl \
  --output   runs/mock_run \
  --epochs   1 \
  --batch-size 4
```

**4. Extract embeddings and cluster**:

```bash
python src/inference.py \
  --input  data/mock_pairs.jsonl \
  --output embeddings/mock \
  --shard 0 --n-shards 1

python src/cluster.py   # end-to-end mock run with random embeddings
```

**5. Baseline vs. fine-tuned experiment (synthetic demo)**:

```bash
python src/experiment.py \
  --pairs  data/mock_pairs.jsonl \
  --output runs/experiment \
  --epochs 2
# Note: results are from synthetic data; see Performance section above.
```

**HPC (full corpus)** — submit the SLURM array job:

```bash
sbatch run_hpc_job.slurm
# 24 array tasks × 1× RTX 4090 × 1h wall time
# Stages: prepare → score → train → inference → cluster
```

---

## 📁 Repository Structure

```
finetune_example/
│
├── README.md                   ← You are here
├── requirements.txt
├── run_hpc_job.slurm           ← Multi-stage SLURM array orchestrator
│
├── data/
│   ├── mock_pairs.jsonl        ← Synthetic statute pairs (generated)
│   └── mock_config.json        ← Sampling & training hyperparameters
│
├── tests/
│   ├── test_triplets.py        ← Triplet construction (zero-triplets regression)
│   ├── test_metrics.py         ← Pairwise F1 metric correctness
│   ├── test_gradient.py        ← Gradient flow verification (slow, requires model)
│   └── test_smoke.py           ← End-to-end pipeline smoke tests
│
└── src/
    ├── generate_mock_data.py   ← Self-contained mock data generator
    ├── data_pipeline.py        ← BM25 mining + async LLM-as-judge scorer
    ├── train.py                ← Qwen3-0.6B contrastive triplet training
    ├── inference.py            ← Shard-based embedding extraction
    ├── cluster.py              ← FAISS K-Means + HDBSCAN + PCA/UMAP
    ├── evaluate.py             ← Pairwise classification F1 (K sweep, train/test split)
    └── experiment.py           ← Synthetic baseline vs. fine-tuned demo
```

---

## 📄 License

MIT
