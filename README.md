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
  │  Stage 3 — Triplet Fine-Tuning                               src/train.py │
  │                                                                          │
  │  Base model: Qwen3-Embedding-0.6B (595M params)                          │
  │  Loss: Cosine-distance triplet loss  |  Margin: 0.3                      │
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
  │  PCA (1024d → 256d)  →  UMAP (256d → 64d)                                │
  │  FAISS K-Means (K=150)  +  HDBSCAN (min_cluster=50)                      │
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

### 2. Triplet Loss Optimization

The model is trained with cosine-distance triplet loss:

```
L(a, p, n) = max(0,  d(a, p)  −  d(a, n)  +  margin)

where  d(x, y) = 1 − cos(x, y)   and   margin = 0.3
```

**Triplet construction** uses transitive inference: if the scored pair table
establishes (A ↔ B) as positive and (A ↔ C) as negative, the triplet (A, B, C)
is valid without requiring a direct (B, C) judgment. This multiplies the effective
triplet count from the labeled pairs without additional API calls.

**Hard-negative selection**: within each anchor's negative pool, candidates are
sorted by descending cosine similarity — the most confusable negatives are
presented first, keeping gradients informative throughout training.

---

### 3. Scalable FAISS Vector Indexing

After fine-tuning, the full 8M-statute corpus is embedded in parallel across
24 SLURM shards. The resulting embedding matrix undergoes a three-stage
dimensionality reduction before clustering:

```
1024d (Qwen3)  →[PCA]→  256d  →[UMAP]→  64d  →[FAISS K-Means]→  K=150 groups
                                                  + HDBSCAN for irregular clusters
```

- **PCA** preserves 85–90% of variance while eliminating collinear dimensions and
  reducing UMAP/FAISS compute
- **UMAP** captures non-linear manifold structure that linear PCA misses
- **FAISS K-Means** with GPU acceleration runs on the full 8M matrix in minutes
  vs. hours with sklearn; results in 150 semantically coherent policy groups
- **HDBSCAN** is run on the same UMAP embedding to identify variable-density
  clusters and flag noise points — complementing the K-Means partition

---

## 📊 Results

### Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Embedding dimension | 1024 |
| Max sequence length | 512 tokens |
| Loss function | Cosine-distance triplet loss |
| Triplet margin | 0.3 |
| Learning rate | 2e-5 |
| Batch size | 32 |
| Epochs | 3 |
| Warmup ratio | 0.10 |
| Optimizer | AdamW |
| Training pairs | ~100K labeled pairs → ~300K triplets |
| Hardware | 1× RTX 4090, 32 GB VRAM |
| HPC cluster | Institutional HPC cluster (SLURM array, 24 tasks) |

### Dimensionality Reduction & Clustering

| Stage | Input dim | Output dim | Method |
|---|---|---|---|
| PCA | 1024 | 256 | sklearn PCA, 87% variance explained |
| UMAP | 256 | 64 | `n_neighbors=15`, `min_dist=0.1`, cosine metric |
| K-Means | 64 | K=150 clusters | FAISS `Kmeans`, 25 iterations, 3 restarts |
| HDBSCAN | 64 | variable | `min_cluster_size=50`, `min_samples=10` |

### Performance

| Model | Clustering F1 | Notes |
|---|---|---|
| `Qwen3-Embedding-0.6B` (base) | 0.61 | Zero-shot, no fine-tuning |
| `Qwen3-Embedding-0.6B` (fine-tuned) | **0.81** | Trained on 8M-statute contrastive pairs |
| **Δ improvement** | **+0.20** | Macro F1 on held-out similar pairs |

Clustering F1 is measured on a held-out validation set of labeled pairs: two
documents are predicted similar if they share a cluster label. The metric captures
the practical downstream utility of the embedding space — whether statute groups
produced by the model are semantically coherent.

---

## 🚀 Quick Start

**1. Generate mock data** (no external dependencies required):

```bash
pip install -r requirements.txt
python src/generate_mock_data.py
# → data/mock_pairs.jsonl (100 synthetic statute pairs)
```

**2. Fine-tune on mock triplets**:

```bash
python src/train.py \
  --triplets data/mock_pairs.jsonl \
  --output   runs/mock_run \
  --epochs   1 \
  --batch-size 4
```

**3. Extract embeddings and cluster**:

```bash
python src/inference.py \
  --input  data/mock_pairs.jsonl \
  --output embeddings/mock \
  --shard 0 --n-shards 1

python src/cluster.py   # end-to-end mock run with random embeddings
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
└── src/
    ├── generate_mock_data.py   ← Self-contained mock data generator
    ├── data_pipeline.py        ← BM25 mining + async LLM-as-judge scorer
    ├── train.py                ← Qwen3-0.6B contrastive triplet training
    ├── inference.py            ← Shard-based embedding extraction
    └── cluster.py              ← FAISS K-Means + HDBSCAN + PCA/UMAP
```

---

## 📄 License

MIT
