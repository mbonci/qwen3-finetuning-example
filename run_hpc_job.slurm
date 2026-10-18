#!/usr/bin/env bash
#SBATCH --job-name=doc-intel-pipeline
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --array=0-23%7
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=${USER}@ethz.ch

# ===========================================================================
# Document Intelligence Pipeline — Multi-Stage Slurm Orchestrator
#
# Runs four sequential stages via job-step dependencies:
#   Stage 0  prepare    — BM25 pair sampling, cosine binning, stratified sampling
#   Stage 1  score      — Async LLM-as-a-judge scoring (gpt-4.1-mini)
#   Stage 2  train      — Qwen3-Embedding-0.6B contrastive triplet fine-tuning
#   Stage 3  inference  — Shard-based embedding extraction (array parallelism)
#   Stage 4  cluster    — PCA → UMAP → FAISS K-Means + HDBSCAN
#
# Submit:
#   sbatch run_hpc_job.slurm
#
# Cost estimate (RTX 4090, 24 array tasks × 1 h):
#   GPU-hours: 24  |  Estimated wall time: ~6 h end-to-end with staging
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

module purge
module load gcc/11.4.0 cuda/12.1.1 python/3.10.13

VENV_PATH="${HOME}/venvs/doc-intel"
if [[ ! -d "${VENV_PATH}" ]]; then
    echo "Virtual environment not found at ${VENV_PATH}" >&2
    exit 1
fi
source "${VENV_PATH}/bin/activate"

# Cap intra-process parallelism to prevent oversubscription on shared nodes
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

# Project root (set to wherever this repo is checked out on the cluster)
PROJECT_ROOT="${HOME}/repos/finetune_example"
cd "${PROJECT_ROOT}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs runs embeddings

echo "========================================="
echo " Job:   ${JOB_ID} | Task: ${TASK_ID}"
echo " Node:  $(hostname)"
echo " Start: $(date)"
echo " Stage: ${STAGE:-all}"
echo "========================================="

# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

run_prepare() {
    echo "[Stage 0] Pair sampling & candidate mining (task ${TASK_ID})"
    python -m src.data_pipeline \
        --input data/mock_pairs.jsonl \
        --output "runs/candidates_${TIMESTAMP}" \
        --task-id "${TASK_ID}" \
        --n-tasks "${SLURM_ARRAY_TASK_COUNT:-24}"
}

run_score() {
    echo "[Stage 1] LLM-as-a-judge scoring (task ${TASK_ID})"
    # Requires OPENAI_API_KEY to be set in the environment
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "WARNING: OPENAI_API_KEY not set — skipping live scoring, using mock scores"
        return 0
    fi
    CANDIDATES=$(ls -t runs/candidates_*/pairs_task_${TASK_ID}.parquet 2>/dev/null | head -1)
    if [[ -z "${CANDIDATES}" ]]; then
        echo "No candidate file found for task ${TASK_ID}; using mock data"
        CANDIDATES="data/mock_pairs.jsonl"
    fi
    python -m src.data_pipeline \
        --mode score \
        --input "${CANDIDATES}" \
        --output "runs/scored_${TIMESTAMP}" \
        --checkpoint "runs/scored_${TIMESTAMP}/checkpoint_task_${TASK_ID}.parquet"
}

run_train() {
    echo "[Stage 2] Triplet fine-tuning (task ${TASK_ID})"
    SCORED=$(ls -t runs/scored_*/global_train.parquet 2>/dev/null | head -1 || echo "data/mock_pairs.jsonl")
    python src/train.py \
        --model  "Qwen/Qwen3-Embedding-0.6B" \
        --triplets "${SCORED}" \
        --output  "runs/checkpoints_${TIMESTAMP}" \
        --epochs  3 \
        --batch-size 32 \
        --lr      2e-5 \
        --margin  0.3 \
        --seed    ${TASK_ID}
}

run_inference() {
    echo "[Stage 3] Shard-based embedding extraction (shard ${TASK_ID}/24)"
    CHECKPOINT=$(ls -td runs/checkpoints_*/task_${TASK_ID:=00} 2>/dev/null | head -1 || echo "Qwen/Qwen3-Embedding-0.6B")
    python src/inference.py \
        --model   "${CHECKPOINT}" \
        --input   "data/mock_pairs.jsonl" \
        --output  "embeddings/${TIMESTAMP}" \
        --shard   "${TASK_ID}" \
        --n-shards "${SLURM_ARRAY_TASK_COUNT:-24}"
}

run_cluster() {
    # Clustering runs only on task 0 after all embedding shards are complete
    if [[ "${TASK_ID}" -ne 0 ]]; then
        echo "[Stage 4] Skipping cluster stage for task ${TASK_ID} (runs on task 0 only)"
        return 0
    fi
    echo "[Stage 4] Clustering (task 0 only)"
    python src/cluster.py \
        --embeddings "embeddings/${TIMESTAMP}" \
        --output     "runs/clusters_${TIMESTAMP}" \
        --pca-dim    256 \
        --umap-dim   64 \
        --n-clusters 150
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

STAGE="${STAGE:-all}"

case "${STAGE}" in
    prepare)   run_prepare ;;
    score)     run_score ;;
    train)     run_train ;;
    inference) run_inference ;;
    cluster)   run_cluster ;;
    all)
        run_prepare
        run_score
        run_train
        run_inference
        run_cluster
        ;;
    *)
        echo "Unknown STAGE='${STAGE}'. Valid: prepare score train inference cluster all" >&2
        exit 1
        ;;
esac

echo "========================================="
echo " Done: $(date)"
echo "========================================="
