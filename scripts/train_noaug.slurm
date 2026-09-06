#!/usr/bin/env bash
# Single-GPU XJTLU example; choose a currently available partition/QoS at submission.
#SBATCH --job-name=router-noaug
#SBATCH --account=yushanpan_surf
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR to the absolute repository path}"
: "${PYTHON_BIN:?Set PYTHON_BIN to your environment Python}"
: "${DATASET:?Set DATASET to mosei or simsv2}"
: "${BACKBONE:?Set BACKBONE to qwen, llama2 or glm}"
: "${MODEL_PATH:?Set MODEL_PATH to the local pretrained model directory}"
: "${DATASET_PATH:?Set DATASET_PATH to the processed pickle}"
: "${MODE:?Set MODE to preflight or train}"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
extra=()
if [[ "$MODE" == train ]]; then
  : "${SEED:?Set a seed accepted by the original dataset/backbone launcher}"
  extra+=(--seed "$SEED")
elif [[ "$MODE" != preflight ]]; then
  printf 'MODE must be preflight or train\n' >&2
  exit 2
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then extra+=(--output-root "$OUTPUT_ROOT"); fi
if [[ -n "${PREFLIGHT_RESULT:-}" ]]; then extra+=(--preflight-result "$PREFLIGHT_RESULT"); fi
exec "$PYTHON_BIN" -u run_noaug.py --dataset "$DATASET" --backbone "$BACKBONE" \
  --model-path "$MODEL_PATH" --dataset-path "$DATASET_PATH" --mode "$MODE" "${extra[@]}"
