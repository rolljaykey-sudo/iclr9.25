#!/usr/bin/env bash
#SBATCH --job-name=pseudotext-sims-best-mae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
set -euo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the absolute repository path}"
: "${MSE_ADAPTER_ROOT:?Set MSE_ADAPTER_ROOT to the external MSE-Adapter checkout}"
python_bin="${PYTHON_BIN:-python}"
config="${EXPERIMENT_CONFIG:-$PROJECT_ROOT/sims/configs/lr1e3_d00.json}"
config_tag="$(basename "$config" .json)"
output_base="${OUTPUT_ROOT:-$PROJECT_ROOT/sims/outputs/$config_tag}"
seeds=(1111 1113 1115)
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"
output_root="$output_base/task_$seed"
mkdir -p "$output_root"
mkdir "$output_root/execution_claim"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREAsims=1 MKL_NUM_THREAsims=1 OPENBLAS_NUM_THREAsims=1 NUMEXPR_NUM_THREAsims=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
entry="sims/scripts/run_chatglm3_sims.py"
common=(--config "$config" --output-root "$output_root" --preflight-result "$output_root/preflight/result.json" --num-workers 0)
"$python_bin" -B "$entry" preflight "${common[@]}"
exec "$python_bin" -B "$entry" train --seed "$seed" "${common[@]}"
