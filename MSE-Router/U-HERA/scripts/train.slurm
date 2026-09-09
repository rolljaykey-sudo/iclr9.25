#!/usr/bin/env bash
#SBATCH --job-name=uhera-qwen-mosi-s1111
#SBATCH --account=yushanpan_surf
#SBATCH --partition=sat3090
#SBATCH --qos=sat8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/work/cpt/jiachenhou23/U-HERA/outputs/slurm/%x-%j.out
#SBATCH --error=/gpfs/work/cpt/jiachenhou23/U-HERA/outputs/slurm/%x-%j.err

set -euo pipefail
project_dir=/gpfs/work/cpt/jiachenhou23/U-HERA
python_bin=/gpfs/work/cpt/jiachenhou23/MSE-Router/.venv-repro/bin/python
source_dir="${UHERA_SOURCE_DIR:-$project_dir}"
preflight_dir="$project_dir/outputs/preflight/job_${SLURM_JOB_ID}"
cd "$source_dir"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# A failed/expired preflight prevents production training in the same allocation.
timeout --signal=TERM 15m "$python_bin" -B -u run.py preflight \
    --config "$source_dir/configs/qwen_mosi.json" --output "$preflight_dir"
"$python_bin" -B -u run.py train --config "$source_dir/configs/qwen_mosi.json" \
    --preflight-report "$preflight_dir/preflight.json" --resume
