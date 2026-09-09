#!/usr/bin/env bash
#SBATCH --job-name=uhera-preflight
#SBATCH --account=yushanpan_surf
#SBATCH --partition=sat3090
#SBATCH --qos=sat8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/work/cpt/jiachenhou23/U-HERA/outputs/slurm/%x-%j.out
#SBATCH --error=/gpfs/work/cpt/jiachenhou23/U-HERA/outputs/slurm/%x-%j.err
set -euo pipefail
project_dir=/gpfs/work/cpt/jiachenhou23/U-HERA
source_dir="${UHERA_SOURCE_DIR:?archived source directory required}"
python_bin=/gpfs/work/cpt/jiachenhou23/MSE-Router/.venv-repro/bin/python
cd "$source_dir"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
timeout --signal=TERM 12m "$python_bin" -B -u run.py preflight \
    --config "$source_dir/configs/qwen_mosi.json" \
    --output "$project_dir/outputs/preflight/job_${SLURM_JOB_ID}"
