#!/usr/bin/env bash
#SBATCH --job-name=glm-simsv2-data
#SBATCH --account=yushanpan_surf
#SBATCH --partition=cpudebug
#SBATCH --qos=cpudebug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/cpt/jiachenhou23/MSE-Router/outputs/chatglm3-simsv2-router-v2-no-augmentation/slurm/%x-%j.out
#SBATCH --error=/gpfs/work/cpt/jiachenhou23/MSE-Router/outputs/chatglm3-simsv2-router-v2-no-augmentation/slurm/%x-%j.err
set -euo pipefail
router_root=/gpfs/work/cpt/jiachenhou23/MSE-Router
cd "$router_root/revisions/simsv2_no_augmentation_v1"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
dataset=/gpfs/work/cpt/jiachenhou23/datasets/MSA/data/SIMS_V2/ch-simsv2s.pkl
deadline=$((SECONDS + 2700))
while [[ ! -f "$dataset" ]]; do
    if (( SECONDS >= deadline )); then
        printf 'Timed out waiting for the verified SIMS v2 download: %s\n' "$dataset" >&2
        exit 1
    fi
    printf '%s Waiting for the verified SIMS v2 download.\n' "$(date --iso-8601=seconds)"
    sleep 30
done
exec "$router_root/.venv-repro/bin/python" -u scripts/inspect_simsv2.py
