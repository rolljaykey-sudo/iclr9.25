#!/usr/bin/env bash
#SBATCH --job-name=qwen-noaug-compare
#SBATCH --account=yushanpan_surf
#SBATCH --partition=cpudebug
#SBATCH --qos=cpudebug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/work/cpt/jiachenhou23/MSE-Router/outputs/qwen-mosei-router-v2-no-augmentation/slurm/%x-%j.out
#SBATCH --error=/gpfs/work/cpt/jiachenhou23/MSE-Router/outputs/qwen-mosei-router-v2-no-augmentation/slurm/%x-%j.err

set -euo pipefail
router_root=/gpfs/work/cpt/jiachenhou23/MSE-Router
revision="$router_root/revisions/no_augmentation_v1"
torch_env=/gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/envs/torch-2.0-env
cd "$revision"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MSE_ADAPTER_ROOT=/gpfs/work/cpt/jiachenhou23/MSE-Adapter
export LD_LIBRARY_PATH="$torch_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$router_root/.venv-repro/bin/python" -u scripts/run_qwen_mosei_router.py aggregate \
  --training-augmentation none --seeds 1111 2222 3333 \
  --output-root "$router_root/outputs/qwen-mosei-router-v2-no-augmentation" \
  --baseline-root "$router_root/outputs/qwen-mosei-router-v2"
