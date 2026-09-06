#!/usr/bin/env bash
#SBATCH --job-name=qwen-stage1-test
#SBATCH --account=yushanpan_surf
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/cpt/jiachenhou23/MSE-Router/outputs/qwen-mosei-router-v2-no-augmentation/stage1_checkpoint_test_20260905/slurm/%x-%A_%a.out
#SBATCH --error=/gpfs/work/cpt/jiachenhou23/MSE-Router/outputs/qwen-mosei-router-v2-no-augmentation/stage1_checkpoint_test_20260905/slurm/%x-%A_%a.err

set -euo pipefail
router_root=/gpfs/work/cpt/jiachenhou23/MSE-Router
revision="$router_root/revisions/no_augmentation_v1"
python_bin="$router_root/.venv-repro/bin/python"
torch_env=/gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/envs/torch-2.0-env
output_root="$router_root/outputs/qwen-mosei-router-v2-no-augmentation/stage1_checkpoint_test_20260905"
seeds=(1111 2222 3333 4444 5555)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
case "$task_id" in 0|1|2|3|4) ;; *) exit 2 ;; esac

cd "$revision"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MSE_ADAPTER_ROOT=/gpfs/work/cpt/jiachenhou23/MSE-Adapter
export LD_LIBRARY_PATH="$torch_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$python_bin" -u scripts/evaluate_qwen_stage1.py evaluate \
  --seed "${seeds[$task_id]}" --fusion checkpoint \
  --output-root "$output_root" --validate-restore
