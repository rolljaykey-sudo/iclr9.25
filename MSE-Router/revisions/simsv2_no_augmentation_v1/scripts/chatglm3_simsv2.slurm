#!/usr/bin/env bash
#SBATCH --job-name=glm-simsv2-noaug
#SBATCH --account=yushanpan_surf
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=/gpfs/work/cpt/jiachenhou23/MSE-Router/outputs/chatglm3-simsv2-router-v2-no-augmentation/slurm/%x-%A_%a.out
#SBATCH --error=/gpfs/work/cpt/jiachenhou23/MSE-Router/outputs/chatglm3-simsv2-router-v2-no-augmentation/slurm/%x-%A_%a.err

set -euo pipefail
router_root=/gpfs/work/cpt/jiachenhou23/MSE-Router
revision="$router_root/revisions/simsv2_no_augmentation_v1"
torch_env=/gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/envs/torch-2.0-env
output_root="$router_root/outputs/chatglm3-simsv2-router-v2-no-augmentation"
mode="${1:-train}"
extra=()
case "$mode" in
  preflight)
    if [[ ! -f "$revision/dataset_identity.json" ]]; then
      /usr/bin/bash "$revision/scripts/inspect_simsv2.slurm"
    fi
    ;;
  train)
    seeds=(1111 2222 3333)
    task_id="${SLURM_ARRAY_TASK_ID:?Submit train as array 0-2}"
    case "$task_id" in 0|1|2) ;; *) exit 2 ;; esac
    extra=(--seed "${seeds[$task_id]}")
    ;;
  *) printf 'Expected preflight or train\n' >&2; exit 2 ;;
esac
cd "$revision"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MSE_ADAPTER_ROOT=/gpfs/work/cpt/jiachenhou23/MSE-Adapter
export LD_LIBRARY_PATH="$torch_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$router_root/.venv-repro/bin/python" -u scripts/run_chatglm3_simsv2.py "$mode" \
  --training-augmentation none --router-variant full --num-workers 0 \
  --output-root "$output_root" --preflight-result "$output_root/preflight/result.json" \
  --skip-robustness "${extra[@]}"
