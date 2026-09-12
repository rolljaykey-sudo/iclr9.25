#!/usr/bin/env bash
#SBATCH --job-name=llama32-sims-lr1e3-d00
#SBATCH --account=yushanpan_surf
#SBATCH --partition=sat3090
#SBATCH --qos=sat8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output="/gpfs/work/cpt/jiachenhou23/mse router new conflict/outputs/sims_backbone_lr1e3_d00_sat3090_20260912/llama32/slurm/%x-%A_%a.out"
#SBATCH --error="/gpfs/work/cpt/jiachenhou23/mse router new conflict/outputs/sims_backbone_lr1e3_d00_sat3090_20260912/llama32/slurm/%x-%A_%a.err"
set -euo pipefail
export MSE_ADAPTER_ROOT=/gpfs/work/cpt/jiachenhou23/MSE-Adapter
torch_env=/gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/envs/torch-2.0-env
export LD_LIBRARY_PATH="$torch_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd -- '/gpfs/work/cpt/jiachenhou23/mse router new conflict/revisions/sims_backbone_lr1e3_d00_sat3090_20260912'
seeds=(1111 1113 1115)
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"
output_root='/gpfs/work/cpt/jiachenhou23/mse router new conflict/outputs/sims_backbone_lr1e3_d00_sat3090_20260912/llama32'/task_${seed}
mkdir -p "$output_root"
mkdir "$output_root/execution_claim"
python_bin='/gpfs/work/cpt/jiachenhou23/mse router new conflict/.venv-mosi-modern/bin/python'
entry=scripts/run_transfer_sims.py
common=(--target llama32 --router-variant full --num-workers 0 --output-root "$output_root" --preflight-result "$output_root/preflight/result.json")
"$python_bin" -B "$entry" preflight "${common[@]}"
"$python_bin" -B - "$output_root/preflight/result.json" <<'CHECK'
import json, sys
r=json.load(open(sys.argv[1]))
assert r['selected_batching']==dict(microbatch=4,accumulation=4,effective_batch=16),r['selected_batching']
CHECK
exec "$python_bin" -B "$entry" train --seed "$seed" "${common[@]}"
