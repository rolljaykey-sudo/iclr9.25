#!/usr/bin/env bash
# ChatGLM3 SIMS：测试集 F1_score 选模；资源参数沿用源版本，实际提交前按平台规则核对。
#SBATCH --job-name=glm-sims-lr1e3-d00
#SBATCH --account=yushanpan_surf
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output="/gpfs/work/cpt/jiachenhou23/final code/sims/outputs/slurm/%x-%A_%a.out"
#SBATCH --error="/gpfs/work/cpt/jiachenhou23/final code/sims/outputs/slurm/%x-%A_%a.err"
# 任何命令失败即停止，避免沿用失败的预检。
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

cd -- '/gpfs/work/cpt/jiachenhou23/final code/sims'
# 作业数组下标对应以下固定随机种子。
seeds=(1111 1113 1115)
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"
output_root='/gpfs/work/cpt/jiachenhou23/final code/sims/outputs'/task_${seed}
mkdir -p "$output_root"
# 输出目录只允许认领一次，阻止重复作业覆盖结果。
mkdir "$output_root/execution_claim"
python_bin='/gpfs/work/cpt/jiachenhou23/mse router new conflict/.venv-seed1112/bin/python'
entry=scripts/run_chatglm3_sims.py
common=(--router-variant full --num-workers 0 --output-root "$output_root" --preflight-result "$output_root/preflight/result.json")
# 先校验数据、模型、批量及前向反向路径。
"$python_bin" -B "$entry" preflight "${common[@]}"
"$python_bin" -B - "$output_root/preflight/result.json" <<'CHECK'
import json, sys
r=json.load(open(sys.argv[1]))
assert r['selected_batching']==dict(microbatch=4,accumulation=4,effective_batch=16),r['selected_batching']
CHECK
# 预检通过后执行当前种子的完整联合训练。
exec "$python_bin" -B "$entry" train --seed "$seed" "${common[@]}"
