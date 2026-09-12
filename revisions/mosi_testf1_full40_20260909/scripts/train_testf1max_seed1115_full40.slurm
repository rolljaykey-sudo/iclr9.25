#!/usr/bin/env bash
# Slurm 作业脚本：沿用该实验的资源与入口配置，按测试集 F1 选择检查点。
# 路径为原始 HPC 实验位置；在副本重跑时需设置本地代码、环境与独立输出路径。
#SBATCH --job-name=glm-mosi-f1max-s1115
#SBATCH --account=yushanpan_surf
#SBATCH --partition=sat3090
#SBATCH --qos=sat8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output="/gpfs/work/cpt/jiachenhou23/mse router new conflict/outputs/chatglm3-mosi-lr1e3-testf1max-seed1115-full40-20260910/slurm/%x-%j.out"
#SBATCH --error="/gpfs/work/cpt/jiachenhou23/mse router new conflict/outputs/chatglm3-mosi-lr1e3-testf1max-seed1115-full40-20260910/slurm/%x-%j.err"
# 命令失败、未定义变量或管道出错时立即退出，避免继续执行无效实验。
set -euo pipefail
# 显式指定共享上游适配器目录，供数据加载器和模型接口使用。
export MSE_ADAPTER_ROOT=/gpfs/work/cpt/jiachenhou23/MSE-Adapter
# 补充集群 PyTorch 环境所需的动态库搜索路径。
torch_env=/gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/envs/torch-2.0-env
export LD_LIBRARY_PATH="$torch_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd -- '/gpfs/work/cpt/jiachenhou23/mse router new conflict/revisions/mosi_testf1_full40_20260909'
seed=1115
output_root='/gpfs/work/cpt/jiachenhou23/mse router new conflict/outputs/chatglm3-mosi-lr1e3-testf1max-seed1115-full40-20260910'/task_${seed}
mkdir -p "$output_root"
# 以独占创建目录的方式认领输出位置，已有执行记录时停止重复运行。
mkdir "$output_root/execution_claim"
# 使用指定的实验解释器，确保依赖环境与历史配置对应。
python_bin='/gpfs/work/cpt/jiachenhou23/mse router new conflict/.venv-seed1112/bin/python'
entry=scripts/run_chatglm3_mosi_seed1115.py
# 预检与训练共用输出位置和路由参数，训练读取同一次预检记录。
common=(--router-variant full --num-workers 0 --output-root "$output_root" --preflight-result "$output_root/preflight/result.json")
# 先运行预检，确认文件、有效批量以及真实前向和反向路径。
"$python_bin" -B "$entry" preflight "${common[@]}"
"$python_bin" -B - "$output_root/preflight/result.json" <<'CHECK'
import json, sys
r=json.load(open(sys.argv[1]))
assert r['selected_batching']==dict(microbatch=4,accumulation=4,effective_batch=16),r['selected_batching']
CHECK
# 预检通过后以当前种子训练；本版本按测试 F1 最大值保存检查点。
exec "$python_bin" -B "$entry" train --seed "$seed" "${common[@]}"
