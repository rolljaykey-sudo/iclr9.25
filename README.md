# conflict_text_pseudo — MOSI / SIMS, BEST MAE

本分支整理 `conflict_text_pseudo*` 实验，只提供 **MOSI、SIMS 的 ChatGLM3 代码**。
包含模型、训练器、配置、Slurm 示例及检查；不包含权重、数据集、训练结果、日志、环境或缓存。

## 模型与输入策略

- 冻结 ChatGLM3 骨干；音频、视觉编码器生成各自的伪 token。
- **文本诊断分支保留**：原始词嵌入经过带 padding mask 的注意力池化、投影和适配器，生成 4 个文本伪 token，用共享骨干与序数诊断头输出分布。
- 最终预测仍输入有序原始文本和门控后的音视频伪 token。
- 根据三模态诊断概率、Wasserstein-1 冲突、熵和模态存在标识计算路由权重；路由输入保留 `detach`。
- 随机模态丢弃、音频加噪、视觉随机遮挡和温度校准全部关闭；三个温度固定为 1。
- 配置中的 `dropout` 仅控制共享诊断头和路由器，音视频 LSTM 的 dropout 保留原配置 0。

## BEST MAE 的确切含义

延续本次实验指定的 **test MAE** 口径：每个 seed 训练完整 40 轮，每轮评估 test，保存 **MAE 最低**的检查点；完全相同时保留最早轮次。
训练结束重新加载这个检查点，生成最终 `result.json`。Acc/F1、MAE、Corr 等全部来自**同一个 MAE 最低的轮次**。
这属于 test-based checkpoint selection，并非独立测试集上的无选模评估；报告结果时应保留这一说明。

`stage1.pt`、`final.pt`、结果元数据和跨 seed 汇总均使用 MAE/min。
旧版按 F1 保存的 `result.json` 不可直接作为本分支的 BEST MAE 结果；汇总入口会拒绝这种混用。
整理前的权重和旧预检记录不随代码发布，也不能直接复用源码指纹不同的预检。

## 配置

每个数据集只有一套实现，学习率/dropout 通过 `--config` 选择；两组优化器使用配置中的 `adapter_lr` 和 `head_router_lr`。

| 数据集 | 已整理的学习率 | dropout | 默认配置 |
| --- | --- | --- | --- |
| MOSI | 0.00005、0.0001、0.0003、0.0005、0.001 | 0.1 | `mosi/configs/lr3e4_d01.json` |
| MOSI | 0.0001、0.0003 | 0.0 | 显式选择对应 `d00` 文件 |
| SIMS | 0.0001、0.0005、0.001 | 0.0 | `sims/configs/lr1e3_d00.json` |

这些是运行预设，不代表完整 MAE 选模实验均已重跑。每个预设使用 seed **1111、1113、1115**；40 轮，microbatch=4，梯度累积=4，有效 batch=16。
默认配置用于提供明确的启动方式，不声明它们在全部预设中最优。

## 外部资源与环境

使用 Python 3.10。`requirements.txt` 记录原运行环境的主要依赖版本；PyTorch/CUDA 构建应与分配到的 GPU 相容。
本实现继续依赖外部 [MSE-Adapter](https://github.com/AZYoung233/MSE-Adapter) 中的 `MSE-ChatGLM3-6B` 配置、数据加载器、指标和 ChatGLM3 模型接口。
设定 `MSE_ADAPTER_ROOT` 指向该仓库根目录；未设时默认查找本仓库的 `external/MSE-Adapter`。
源实验的上游文件 SHA-256 记录于 `EXPORT_MANIFEST.json`，方便核对本地依赖版本。

模型文件从本地目录读取，不自动下载。准备 ChatGLM3-6B-base 权重及 tokenizer 文件，设定 `CHATGLM_MODEL_PATH`，或使用 `--model-path`。
权重所需文件清单由 `scripts/chatglm_setup.py` 检查。

数据同样单独准备：

| 数据集 | 默认位置（相对仓库） | train / valid / test |
| --- | --- | --- |
| MOSI | `data/CMU-MOSI/Processed/unaligned_50.pkl` | 1284 / 229 / 686 |
| SIMS | `data/CH-SIMS/unaligned_39_router_fp16_safe.pkl` | 1368 / 456 / 457 |

可通过 `MOSI_DATASET_PATH`、`SIMS_DATASET_PATH` 或 `--dataset-path` 指定现有文件。
配置保留源实验的数据大小与 SHA-256，预检会校验。
SIMS 使用源实验已处理的 fp16-safe 文件：视觉维度 288、289、290 中绝对值大于 10000 的值替换为 0；记录在 SIMS 配置的 `deterministic_sanitization` 中。
这不是训练时的随机遮挡。原始未经处理的 SIMS 文件不能冒充该输入。
MOSI/SIMS 的上游配置模板分别复用上游内部的 `mosei` / `simsv2` 参数块，再覆盖数据集、特征维度和数据路径；它们不是本分支额外支持的数据集。

## 运行

先设置外部资源（路径均为示例）：

```bash
export PROJECT_ROOT="$(pwd)"
export MSE_ADAPTER_ROOT=/path/to/MSE-Adapter
export CHATGLM_MODEL_PATH=/path/to/chatglm3-6b-base
export MOSI_DATASET_PATH=/path/to/unaligned_50.pkl
export SIMS_DATASET_PATH=/path/to/unaligned_39_router_fp16_safe.pkl
export PYTHON_BIN=/path/to/python
```

查看参数不需要 GPU：

```bash
"$PYTHON_BIN" mosi/scripts/run_chatglm3_mosi.py --help
"$PYTHON_BIN" sims/scripts/run_chatglm3_sims.py --help
```

GPU 预检和训练必须在有效的 Slurm GPU 分配内执行。示例：

```bash
"$PYTHON_BIN" mosi/scripts/run_chatglm3_mosi.py preflight \
  --config mosi/configs/lr3e4_d01.json
"$PYTHON_BIN" mosi/scripts/run_chatglm3_mosi.py train --seed 1111 \
  --config mosi/configs/lr3e4_d01.json
```

直接运行默认输出到 `mosi/outputs/lr3e4_d01/seed_1111/`。
为 SIMS 替换入口与配置文件即可；资源路径、输出位置可通过 CLI 覆盖。
改变配置或代码后需要重新运行预检。已有完整结果请使用新的 `--output-root`，避免覆盖。

### Slurm 数组

脚本每个任务分配一张 GPU，数组下标 0/1/2 对应 1111/1113/1115。
账户、分区、QoS 由提交者提供，不在公开脚本中硬编码个人账户。提交前配置兼容的环境并创建日志目录，例如：

```bash
export EXPERIMENT_CONFIG="$PROJECT_ROOT/mosi/configs/lr3e4_d01.json"
export OUTPUT_ROOT="$PROJECT_ROOT/mosi/outputs/lr3e4_d01"
mkdir -p "$OUTPUT_ROOT/slurm"
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION --qos=YOUR_QOS \
  --output="$OUTPUT_ROOT/slurm/%x-%A_%a.out" \
  --error="$OUTPUT_ROOT/slurm/%x-%A_%a.err" \
  "$PROJECT_ROOT/mosi/scripts/train.slurm"
```

SIMS 使用 `sims/scripts/train.slurm` 与 `sims/configs/lr1e3_d00.json`，同时更换 `OUTPUT_ROOT`。
脚本在 GPU 上先做前向/反向/生成预检，成功后才训练。任务输出为 `OUTPUT_ROOT/task_<seed>/seed_<seed>/`；`execution_claim` 阻止重复任务覆盖。
`sbatch` 使用的是提交时环境；如果依赖集群模块或特定动态库，需在提交前正确配置，或在脚本中加入本地环境初始化。

### 三个 seed 的 mean

全部任务完成后运行：

```bash
"$PYTHON_BIN" mosi/scripts/run_chatglm3_mosi.py aggregate \
  --config mosi/configs/lr3e4_d01.json \
  --output-root "$OUTPUT_ROOT"
```

输出 `seed_summary.json`，包含三个 seed 的值、mean 和样本标准差；支持直接运行和数组目录结构。
MAE/Corr 保持原始数值；准确率/F1 在 JSON 中为 0–1，展示百分数时乘 100。

## 验证

```bash
python -B -m unittest discover -s tests -v
```

CPU 检查使用轻量替身覆盖最低 MAE 与最高 F1 不同轮次、MAE 并列、40 轮保存/重载和最终结果一致性，不加载完整模型权重。
真实训练仍需脚本内的 GPU 预检；本次代码整理不声称已完成新版本的完整训练。
