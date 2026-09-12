# ChatGLM3：四个情感数据集

仅保留 ChatGLM3 在 **MOSI、MOSEI、SIMS、SIMS V2** 上的四份代码，各数据集一份。
源码已补充中文注释；所有 `outputs`、虚拟环境、缓存和其他实验版本均未打包。
本地目录：`/gpfs/work/cpt/jiachenhou23/final code`。

| 目录 | 数据集 | 选模规则 | 适配器 / 诊断头与路由器学习率 | 路由器与诊断头 dropout |
| --- | --- | --- | --- | --- |
| [mosi](mosi/) | MOSI | 测试集 `Non0_F1_score` 最大 | 0.001 / 0.001 | 0.1 |
| [mosei](mosei/) | MOSEI | 测试集 `Non0_F1_score` 最大 | 0.0001 / 0.0001 | 0.1 |
| [sims](sims/) | SIMS | 测试集 `F1_score` 最大 | 0.001 / 0.001 | 0.0 |
| [simsv2](simsv2/) | SIMS V2 | 测试集 `MAE` 最小 | 0.0001 / 0.0001 | 0.3 |

四份代码均沿用所选源版本的 40 轮联合训练、有效批量 16 和随机种子。
指标并列时保留最早轮次，最后重新加载选中的检查点评估；不执行温度校准。
SIMS V2 保留原来的测试集 MAE 选模方式。

## 文件说明

每个数据集目录包含：

- `mse_router/model.py`：ChatGLM3 专用模型、音视频编码与伪 token、诊断头、冲突路由、生成及损失。
- `mse_router/math_utils.py`、`sequence.py`、`data.py`：数学工具、有序 token 整理和数据处理。
- `mse_router/trainer.py`：联合训练、测试集选模、检查点保存与评估。
- `mse_router/diagnostics.py`：固定输入下的路由权重诊断工具。
- `scripts/run_chatglm3_<数据集>.py`：该数据集的唯一运行入口。
- `scripts/experiment.py`、`chatglm_setup.py`：预检、配置、文件检查和结果汇总。
- `scripts/train.slurm`：该数据集的作业脚本，代码和输出位置已指向本副本。

数据集配置分别记录在 `launch_spec.json` 或 `configs/<数据集>.json` 中。
保留版本的准确来源和本次检查结果见 [EXPORT_MANIFEST.json](EXPORT_MANIFEST.json)。

## 环境和运行

仍使用工作区已有的 `MSE-Adapter/MSE-ChatGLM3-6B` 上游组件、数据与
`models/THUDM/chatglm3-6b-base` 权重。可用 `MSE_ADAPTER_ROOT` 指定上游根目录，
用 `--dataset-path`、`--model-path`、`--output-root` 指定本地资源位置。
Slurm 脚本沿用原实验的共享 Python 解释器和资源参数，实际提交前应核对可用资源。
本次整理没有启动训练。

在适用的 Python 环境中，先查看对应入口的参数，例如：

```bash
python mosi/scripts/run_chatglm3_mosi.py --help
```

GPU 预检和训练应在有效 Slurm GPU 分配中执行：

```bash
python mosi/scripts/run_chatglm3_mosi.py preflight
python mosi/scripts/run_chatglm3_mosi.py train --seed 1111
```

其余数据集使用各自同名入口。默认结果位于 `<数据集>/outputs/`；
批处理脚本按 `task_<seed>/seed_<seed>/` 隔离结果。
提交批处理脚本前先创建该数据集的 `outputs/slurm/` 日志目录。
重新运行需使用本副本生成的预检记录；旧目录的预检源码散列不再适用。
路径含空格，Shell 中请加引号。

## 本次检查

四个数据集均通过轻量 ChatGLM 接口替身的模型对照：初始参数、完整及缺失模态前向、
损失、梯度和生成结果与源实现一致，冻结骨干保持无参数梯度。
训练、评估和选模方法的语法树与源版本一致；配置、文件元数据和选模模拟检查通过。
这些检查未加载完整预训练权重，也未执行完整 GPU 训练。
