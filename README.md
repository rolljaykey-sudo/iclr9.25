# Final Code：测试集 F1 选模版本

本仓库从 `/gpfs/work/cpt/jiachenhou23/mse router new conflict` 独立复制，
本地副本为 `/gpfs/work/cpt/jiachenhou23/final code`。
仅保留实际按**测试集 F1 最大值**选择检查点的 13 个实验版本，源码已补充中文注释。
原始目录不变；所有层级的 `outputs`、虚拟环境和 Python 缓存均未复制。

## 选模规则

| 数据集 | 选模指标 | 规则 |
| --- | --- | --- |
| MOSI、MOSEI | 测试集 `Non0_F1_score` | 40 轮中取最大值，F1 并列保留最早轮次 |
| SIMS | 测试集 `F1_score` | 40 轮中取最大值，F1 并列保留最早轮次 |

这些版本使用联合训练，最终重新加载所选检查点进行评估，`fit` 不执行温度校准。
MAE、相关系数等仍作为伴随指标记录，**不参与检查点选择或并列比较**。
验证集选模、测试集 MAE 选模版本以及指向旧 MAE 版本的两个遗留启动脚本已从副本移除。

## 保留的实验

每个 `revisions/<版本>/` 都保存对应的 `mse_router/` 实现、`scripts/` 入口及配置。
按目录内的实际配置选择入口；不要把不同版本的训练器和配置混用。

| 数据集 | 实验目录 | 主要入口 |
| --- | --- | --- |
| MOSEI | [mosei_lr1e4_testf1_20260910](revisions/mosei_lr1e4_testf1_20260910/) | `scripts/run_chatglm3_mosei.py` |
| MOSEI | [mosei_lr5e3_head1e3_testf1_20260910](revisions/mosei_lr5e3_head1e3_testf1_20260910/) | `scripts/run_chatglm3_mosei.py` |
| MOSI | [mosi_backbone_lr1e3_d01_20260912](revisions/mosi_backbone_lr1e3_d01_20260912/) | `scripts/run_transfer_mosi.py` |
| MOSI | [mosi_llama32_lr1e3_d01_20260912](revisions/mosi_llama32_lr1e3_d01_20260912/) | `scripts/run_transfer_mosi.py` |
| MOSI | [mosi_lr1e3_d00_three_seeds_20260910](revisions/mosi_lr1e3_d00_three_seeds_20260910/) | `scripts/run_chatglm3_mosi.py` |
| MOSI | [mosi_lr1e4_testf1_full40_20260910](revisions/mosi_lr1e4_testf1_full40_20260910/) | `scripts/run_chatglm3_mosi.py` |
| MOSI | [mosi_testf1_full40_20260909](revisions/mosi_testf1_full40_20260909/) | `scripts/run_chatglm3_mosi.py` |
| SIMS | [sims_backbone_lr1e3_d00_sat3090_20260912](revisions/sims_backbone_lr1e3_d00_sat3090_20260912/) | `scripts/run_transfer_sims.py` |
| SIMS | [sims_lr1e3_d00_testf1_three_seeds_20260910](revisions/sims_lr1e3_d00_testf1_three_seeds_20260910/) | `scripts/run_chatglm3_sims.py` |
| SIMS | [sims_lr1e3_testf1_three_seeds_20260910](revisions/sims_lr1e3_testf1_three_seeds_20260910/) | `scripts/run_chatglm3_sims.py` |
| SIMS | [sims_lr1e4_testf1_three_seeds_20260910](revisions/sims_lr1e4_testf1_three_seeds_20260910/) | `scripts/run_chatglm3_sims.py` |
| SIMS | [sims_lr5e3_head1e3_s1113_s1115_20260910](revisions/sims_lr5e3_head1e3_s1113_s1115_20260910/) | `scripts/run_chatglm3_sims.py` |
| SIMS | [sims_testf1_extra_full40_20260909](revisions/sims_testf1_extra_full40_20260909/) | `scripts/run_chatglm3_sims.py` |

## 中文注释导航

- `mse_router/math_utils.py`：连续标签插值、熵、Wasserstein-1 冲突、缺失模态权重。
- `mse_router/model.py`：冻结骨干、单模态诊断、30 维路由输入、音视频门控、生成与辅助损失。
- `mse_router/sequence.py`：有序 token 压紧、左侧填充及监督标签对齐。
- `mse_router/data.py`：视频分组划分、批次迁移、音视频扰动及模态存在掩码。
- `mse_router/trainer.py`：梯度累积、混合精度、测试 F1 选模、保存及重载检查点。
- `mse_router/backbone_model.py`：不同语言模型的加载方式和前向接口。
- `scripts/`：数据集与骨干配置、预检、随机种子入口及 Slurm 作业说明。

## 环境与历史记录

本次整理保留原始算法、超参数、配置和路径常量。源码仍依赖工作区外部的
`MSE-Adapter`、数据、预训练模型及对应 Python 环境，这些资源未打包。
Slurm 脚本中的代码、日志和输出路径仍是原始 HPC 实验路径；在副本中重跑时，
应将它们设置为自己的代码目录和独立输出目录，并在有效 Slurm GPU 分配中执行。
目录名称含空格，Shell 中请为路径加引号。

注释会改变源码 SHA-256，所以副本需要重新生成预检记录后才能训练。
各版本原有的 `validation.json`、`source_hashes.json`、`baseline_verification.json`
等文件保留为历史记录，不代表注释后版本的新验证结果。
本次导出的文件清单、来源散列和检查结果见 [EXPORT_MANIFEST.json](EXPORT_MANIFEST.json)。
