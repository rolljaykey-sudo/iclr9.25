# CoMoR: Modeling Cross-Modal Conflict for Adaptive Evidence Routing

**CoMoR**（Conflict-Aware Modality Routing，模态冲突感知路由）研究跨模态冲突的显式建模与自适应证据路由，以多模态情感分析作为实验载体。

本目录整理了 2026-09-12 实际提交训练的版本
`sims_backbone_lr1e3_d00_sat3090_20260912`。完整中文说明见
**[VERSION.md](VERSION.md)**，涵盖模型结构、数据处理、训练设置、模型选择规则和复现步骤。

训练入口为 `scripts/run_transfer_sims.py`，支持 Qwen-1.8B、Llama-2-7B 和
Llama-3.2-3B。三个骨干均冻结，使用同一套 CH-SIMS 数据、Wasserstein 冲突路由与联合训练流程。

这是实际实验代码的归档：`mse_router/`、原始 `scripts/`、`configs/` 保持源文件内容，
并补齐其依赖的上游 Python 文件。数据集、预训练权重、训练 checkpoint、虚拟环境及完整日志不在归档内。

该实验每轮在 **test** 集计算 F1，选取 40 轮中 F1 最高的 checkpoint，平分时取最早轮次。
因此它的结果是 **test-selected** 实验结果，不能表述为通过独立保留测试集得到的无偏最终评估。

```bash
# 仅用 Python 标准库核对原始源码哈希、Python/JSON 语法和 Slurm 脚本语法。
python tools/verify_snapshot.py
```

实际训练需配置外部数据/模型路径并处于 Slurm GPU allocation 中，见 [复现说明](VERSION.md#复现步骤)。
原始 Slurm 脚本保留 XJTLU 提交时的绝对路径，不能直接当成任意机器的启动脚本。
