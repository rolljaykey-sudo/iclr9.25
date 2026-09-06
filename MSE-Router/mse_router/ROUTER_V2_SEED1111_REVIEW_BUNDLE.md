# Router V2 / MOSEI / seed1111 单文件审查包

> 快照时间：2026-09-04 09:34:29（Asia/Shanghai）  
> 项目：`/gpfs/work/cpt/jiachenhou23/MSE-Router`  
> Git HEAD：`e0617fec510b2f2c4539e7f8b000efb3dbd98740`  
> 当前 V2 架构：`natural_text_residual_gated_audio_vision_v2`

这个文件把本次实验自有的完整代码、实际调用的本地上游数据/配置/指标代码、Slurm 提交代码、测试、seed1111 的精确 JSON 结果，以及当前评价合并在一起，便于逐行审查。后面的代码块按生成时的原文件内容嵌入；每个文件前给出原文件 SHA-256。三个上游 Python 文件原本没有 EOF 换行（`config_regression.py`、`load_data.py`、`metricsTop.py`），Markdown 为闭合代码围栏显示了一个结构性换行；字符内容没有变化，哈希仍以原文件为准。

## 1. 先给结论

seed1111 已正常完成，测试集主指标为 **MAE 0.5128、Corr 0.7774**；没有无效生成，测试时平均 Router 权重为 **文本 0.4951、音频 0.2393、视觉 0.2657**。与旧版同 seed 基线（MAE 0.5918、Corr 0.6999）相比，MAE 下降约 **0.0790（13.3%）**，Corr 上升 **0.0775**。

我的当前判断是：**这是一个可信且有竞争力的单 seed 结果，但还不是“Router 的效果已经被证明”或“五 seed 结论已经稳定”的证据。** 数据流检查未发现标签泄漏或 train/valid/test 样本重叠；指标也明显好于常数预测。然而，阶段一最佳验证 MAE 为 0.5072，而 Router-only 阶段最佳验证 MAE 为 0.5083，后者没有改善阶段一。因此，现阶段只能把提升归因于完整 V2 设计和联合训练，不能单独归因于最后的 Router-only 微调。

另一个重要审查缺口是：当前结果文件只保存聚合指标、平均权重和原始生成计数，**没有保存逐样本 ID、标签、预测值、原始生成字符串和逐样本权重**。所以代码路径可以检查，但无法仅靠现有结果文件离线、独立复算最终 MAE/Corr。这不等于结果错误，但复核强度还不够。

## 2. 当前方法与“故事”

原始 MSE-Adapter 的核心是把多模态信息转成 LLM 可处理的表示。V2 延续这一叙事，但不再让所有模态固定贡献：

1. 原始文本保留为自然语言 token，避免把最强的语义锚点完全压进少量伪 token。
2. 音频和视觉各自由 Adapter 编码为残差式伪 token。
3. 诊断分支分别形成文本、音频、视觉的离散情感分布。
4. 用两两 Jensen–Shannon 散度表示冲突，用分布熵表示不确定性。
5. 两层 MLP Router 根据预测、冲突、不确定性和模态存在掩码产生样本级权重。
6. 权重作用于音频/视觉残差表示；加权后的联合输入再由冻结 Qwen 生成最终连续情感值。
7. 训练分为联合训练、温度校准、Router-only 三段；最终载入最佳 checkpoint 做 clean test 和鲁棒性评估。

可概括为：

```text
自然文本 + 音频/视觉特征
          │
          ├─ 各模态诊断分布 ── 冲突(JS) + 不确定性(熵)
          │                                  │
          └─ 表示分支 ──────────────────── Router 权重
                                             │
                               门控音频/视觉残差伪 token
                                             │
                                      Frozen Qwen
                                             │
                                      最终情感数值
```

完整叙事说明另存于 `mse_router/ARCHITECTURE_STORY_ZH.md`；本文件下面嵌入的 `model.py` 和 `trainer.py` 是最终审查依据。

## 3. 训练逻辑摘要

- 优化器：**AdamW**，不是经典 Adam；`eps=1e-4`，`weight_decay=0.01`。
- 阶段一：Adapter 学习率 `5e-3`，ordinal head 与 Router 学习率 `1e-3`；最多 40 epoch，patience 10。
- 温度校准：在独立 calibration split 上，分别最小化各模态诊断分布的 NLL；不是用 MSE 处理输入 token。
- Router-only：只更新 Router，学习率 `1e-3`；最多 10 epoch，patience 3。
- 冻结项：Qwen 参数不更新；梯度仍可穿过 Qwen 回到可训练模块。
- 最终监督：生成式损失与 soft ordinal 辅助损失组合；具体公式和权重以嵌入代码为准。
- 阶段一结束标准：验证 MAE 改善时刷新最佳 epoch；距离最佳 epoch 达到 patience 10 即停止，或达到 40 epoch。seed1111 在 epoch 9 最佳，训练到 epoch 19 后结束。
- 数据批量：microbatch 4，gradient accumulation 4，有效 batch 16。
- 随机种子：本份最终结果明确是 **1111**；calibration 子集的固定划分 seed 是 **20260903**，不要与训练随机种子混淆。

## 4. seed1111 精确结果摘要

### 4.1 训练与校准

| 环节 | 最佳 epoch | 最佳 valid MAE | 实际 epoch | 说明 |
|---|---:|---:|---:|---|
| 阶段一联合训练 | 9 | 0.5072 | 19 | best epoch 后连续 10 epoch 未改善而停止 |
| 温度校准 | — | — | — | 1,648 个 calibration 样本；校准有效 |
| Router-only | 5 | 0.5083 | 8 | best epoch 后连续 3 epoch 未改善而停止 |

温度分别为：文本 1.10717、音频 1.16684、视觉 1.11517。三者校准 NLL 均小幅下降：

| 模态 | NLL（tau=1） | NLL（校准后） |
|---|---:|---:|
| 文本 | 1.57134 | 1.56826 |
| 音频 | 1.57476 | 1.56757 |
| 视觉 | 1.54796 | 1.54430 |

### 4.2 最终 clean test

| 指标 | 数值 |
|---|---:|
| Has0 Acc-2 | 0.8566 |
| Has0 F1 | 0.8514 |
| Non0 Acc-2 | 0.7617 |
| Non0 F1 | 0.7646 |
| Mult Acc-5 | 0.5697 |
| Mult Acc-7 | 0.5503 |
| MAE | **0.5128** |
| Corr | **0.7774** |
| 测试样本 | 4,659 |
| invalid generations | 0 |
| out-of-range generations | 0 |
| 平均权重 T/A/V | 0.4951 / 0.2393 / 0.2657 |

应用层 elapsed 为 30,404.75 秒（约 8:26:45），Slurm 计时为 08:27:29；峰值 allocated GPU memory 为 5,184,145,408 bytes，reserved 为 5,595,201,536 bytes，MaxRSS 为 19,585,847,296 bytes。

### 4.3 鲁棒性

| 条件 | MAE | Corr | 平均 T/A/V 权重 |
|---|---:|---:|---|
| clean | 0.5128 | 0.7774 | 0.4951 / 0.2393 / 0.2657 |
| missing_text | 0.8145 | 0.1714 | 0 / 0.4745 / 0.5255 |
| missing_audio | 0.5507 | 0.7604 | 0.5968 / 0 / 0.4032 |
| missing_vision | 0.5993 | 0.7000 | 0.5275 / 0.4725 / 0 |
| audio_snr_20 | 0.5122 | 0.7777 | 近似不变 |
| audio_snr_10 | 0.5131 | 0.7770 | 近似不变 |
| audio_snr_0 | 0.5133 | 0.7766 | 近似不变 |
| vision_mask_25 | 0.5152 | 0.7768 | 近似不变 |
| vision_mask_50 | 0.5180 | 0.7747 | 近似不变 |
| vision_mask_75 | 0.5212 | 0.7737 | 近似不变 |

这说明模型对音频噪声和视觉随机遮挡有一定稳健性，但“权重会随噪声动态显著变化”的故事目前没有由平均权重支持；权重几乎不变。缺失文本时性能大幅下降，说明文本仍是语义锚点。missing_vision 出现 1 个 out-of-range generation，也应保留关注。

## 5. 基线与提升幅度

旧版 `qwen-mosei-repro` 已完成的三个 seed：

| seed | MAE | Corr |
|---:|---:|---:|
| 1111 | 0.5918 | 0.6999 |
| 2222 | 0.5291 | 0.7555 |
| 3333 | 0.5949 | 0.6780 |
| 三 seed 均值 ± 样本标准差 | 0.5719 ± 0.0371 | 0.7111 ± 0.0400 |

V2 seed1111 的同 seed 比较是清晰提升，但旧版 seed 间波动不小，因此必须等 V2 五 seed 都完成后再比较均值和标准差。只比较 V2 seed1111 与旧版三 seed 均值也不够严格。

## 6. 数据和评估流程审查

目前检查所得证据：

- MOSEI split 数量是 train 16,326、valid 1,871、test 4,659。
- 视频组数是 train 2,249、valid 300、test 676。
- 三个 split 的视频 ID 交集为 0，样本 ID 交集也为 0。
- 精确文本有少量跨 split 重复：train-valid 18、train-test 22、valid-test 7，内容主要是 “hi”“okay” 一类通用短句；没有对应样本或视频重叠，因此不是直接泄漏证据。
- valid 上预测固定 0 或中位数的 MAE 都约为 0.7772；当前 MAE 0.5128、Corr 0.7774，不是常数坍缩。
- 上游 batch 虽包含 `labels_prefix`，但 V2 Router 代码不把它输入模型；`generate()` 也没有接收标签。
- 训练使用标准 causal language-model shift；Qwen 内部将 logits 与 labels 错位一位计算交叉熵。
- 所有目标标签字符串（-3.0 到 +3.0、带符号的一位小数）在当前 tokenizer 中都是 4 token，不存在目标长度不一致导致的左 padding 对齐偏差。
- clean 评估没有训练期模态增强，使用 `model.eval()` 与 `torch.no_grad()`；test 只在阶段训练、校准和最佳 checkpoint 选择之后运行。
- 当前五个被 preflight 锁定的核心源码哈希与 seed1111 manifest 一致。
- 单元测试共 11 项，通过；核心 Python 文件通过 `py_compile`。
- 尚未保存逐样本结果，无法对聚合指标做 artifact-only 独立复算。这是下一版最应优先修补的审计能力。

## 7. 当前风险、建议与审查判定

### 支持继续实验的理由

- clean test 数值合理，且相对同 seed 基线有明显提升。
- split、标签路径和生成调用没有发现明显泄漏。
- invalid generation 为 0，说明解析失败没有人为拉低/抬高汇总。
- seed2222–5555 的阶段一验证最佳 MAE 暂时也在约 0.4995–0.5084 区间，不像只有 seed1111 偶然跑通。

### 不能过度表述的地方

- Router-only 最佳 valid MAE 0.5083，略差于阶段一 0.5072；不能宣称第二阶段路由微调带来增益。
- 平均权重偏文本，且连续噪声/遮挡下平均权重近乎不变；动态噪声感知还未被证明。
- missing_text 严重退化，说明架构仍高度依赖自然文本。
- 当前只有一个 V2 seed 有最终 test；五 seed summary 尚未产生。
- 没有逐样本输出，结果复算链不完整。

### 建议的最低补充实验

1. 完成 V2 五 seed，报告 MAE/Corr 的均值、样本标准差和每 seed 明细。
2. 做 `uniform`、`no_conflict`、`no_uncertainty`、`predictions_only` 消融。
3. 做 text-only、audio-only、vision-only，以及打乱 audio/vision 对应关系的 sanity check。
4. 保存每个样本的 ID、真实标签、预测值、原始生成字符串、三模态权重和 condition。
5. 除平均权重外，报告逐样本权重分布、方差，以及权重变化与噪声强度/错误的相关性。
6. 对阶段一 best checkpoint 与 Router-only best checkpoint 在同一 test 条件下分别评估，直接量化第二阶段的净贡献。

**当前审查判定：可以继续跑完五 seed；不建议现在就把“动态 Router 显著提升”写成最终结论。**

## 8. 集群任务快照

2026-09-04 09:34:29 的 Slurm 状态：

```text
2883058    preflight          COMPLETED  00:01:32  gpu3090n1
2883059_0  seed1111           COMPLETED  08:27:29  gpu3090n1
2883060_1  seed2222           RUNNING    04:57:44  gpu3090n1
2883060_2  seed3333           RUNNING    04:57:44  gpu3090n1
2883060_3  seed4444           RUNNING    04:57:44  gpu3090n2
2883060_4  seed5555           RUNNING    04:57:44  gpu3090n2
2883061    five-seed summary  PENDING    Dependency
```

当时尚在运行的阶段一历史快照（文件会继续变化）：

| seed | 已完成 epoch | 当前最佳 valid MAE | 最佳 epoch | 最后记录 MAE / Corr |
|---:|---:|---:|---:|---:|
| 2222 | 14 | 0.5029 | 9 | 0.5257 / 0.7380 |
| 3333 | 15 | 0.4995 | 13 | 0.5286 / 0.7514 |
| 4444 | 14 | 0.5022 | 14 | 0.5022 / 0.7502 |
| 5555 | 14 | 0.5084 | 11 | 0.5091 / 0.7452 |

汇总作业等待原因是正常的 Slurm dependency：它要等训练 array 成功结束，不能把 `PENDING (Dependency)` 当成故障。

## 9. 复现入口与产物路径

提交入口：

```bash
cd /gpfs/work/cpt/jiachenhou23/MSE-Router
bash scripts/submit_qwen_mosei_router.sh
```

核心产物：

- seed1111：`outputs/qwen-mosei-router-v2/seed_1111/`
- 阶段一 checkpoint：`outputs/qwen-mosei-router-v2/seed_1111/checkpoints/stage1.pt`
- 最终 checkpoint：`outputs/qwen-mosei-router-v2/seed_1111/checkpoints/final.pt`
- 完整运行日志：`outputs/qwen-mosei-router-v2/seed_1111/run.log`
- 五 seed 最终汇总（任务完成后）：`outputs/qwen-mosei-router-v2/five_seed_summary.json`

本文件没有复制以下大体积或第三方内容：

- Qwen 1.8B 权重和其 remote-code 实现：`/gpfs/work/cpt/jiachenhou23/models/Qwen/Qwen-1_8B`
- MOSEI 13.65 GB pickle：`/gpfs/work/cpt/jiachenhou23/datasets/MSA/data/CMU-MOSEI/Processed/unaligned_50.pkl`
- 202 KB 的逐 batch `run.log`（SHA-256：`adbb03db9347ccbd2c022baec5fed9f656c46d1b2e0ffb46d05a0481a936744b`）
- PyTorch、Transformers、SciPy、scikit-learn 等第三方库源码

数据集 SHA-256 已由 preflight 记录为 `ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f`；Qwen 文件的精确大小与 mtime 见下面嵌入的 preflight JSON。

## 10. 完整结果 JSON

下面均为原始产物的逐字嵌入。

### 10.1 V2 preflight 完整结果

路径：`outputs/qwen-mosei-router-v2/preflight/result.json`  
SHA-256：`19eb69ff10a74100fa0847a869baf45897dab5e3b861b4d9a6c52d2e0ce82431`

<!-- BEGIN ARTIFACT: outputs/qwen-mosei-router-v2/preflight/result.json -->

```json
{
  "status": "ok",
  "mode": "preflight",
  "architecture": "natural_text_residual_gated_audio_vision_v2",
  "selected_batching": {
    "microbatch": 4,
    "accumulation": 4,
    "effective_batch": 16
  },
  "attempts": [
    {
      "microbatch": 4,
      "accumulation": 4,
      "status": "ok",
      "batch_size": 4,
      "effective_batch_size": 16,
      "loss": 4.743897914886475,
      "generation_loss": 4.109884262084961,
      "auxiliary_loss": 2.1133782863616943,
      "gradient_tensors": 81,
      "gradient_categories": {
        "adapter": true,
        "ordinal_head": true,
        "router": true
      },
      "loss_scale_attempts": [
        {
          "attempt": 1,
          "scale_before": 65536.0,
          "scale_after": 32768.0,
          "nonfinite_gradient_names": [
            "vision_adapter.token_projector.bias"
          ]
        },
        {
          "attempt": 2,
          "scale_before": 32768.0,
          "scale_after": 32768.0,
          "nonfinite_gradient_names": []
        }
      ],
      "elapsed_seconds": 1.697,
      "peak_gpu_memory_bytes": 4480146944,
      "peak_gpu_reserved_bytes": 4758437888
    }
  ],
  "generation_smoke": {
    "values": [
      0.0
    ],
    "raw_responses": [
      "见证奥哟！"
    ],
    "weights": [
      [
        0.33181777596473694,
        0.3315874934196472,
        0.33659473061561584
      ]
    ],
    "invalid_count": 1,
    "out_of_range_count": 0
  },
  "trainable": {
    "trainable_parameters": 3405935,
    "total_parameters": 1840234610,
    "trainable_tensors": 81,
    "names": [
      "modality_embeddings",
      "text_pool.score.weight",
      "text_pool.score.bias",
      "text_projection.0.weight",
      "text_projection.0.bias",
      "text_adapter.scale1.0.weight",
      "text_adapter.scale1.0.bias",
      "text_adapter.scale1.2.weight",
      "text_adapter.scale1.2.bias",
      "text_adapter.scale2.0.weight",
      "text_adapter.scale2.0.bias",
      "text_adapter.scale2.2.weight",
      "text_adapter.scale2.2.bias",
      "text_adapter.scale3.0.weight",
      "text_adapter.scale3.0.bias",
      "text_adapter.scale3.2.weight",
      "text_adapter.scale3.2.bias",
      "text_adapter.integrating.weight",
      "text_adapter.integrating.bias",
      "text_adapter.multi_scale_projector.weight",
      "text_adapter.multi_scale_projector.bias",
      "text_adapter.token_projector.weight",
      "text_adapter.token_projector.bias",
      "audio_encoder.rnn.weight_ih_l0",
      "audio_encoder.rnn.weight_hh_l0",
      "audio_encoder.rnn.bias_ih_l0",
      "audio_encoder.rnn.bias_hh_l0",
      "audio_encoder.projection.weight",
      "audio_encoder.projection.bias",
      "audio_adapter.scale1.0.weight",
      "audio_adapter.scale1.0.bias",
      "audio_adapter.scale1.2.weight",
      "audio_adapter.scale1.2.bias",
      "audio_adapter.scale2.0.weight",
      "audio_adapter.scale2.0.bias",
      "audio_adapter.scale2.2.weight",
      "audio_adapter.scale2.2.bias",
      "audio_adapter.scale3.0.weight",
      "audio_adapter.scale3.0.bias",
      "audio_adapter.scale3.2.weight",
      "audio_adapter.scale3.2.bias",
      "audio_adapter.integrating.weight",
      "audio_adapter.integrating.bias",
      "audio_adapter.multi_scale_projector.weight",
      "audio_adapter.multi_scale_projector.bias",
      "audio_adapter.token_projector.weight",
      "audio_adapter.token_projector.bias",
      "vision_encoder.rnn.weight_ih_l0",
      "vision_encoder.rnn.weight_hh_l0",
      "vision_encoder.rnn.bias_ih_l0",
      "vision_encoder.rnn.bias_hh_l0",
      "vision_encoder.projection.weight",
      "vision_encoder.projection.bias",
      "vision_adapter.scale1.0.weight",
      "vision_adapter.scale1.0.bias",
      "vision_adapter.scale1.2.weight",
      "vision_adapter.scale1.2.bias",
      "vision_adapter.scale2.0.weight",
      "vision_adapter.scale2.0.bias",
      "vision_adapter.scale2.2.weight",
      "vision_adapter.scale2.2.bias",
      "vision_adapter.scale3.0.weight",
      "vision_adapter.scale3.0.bias",
      "vision_adapter.scale3.2.weight",
      "vision_adapter.scale3.2.bias",
      "vision_adapter.integrating.weight",
      "vision_adapter.integrating.bias",
      "vision_adapter.multi_scale_projector.weight",
      "vision_adapter.multi_scale_projector.bias",
      "vision_adapter.token_projector.weight",
      "vision_adapter.token_projector.bias",
      "ordinal_head.0.weight",
      "ordinal_head.0.bias",
      "ordinal_head.1.weight",
      "ordinal_head.1.bias",
      "ordinal_head.4.weight",
      "ordinal_head.4.bias",
      "router.network.0.weight",
      "router.network.0.bias",
      "router.network.3.weight",
      "router.network.3.bias"
    ]
  },
  "split": {
    "optimization_samples": 14678,
    "calibration_samples": 1648,
    "optimization_groups": 2027,
    "calibration_groups": 222,
    "fingerprint": "f0d8327b684215830a252a322567e55b7cfb379bacccb1469f1c0d706a2aa451",
    "seed": 20260903
  },
  "files": {
    "dataset": {
      "path": "/gpfs/work/cpt/jiachenhou23/datasets/MSA/data/CMU-MOSEI/Processed/unaligned_50.pkl",
      "size": 13652131313,
      "mtime_ns": 1788282495510194736,
      "sha256": "ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f"
    },
    "model": {
      "path": "/gpfs/work/cpt/jiachenhou23/models/Qwen/Qwen-1_8B",
      "files": {
        "config.json": {
          "size": 910,
          "mtime_ns": 1788380112467328000
        },
        "model.safetensors.index.json": {
          "size": 14706,
          "mtime_ns": 1788380180287874000
        },
        "model-00001-of-00002.safetensors": {
          "size": 2039259008,
          "mtime_ns": 1788380146668453224
        },
        "model-00002-of-00002.safetensors": {
          "size": 1634419264,
          "mtime_ns": 1788380175092632374
        },
        "qwen.tiktoken": {
          "size": 2561218,
          "mtime_ns": 1788380182211237000
        }
      }
    }
  },
  "provenance": {
    "files": {
      "mse_router/math_utils.py": "921ab0b3841937dc5b90a5061a38474051caae4a4344004135e1e897f72973cc",
      "mse_router/data.py": "4742d82a8809c26f491fdf8becf6f185216f87e4654f471fa7f07c5cf4e2982e",
      "mse_router/model.py": "0e7dde3741c88833d67443d624d17a0aec043dc659df92d133acde742ef08c86",
      "mse_router/trainer.py": "a09e7dfd81d6e283e53396017fc232b039f349b37d5f3c4a4f9cea7f8baeeb81",
      "scripts/run_qwen_mosei_router.py": "3759ba24294e47fc20a49e2fe4ab63f96baa9b5fb1514e22a72ba53351dae7ea"
    },
    "git_commit": "e0617fec510b2f2c4539e7f8b000efb3dbd98740"
  },
  "environment": {
    "hostname": "gpu3090n1",
    "python": "3.10.12",
    "python_executable": "/gpfs/work/cpt/jiachenhou23/MSE-Adapter/.venv-repro/bin/python",
    "torch": "2.0.1",
    "torch_cuda": "11.7",
    "transformers": "4.36.1",
    "modelscope": "1.10.0",
    "scikit_learn": "1.3.2",
    "scipy": "1.11.4",
    "cuda_visible_devices": "0",
    "gpu": "NVIDIA GeForce RTX 3090",
    "gpu_total_memory_bytes": 25323503616,
    "slurm_job_id": "2883058",
    "slurm_partition": "gpu3090",
    "slurm_qos": "gpudebug"
  },
  "max_rss_bytes": 19583283200
}
```

<!-- END ARTIFACT: outputs/qwen-mosei-router-v2/preflight/result.json -->

### 10.2 V2 seed1111 manifest

路径：`outputs/qwen-mosei-router-v2/seed_1111/manifest.json`  
SHA-256：`24a2690bcb19ac70f188845cea3522fcfdaec3146590342449add5a44a10ad27`

<!-- BEGIN ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/manifest.json -->

```json
{
  "status": "ok",
  "seed": 1111,
  "router_variant": "full",
  "architecture": "natural_text_residual_gated_audio_vision_v2",
  "batching": {
    "microbatch": 4,
    "accumulation": 4,
    "effective_batch": 16
  },
  "split": {
    "optimization_samples": 14678,
    "calibration_samples": 1648,
    "optimization_groups": 2027,
    "calibration_groups": 222,
    "fingerprint": "f0d8327b684215830a252a322567e55b7cfb379bacccb1469f1c0d706a2aa451",
    "seed": 20260903
  },
  "files": {
    "dataset": {
      "path": "/gpfs/work/cpt/jiachenhou23/datasets/MSA/data/CMU-MOSEI/Processed/unaligned_50.pkl",
      "size": 13652131313,
      "mtime_ns": 1788282495510194736,
      "sha256": "ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f"
    },
    "model": {
      "path": "/gpfs/work/cpt/jiachenhou23/models/Qwen/Qwen-1_8B",
      "files": {
        "config.json": {
          "size": 910,
          "mtime_ns": 1788380112467328000
        },
        "model.safetensors.index.json": {
          "size": 14706,
          "mtime_ns": 1788380180287874000
        },
        "model-00001-of-00002.safetensors": {
          "size": 2039259008,
          "mtime_ns": 1788380146668453224
        },
        "model-00002-of-00002.safetensors": {
          "size": 1634419264,
          "mtime_ns": 1788380175092632374
        },
        "qwen.tiktoken": {
          "size": 2561218,
          "mtime_ns": 1788380182211237000
        }
      }
    }
  },
  "provenance": {
    "files": {
      "mse_router/math_utils.py": "921ab0b3841937dc5b90a5061a38474051caae4a4344004135e1e897f72973cc",
      "mse_router/data.py": "4742d82a8809c26f491fdf8becf6f185216f87e4654f471fa7f07c5cf4e2982e",
      "mse_router/model.py": "0e7dde3741c88833d67443d624d17a0aec043dc659df92d133acde742ef08c86",
      "mse_router/trainer.py": "a09e7dfd81d6e283e53396017fc232b039f349b37d5f3c4a4f9cea7f8baeeb81",
      "scripts/run_qwen_mosei_router.py": "3759ba24294e47fc20a49e2fe4ab63f96baa9b5fb1514e22a72ba53351dae7ea"
    },
    "git_commit": "e0617fec510b2f2c4539e7f8b000efb3dbd98740"
  },
  "environment": {
    "hostname": "gpu3090n1",
    "python": "3.10.12",
    "python_executable": "/gpfs/work/cpt/jiachenhou23/MSE-Adapter/.venv-repro/bin/python",
    "torch": "2.0.1",
    "torch_cuda": "11.7",
    "transformers": "4.36.1",
    "modelscope": "1.10.0",
    "scikit_learn": "1.3.2",
    "scipy": "1.11.4",
    "cuda_visible_devices": "0",
    "gpu": "NVIDIA GeForce RTX 3090",
    "gpu_total_memory_bytes": 25323503616,
    "slurm_job_id": "2883059",
    "slurm_partition": "gpu3090",
    "slurm_qos": "8gpus"
  },
  "started_unix": 1788437393.8659687,
  "finished_unix": 1788467804.4546669,
  "result": "/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-router-v2/seed_1111/result.json"
}
```

<!-- END ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/manifest.json -->

### 10.3 V2 seed1111 阶段一历史

路径：`outputs/qwen-mosei-router-v2/seed_1111/stage1_history.json`  
SHA-256：`c8d394d3ad024c2e9dbfc41b3172ef489dd274d3c7abbeb10b805eb51de79c70`

<!-- BEGIN ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/stage1_history.json -->

```json
[
  {
    "epoch": 1,
    "elapsed_seconds": 1207.672,
    "train": {
      "loss": 1.3032223549785666,
      "generation": 0.8211384999735803,
      "auxiliary": 1.6069461160037433
    },
    "valid": {
      "Has0_acc_2": 0.8226,
      "Has0_F1_score": 0.8108,
      "Non0_acc_2": 0.5313,
      "Non0_F1_score": 0.4949,
      "Mult_acc_5": 0.5013,
      "Mult_acc_7": 0.4939,
      "MAE": 0.6283000111579895,
      "Corr": 0.6135,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.3279135823249817,
        0.35911086201667786,
        0.3129754960536957
      ]
    }
  },
  {
    "epoch": 2,
    "elapsed_seconds": 1195.183,
    "train": {
      "loss": 1.036327022768821,
      "generation": 0.5679876130429535,
      "auxiliary": 1.5611313019364017
    },
    "valid": {
      "Has0_acc_2": 0.8295,
      "Has0_F1_score": 0.8134,
      "Non0_acc_2": 0.678,
      "Non0_F1_score": 0.6787,
      "Mult_acc_5": 0.5345,
      "Mult_acc_7": 0.5259,
      "MAE": 0.5565000176429749,
      "Corr": 0.6772,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.35645532608032227,
        0.3773873746395111,
        0.2661575376987457
      ]
    }
  },
  {
    "epoch": 3,
    "elapsed_seconds": 1193.401,
    "train": {
      "loss": 1.0203264414776898,
      "generation": 0.552490392083693,
      "auxiliary": 1.5594534391440877
    },
    "valid": {
      "Has0_acc_2": 0.853,
      "Has0_F1_score": 0.846,
      "Non0_acc_2": 0.573,
      "Non0_F1_score": 0.5547,
      "Mult_acc_5": 0.5275,
      "Mult_acc_7": 0.5152,
      "MAE": 0.579200029373169,
      "Corr": 0.6931,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.3719116747379303,
        0.38247641921043396,
        0.24561218917369843
      ]
    }
  },
  {
    "epoch": 4,
    "elapsed_seconds": 1194.162,
    "train": {
      "loss": 1.0136460820887978,
      "generation": 0.5465428640634552,
      "auxiliary": 1.5570106606392509
    },
    "valid": {
      "Has0_acc_2": 0.8487,
      "Has0_F1_score": 0.8479,
      "Non0_acc_2": 0.6697,
      "Non0_F1_score": 0.6708,
      "Mult_acc_5": 0.5425,
      "Mult_acc_7": 0.5318,
      "MAE": 0.5491999983787537,
      "Corr": 0.7141,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.41579148173332214,
        0.3529435694217682,
        0.23126472532749176
      ]
    }
  },
  {
    "epoch": 5,
    "elapsed_seconds": 1194.594,
    "train": {
      "loss": 1.0037938958779993,
      "generation": 0.5371668060972515,
      "auxiliary": 1.5554235729112287
    },
    "valid": {
      "Has0_acc_2": 0.8541,
      "Has0_F1_score": 0.8474,
      "Non0_acc_2": 0.6314,
      "Non0_F1_score": 0.6248,
      "Mult_acc_5": 0.5452,
      "Mult_acc_7": 0.5238,
      "MAE": 0.5776000022888184,
      "Corr": 0.7061,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.409064918756485,
        0.35280880331993103,
        0.23812678456306458
      ]
    }
  },
  {
    "epoch": 6,
    "elapsed_seconds": 1192.322,
    "train": {
      "loss": 0.9972809521151499,
      "generation": 0.5306594065651906,
      "auxiliary": 1.5554050876918866
    },
    "valid": {
      "Has0_acc_2": 0.8632,
      "Has0_F1_score": 0.8594,
      "Non0_acc_2": 0.8345,
      "Non0_F1_score": 0.8366,
      "Mult_acc_5": 0.5591,
      "Mult_acc_7": 0.5425,
      "MAE": 0.5254999995231628,
      "Corr": 0.7466,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.42726171016693115,
        0.34028610587120056,
        0.2324521243572235
      ]
    }
  },
  {
    "epoch": 7,
    "elapsed_seconds": 1188.074,
    "train": {
      "loss": 0.9930360675020504,
      "generation": 0.527275339016635,
      "auxiliary": 1.55253570087924
    },
    "valid": {
      "Has0_acc_2": 0.8653,
      "Has0_F1_score": 0.8607,
      "Non0_acc_2": 0.6725,
      "Non0_F1_score": 0.6731,
      "Mult_acc_5": 0.5575,
      "Mult_acc_7": 0.5393,
      "MAE": 0.5241000056266785,
      "Corr": 0.7338,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.4427087903022766,
        0.3367369472980499,
        0.22055403888225555
      ]
    }
  },
  {
    "epoch": 8,
    "elapsed_seconds": 1188.04,
    "train": {
      "loss": 0.9879785862215859,
      "generation": 0.5220951926440244,
      "auxiliary": 1.552944584584691
    },
    "valid": {
      "Has0_acc_2": 0.8493,
      "Has0_F1_score": 0.8362,
      "Non0_acc_2": 0.6113,
      "Non0_F1_score": 0.601,
      "Mult_acc_5": 0.5756,
      "Mult_acc_7": 0.5612,
      "MAE": 0.5320000052452087,
      "Corr": 0.7124,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.4319245219230652,
        0.32866594195365906,
        0.23940981924533844
      ]
    }
  },
  {
    "epoch": 9,
    "elapsed_seconds": 1188.339,
    "train": {
      "loss": 0.9849580114803782,
      "generation": 0.5197286282475703,
      "auxiliary": 1.5507645483887489
    },
    "valid": {
      "Has0_acc_2": 0.8632,
      "Has0_F1_score": 0.8563,
      "Non0_acc_2": 0.726,
      "Non0_F1_score": 0.731,
      "Mult_acc_5": 0.5703,
      "Mult_acc_7": 0.5548,
      "MAE": 0.5072000026702881,
      "Corr": 0.7355,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.42517775297164917,
        0.31495150923728943,
        0.2598707973957062
      ]
    }
  },
  {
    "epoch": 10,
    "elapsed_seconds": 1202.811,
    "train": {
      "loss": 0.9809709739214069,
      "generation": 0.5157564894704793,
      "auxiliary": 1.5507148855713473
    },
    "valid": {
      "Has0_acc_2": 0.86,
      "Has0_F1_score": 0.8524,
      "Non0_acc_2": 0.8428,
      "Non0_F1_score": 0.8449,
      "Mult_acc_5": 0.5569,
      "Mult_acc_7": 0.5409,
      "MAE": 0.5087000131607056,
      "Corr": 0.7483,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.4196292459964752,
        0.3079070746898651,
        0.2724635601043701
      ]
    }
  },
  {
    "epoch": 11,
    "elapsed_seconds": 1207.32,
    "train": {
      "loss": 0.9776480354064166,
      "generation": 0.513586049126994,
      "auxiliary": 1.5468732251783157
    },
    "valid": {
      "Has0_acc_2": 0.8642,
      "Has0_F1_score": 0.862,
      "Non0_acc_2": 0.6982,
      "Non0_F1_score": 0.7005,
      "Mult_acc_5": 0.5665,
      "Mult_acc_7": 0.5484,
      "MAE": 0.515500009059906,
      "Corr": 0.7383,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.42593008279800415,
        0.3067668676376343,
        0.26730263233184814
      ]
    }
  },
  {
    "epoch": 12,
    "elapsed_seconds": 1209.41,
    "train": {
      "loss": 0.9727696291107573,
      "generation": 0.5097445108625804,
      "auxiliary": 1.5434170001858911
    },
    "valid": {
      "Has0_acc_2": 0.8621,
      "Has0_F1_score": 0.8584,
      "Non0_acc_2": 0.7156,
      "Non0_F1_score": 0.7191,
      "Mult_acc_5": 0.5687,
      "Mult_acc_7": 0.5553,
      "MAE": 0.5188000202178955,
      "Corr": 0.7235,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.4131513833999634,
        0.31640008091926575,
        0.2704484164714813
      ]
    }
  },
  {
    "epoch": 13,
    "elapsed_seconds": 1208.789,
    "train": {
      "loss": 0.9712469962539725,
      "generation": 0.5079315917449686,
      "auxiliary": 1.5443846193743662
    },
    "valid": {
      "Has0_acc_2": 0.852,
      "Has0_F1_score": 0.8543,
      "Non0_acc_2": 0.685,
      "Non0_F1_score": 0.6859,
      "Mult_acc_5": 0.5681,
      "Mult_acc_7": 0.5505,
      "MAE": 0.5199000239372253,
      "Corr": 0.7377,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.4108422100543976,
        0.30110490322113037,
        0.28805291652679443
      ]
    }
  },
  {
    "epoch": 14,
    "elapsed_seconds": 1208.741,
    "train": {
      "loss": 0.9666716298550286,
      "generation": 0.504585767267348,
      "auxiliary": 1.5402861459865882
    },
    "valid": {
      "Has0_acc_2": 0.8632,
      "Has0_F1_score": 0.8618,
      "Non0_acc_2": 0.735,
      "Non0_F1_score": 0.7394,
      "Mult_acc_5": 0.5665,
      "Mult_acc_7": 0.5526,
      "MAE": 0.5214999914169312,
      "Corr": 0.7359,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.38926276564598083,
        0.3235550820827484,
        0.28718239068984985
      ]
    }
  },
  {
    "epoch": 15,
    "elapsed_seconds": 1207.64,
    "train": {
      "loss": 0.9633748888725806,
      "generation": 0.5017554766110244,
      "auxiliary": 1.5387313116473789
    },
    "valid": {
      "Has0_acc_2": 0.8594,
      "Has0_F1_score": 0.8568,
      "Non0_acc_2": 0.7531,
      "Non0_F1_score": 0.758,
      "Mult_acc_5": 0.5671,
      "Mult_acc_7": 0.5404,
      "MAE": 0.5389000177383423,
      "Corr": 0.7418,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.40507638454437256,
        0.3087596297264099,
        0.28616392612457275
      ]
    }
  },
  {
    "epoch": 16,
    "elapsed_seconds": 1207.321,
    "train": {
      "loss": 0.9611244640574468,
      "generation": 0.49962532149592276,
      "auxiliary": 1.5383304110664762
    },
    "valid": {
      "Has0_acc_2": 0.86,
      "Has0_F1_score": 0.8516,
      "Non0_acc_2": 0.6725,
      "Non0_F1_score": 0.6723,
      "Mult_acc_5": 0.5703,
      "Mult_acc_7": 0.5548,
      "MAE": 0.5192000269889832,
      "Corr": 0.7154,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.37575867772102356,
        0.31719356775283813,
        0.3070475459098816
      ]
    }
  },
  {
    "epoch": 17,
    "elapsed_seconds": 1207.472,
    "train": {
      "loss": 0.9559285185642399,
      "generation": 0.4956045611840178,
      "auxiliary": 1.5344131310239475
    },
    "valid": {
      "Has0_acc_2": 0.8594,
      "Has0_F1_score": 0.8536,
      "Non0_acc_2": 0.6412,
      "Non0_F1_score": 0.6368,
      "Mult_acc_5": 0.5542,
      "Mult_acc_7": 0.5425,
      "MAE": 0.5311999917030334,
      "Corr": 0.7193,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.37227004766464233,
        0.315396785736084,
        0.3123331367969513
      ]
    }
  },
  {
    "epoch": 18,
    "elapsed_seconds": 1208.046,
    "train": {
      "loss": 0.9535638366635554,
      "generation": 0.49344551796728,
      "auxiliary": 1.5337276670523496
    },
    "valid": {
      "Has0_acc_2": 0.86,
      "Has0_F1_score": 0.8581,
      "Non0_acc_2": 0.735,
      "Non0_F1_score": 0.7396,
      "Mult_acc_5": 0.5676,
      "Mult_acc_7": 0.5478,
      "MAE": 0.5285999774932861,
      "Corr": 0.7308,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.35491621494293213,
        0.320204496383667,
        0.32487910985946655
      ]
    }
  },
  {
    "epoch": 19,
    "elapsed_seconds": 1183.442,
    "train": {
      "loss": 0.9489075007571837,
      "generation": 0.4893813996086004,
      "auxiliary": 1.5317536086574888
    },
    "valid": {
      "Has0_acc_2": 0.8621,
      "Has0_F1_score": 0.8589,
      "Non0_acc_2": 0.749,
      "Non0_F1_score": 0.7538,
      "Mult_acc_5": 0.55,
      "Mult_acc_7": 0.5393,
      "MAE": 0.5098999738693237,
      "Corr": 0.7362,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.3607591390609741,
        0.32262274622917175,
        0.3166183829307556
      ]
    }
  }
]
```

<!-- END ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/stage1_history.json -->

### 10.4 V2 seed1111 温度校准

路径：`outputs/qwen-mosei-router-v2/seed_1111/calibration.json`  
SHA-256：`cb3ffd00fb9dfd7a081435e98ab5e683fade534be6764e4fc2335a45e7089c06`

<!-- BEGIN ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/calibration.json -->

```json
{
  "valid": true,
  "samples": 1648,
  "temperatures": [
    1.1071734577318335,
    1.1668446795297556,
    1.1151741602476124
  ],
  "modalities": [
    {
      "modality": "text",
      "temperature": 1.1071734577318335,
      "tau1_nll": 1.5713365077972412,
      "calibrated_nll": 1.5682612657546997,
      "optimizer_success": true,
      "at_bound": false,
      "valid": true,
      "message": "Solution found."
    },
    {
      "modality": "audio",
      "temperature": 1.1668446795297556,
      "tau1_nll": 1.5747593641281128,
      "calibrated_nll": 1.5675688982009888,
      "optimizer_success": true,
      "at_bound": false,
      "valid": true,
      "message": "Solution found."
    },
    {
      "modality": "vision",
      "temperature": 1.1151741602476124,
      "tau1_nll": 1.547956109046936,
      "calibrated_nll": 1.5443017482757568,
      "optimizer_success": true,
      "at_bound": false,
      "valid": true,
      "message": "Solution found."
    }
  ]
}
```

<!-- END ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/calibration.json -->

### 10.5 V2 seed1111 Router-only 历史

路径：`outputs/qwen-mosei-router-v2/seed_1111/router_history.json`  
SHA-256：`fcc8ba61984a581de36254eb32464a91852ee0171fca7baced9c3989434a4165`

<!-- BEGIN ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/router_history.json -->

```json
[
  {
    "epoch": 1,
    "elapsed_seconds": 633.108,
    "train": {
      "loss": 0.5107663321357008,
      "generation": 0.5107663321357008,
      "auxiliary": 1.5441185768521124
    },
    "valid": {
      "Has0_acc_2": 0.8637,
      "Has0_F1_score": 0.8581,
      "Non0_acc_2": 0.7274,
      "Non0_F1_score": 0.7323,
      "Mult_acc_5": 0.5714,
      "Mult_acc_7": 0.5532,
      "MAE": 0.5098000168800354,
      "Corr": 0.7366,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.47623971104621887,
        0.2458207607269287,
        0.27793940901756287
      ]
    }
  },
  {
    "epoch": 2,
    "elapsed_seconds": 634.073,
    "train": {
      "loss": 0.5105705065282229,
      "generation": 0.5105705065282229,
      "auxiliary": 1.5447494907177761
    },
    "valid": {
      "Has0_acc_2": 0.8648,
      "Has0_F1_score": 0.8592,
      "Non0_acc_2": 0.7177,
      "Non0_F1_score": 0.7221,
      "Mult_acc_5": 0.5719,
      "Mult_acc_7": 0.5542,
      "MAE": 0.5088000297546387,
      "Corr": 0.7366,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.5041369199752808,
        0.24289050698280334,
        0.2529726028442383
      ]
    }
  },
  {
    "epoch": 3,
    "elapsed_seconds": 634.161,
    "train": {
      "loss": 0.5108126615349213,
      "generation": 0.5108126615349213,
      "auxiliary": 1.544699405145905
    },
    "valid": {
      "Has0_acc_2": 0.8642,
      "Has0_F1_score": 0.8593,
      "Non0_acc_2": 0.7086,
      "Non0_F1_score": 0.7126,
      "Mult_acc_5": 0.5676,
      "Mult_acc_7": 0.55,
      "MAE": 0.5145999789237976,
      "Corr": 0.7342,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.5536355972290039,
        0.20867900550365448,
        0.23768575489521027
      ]
    }
  },
  {
    "epoch": 4,
    "elapsed_seconds": 631.989,
    "train": {
      "loss": 0.5106781062703042,
      "generation": 0.5106781062703042,
      "auxiliary": 1.5424348361972893
    },
    "valid": {
      "Has0_acc_2": 0.8653,
      "Has0_F1_score": 0.8601,
      "Non0_acc_2": 0.7253,
      "Non0_F1_score": 0.73,
      "Mult_acc_5": 0.5697,
      "Mult_acc_7": 0.5516,
      "MAE": 0.510699987411499,
      "Corr": 0.7371,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.4958999454975128,
        0.235567107796669,
        0.268532931804657
      ]
    }
  },
  {
    "epoch": 5,
    "elapsed_seconds": 631.668,
    "train": {
      "loss": 0.5102323932771137,
      "generation": 0.5102323932771137,
      "auxiliary": 1.5439174972210659
    },
    "valid": {
      "Has0_acc_2": 0.8642,
      "Has0_F1_score": 0.859,
      "Non0_acc_2": 0.7239,
      "Non0_F1_score": 0.7286,
      "Mult_acc_5": 0.5708,
      "Mult_acc_7": 0.5532,
      "MAE": 0.5083000063896179,
      "Corr": 0.7384,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.4949560761451721,
        0.23929601907730103,
        0.26574844121932983
      ]
    }
  },
  {
    "epoch": 6,
    "elapsed_seconds": 630.609,
    "train": {
      "loss": 0.5100409037410726,
      "generation": 0.5100409037410726,
      "auxiliary": 1.5422075311072192
    },
    "valid": {
      "Has0_acc_2": 0.8642,
      "Has0_F1_score": 0.8591,
      "Non0_acc_2": 0.7149,
      "Non0_F1_score": 0.7192,
      "Mult_acc_5": 0.5724,
      "Mult_acc_7": 0.5548,
      "MAE": 0.510200023651123,
      "Corr": 0.7374,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.5187897086143494,
        0.22924913465976715,
        0.2519606947898865
      ]
    }
  },
  {
    "epoch": 7,
    "elapsed_seconds": 634.343,
    "train": {
      "loss": 0.510736440812372,
      "generation": 0.510736440812372,
      "auxiliary": 1.5433198054090183
    },
    "valid": {
      "Has0_acc_2": 0.8648,
      "Has0_F1_score": 0.8595,
      "Non0_acc_2": 0.7239,
      "Non0_F1_score": 0.7286,
      "Mult_acc_5": 0.5697,
      "Mult_acc_7": 0.5516,
      "MAE": 0.5091999769210815,
      "Corr": 0.7376,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.491777628660202,
        0.24303165078163147,
        0.26519089937210083
      ]
    }
  },
  {
    "epoch": 8,
    "elapsed_seconds": 633.878,
    "train": {
      "loss": 0.5093501533125336,
      "generation": 0.5093501533125336,
      "auxiliary": 1.544469125589168
    },
    "valid": {
      "Has0_acc_2": 0.8648,
      "Has0_F1_score": 0.8596,
      "Non0_acc_2": 0.7225,
      "Non0_F1_score": 0.7271,
      "Mult_acc_5": 0.5708,
      "Mult_acc_7": 0.5526,
      "MAE": 0.5105999708175659,
      "Corr": 0.7362,
      "condition": "clean",
      "samples": 1871,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.5042438507080078,
        0.23306064307689667,
        0.26269540190696716
      ]
    }
  }
]
```

<!-- END ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/router_history.json -->

### 10.6 V2 seed1111 鲁棒性结果

路径：`outputs/qwen-mosei-router-v2/seed_1111/robustness.json`  
SHA-256：`95c4884207244e7d48385808d794c65d943fd93e32a5017578c1eed5512f2784`

<!-- BEGIN ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/robustness.json -->

```json
{
  "clean": {
    "Has0_acc_2": 0.8566,
    "Has0_F1_score": 0.8514,
    "Non0_acc_2": 0.7617,
    "Non0_F1_score": 0.7646,
    "Mult_acc_5": 0.5697,
    "Mult_acc_7": 0.5503,
    "MAE": 0.5127999782562256,
    "Corr": 0.7774,
    "condition": "clean",
    "samples": 4659,
    "invalid_generations": 0,
    "out_of_range_generations": 0,
    "mean_router_weights": [
      0.4950696527957916,
      0.23926040530204773,
      0.26566997170448303
    ]
  },
  "missing_text": {
    "Has0_acc_2": 0.7102,
    "Has0_F1_score": 0.5899,
    "Non0_acc_2": 0.5173,
    "Non0_F1_score": 0.5058,
    "Mult_acc_5": 0.4209,
    "Mult_acc_7": 0.4209,
    "MAE": 0.8144999742507935,
    "Corr": 0.1714,
    "condition": "missing_text",
    "samples": 4659,
    "invalid_generations": 0,
    "out_of_range_generations": 0,
    "mean_router_weights": [
      0.0,
      0.4745136797428131,
      0.5254865288734436
    ]
  },
  "missing_audio": {
    "Has0_acc_2": 0.853,
    "Has0_F1_score": 0.8507,
    "Non0_acc_2": 0.7369,
    "Non0_F1_score": 0.7396,
    "Mult_acc_5": 0.5544,
    "Mult_acc_7": 0.5289,
    "MAE": 0.5507000088691711,
    "Corr": 0.7604,
    "condition": "missing_audio",
    "samples": 4659,
    "invalid_generations": 0,
    "out_of_range_generations": 0,
    "mean_router_weights": [
      0.5967596769332886,
      0.0,
      0.40324103832244873
    ]
  },
  "missing_vision": {
    "Has0_acc_2": 0.8433,
    "Has0_F1_score": 0.8319,
    "Non0_acc_2": 0.6293,
    "Non0_F1_score": 0.6175,
    "Mult_acc_5": 0.4969,
    "Mult_acc_7": 0.4834,
    "MAE": 0.5993000268936157,
    "Corr": 0.7,
    "condition": "missing_vision",
    "samples": 4659,
    "invalid_generations": 0,
    "out_of_range_generations": 1,
    "mean_router_weights": [
      0.5275357365608215,
      0.4724627435207367,
      0.0
    ]
  },
  "audio_snr_20": {
    "Has0_acc_2": 0.8568,
    "Has0_F1_score": 0.8516,
    "Non0_acc_2": 0.7628,
    "Non0_F1_score": 0.7658,
    "Mult_acc_5": 0.5699,
    "Mult_acc_7": 0.5501,
    "MAE": 0.5121999979019165,
    "Corr": 0.7777,
    "condition": "audio_snr_20",
    "samples": 4659,
    "invalid_generations": 0,
    "out_of_range_generations": 0,
    "mean_router_weights": [
      0.4950708746910095,
      0.23925988376140594,
      0.2656693160533905
    ]
  },
  "audio_snr_10": {
    "Has0_acc_2": 0.8568,
    "Has0_F1_score": 0.8514,
    "Non0_acc_2": 0.7565,
    "Non0_F1_score": 0.7593,
    "Mult_acc_5": 0.5697,
    "Mult_acc_7": 0.5501,
    "MAE": 0.5131000280380249,
    "Corr": 0.777,
    "condition": "audio_snr_10",
    "samples": 4659,
    "invalid_generations": 0,
    "out_of_range_generations": 0,
    "mean_router_weights": [
      0.49504995346069336,
      0.23926745355129242,
      0.2656826674938202
    ]
  },
  "audio_snr_0": {
    "Has0_acc_2": 0.856,
    "Has0_F1_score": 0.8504,
    "Non0_acc_2": 0.7521,
    "Non0_F1_score": 0.7548,
    "Mult_acc_5": 0.5709,
    "Mult_acc_7": 0.5514,
    "MAE": 0.5133000016212463,
    "Corr": 0.7766,
    "condition": "audio_snr_0",
    "samples": 4659,
    "invalid_generations": 0,
    "out_of_range_generations": 0,
    "mean_router_weights": [
      0.49504026770591736,
      0.2392713576555252,
      0.2656882107257843
    ]
  },
  "vision_mask_25": {
    "Has0_acc_2": 0.8562,
    "Has0_F1_score": 0.8509,
    "Non0_acc_2": 0.7504,
    "Non0_F1_score": 0.7529,
    "Mult_acc_5": 0.5684,
    "Mult_acc_7": 0.5488,
    "MAE": 0.5152000188827515,
    "Corr": 0.7768,
    "condition": "vision_mask_25",
    "samples": 4659,
    "invalid_generations": 0,
    "out_of_range_generations": 0,
    "mean_router_weights": [
      0.4949548542499542,
      0.23929713666439056,
      0.26574766635894775
    ]
  },
  "vision_mask_50": {
    "Has0_acc_2": 0.8562,
    "Has0_F1_score": 0.851,
    "Non0_acc_2": 0.7413,
    "Non0_F1_score": 0.7435,
    "Mult_acc_5": 0.5669,
    "Mult_acc_7": 0.5478,
    "MAE": 0.5180000066757202,
    "Corr": 0.7747,
    "condition": "vision_mask_50",
    "samples": 4659,
    "invalid_generations": 0,
    "out_of_range_generations": 0,
    "mean_router_weights": [
      0.49487215280532837,
      0.2393246442079544,
      0.26580309867858887
    ]
  },
  "vision_mask_75": {
    "Has0_acc_2": 0.8577,
    "Has0_F1_score": 0.8526,
    "Non0_acc_2": 0.7223,
    "Non0_F1_score": 0.7232,
    "Mult_acc_5": 0.5673,
    "Mult_acc_7": 0.5482,
    "MAE": 0.5212000012397766,
    "Corr": 0.7737,
    "condition": "vision_mask_75",
    "samples": 4659,
    "invalid_generations": 0,
    "out_of_range_generations": 0,
    "mean_router_weights": [
      0.4947585463523865,
      0.23936185240745544,
      0.2658792734146118
    ]
  }
}
```

<!-- END ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/robustness.json -->

### 10.7 V2 seed1111 最终汇总

路径：`outputs/qwen-mosei-router-v2/seed_1111/result.json`  
SHA-256：`4bd4e6db5467a76fd96f55b1e1c15d76cc09a729573df61cc462e508d3223582`

<!-- BEGIN ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/result.json -->

```json
{
  "status": "ok",
  "stage1": {
    "stage": "stage1",
    "best_epoch": 9,
    "best_valid_mae": 0.5072000026702881,
    "epochs_ran": 19,
    "checkpoint": "/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-router-v2/seed_1111/checkpoints/stage1.pt"
  },
  "calibration": {
    "valid": true,
    "samples": 1648,
    "temperatures": [
      1.1071734577318335,
      1.1668446795297556,
      1.1151741602476124
    ],
    "modalities": [
      {
        "modality": "text",
        "temperature": 1.1071734577318335,
        "tau1_nll": 1.5713365077972412,
        "calibrated_nll": 1.5682612657546997,
        "optimizer_success": true,
        "at_bound": false,
        "valid": true,
        "message": "Solution found."
      },
      {
        "modality": "audio",
        "temperature": 1.1668446795297556,
        "tau1_nll": 1.5747593641281128,
        "calibrated_nll": 1.5675688982009888,
        "optimizer_success": true,
        "at_bound": false,
        "valid": true,
        "message": "Solution found."
      },
      {
        "modality": "vision",
        "temperature": 1.1151741602476124,
        "tau1_nll": 1.547956109046936,
        "calibrated_nll": 1.5443017482757568,
        "optimizer_success": true,
        "at_bound": false,
        "valid": true,
        "message": "Solution found."
      }
    ]
  },
  "router_stage": {
    "stage": "router",
    "best_epoch": 5,
    "best_valid_mae": 0.5083000063896179,
    "epochs_ran": 8,
    "checkpoint": "/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-router-v2/seed_1111/checkpoints/final.pt"
  },
  "test": {
    "Has0_acc_2": 0.8566,
    "Has0_F1_score": 0.8514,
    "Non0_acc_2": 0.7617,
    "Non0_F1_score": 0.7646,
    "Mult_acc_5": 0.5697,
    "Mult_acc_7": 0.5503,
    "MAE": 0.5127999782562256,
    "Corr": 0.7774,
    "condition": "clean",
    "samples": 4659,
    "invalid_generations": 0,
    "out_of_range_generations": 0,
    "mean_router_weights": [
      0.4950696527957916,
      0.23926040530204773,
      0.26566997170448303
    ]
  },
  "robustness": {
    "clean": {
      "Has0_acc_2": 0.8566,
      "Has0_F1_score": 0.8514,
      "Non0_acc_2": 0.7617,
      "Non0_F1_score": 0.7646,
      "Mult_acc_5": 0.5697,
      "Mult_acc_7": 0.5503,
      "MAE": 0.5127999782562256,
      "Corr": 0.7774,
      "condition": "clean",
      "samples": 4659,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.4950696527957916,
        0.23926040530204773,
        0.26566997170448303
      ]
    },
    "missing_text": {
      "Has0_acc_2": 0.7102,
      "Has0_F1_score": 0.5899,
      "Non0_acc_2": 0.5173,
      "Non0_F1_score": 0.5058,
      "Mult_acc_5": 0.4209,
      "Mult_acc_7": 0.4209,
      "MAE": 0.8144999742507935,
      "Corr": 0.1714,
      "condition": "missing_text",
      "samples": 4659,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.0,
        0.4745136797428131,
        0.5254865288734436
      ]
    },
    "missing_audio": {
      "Has0_acc_2": 0.853,
      "Has0_F1_score": 0.8507,
      "Non0_acc_2": 0.7369,
      "Non0_F1_score": 0.7396,
      "Mult_acc_5": 0.5544,
      "Mult_acc_7": 0.5289,
      "MAE": 0.5507000088691711,
      "Corr": 0.7604,
      "condition": "missing_audio",
      "samples": 4659,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.5967596769332886,
        0.0,
        0.40324103832244873
      ]
    },
    "missing_vision": {
      "Has0_acc_2": 0.8433,
      "Has0_F1_score": 0.8319,
      "Non0_acc_2": 0.6293,
      "Non0_F1_score": 0.6175,
      "Mult_acc_5": 0.4969,
      "Mult_acc_7": 0.4834,
      "MAE": 0.5993000268936157,
      "Corr": 0.7,
      "condition": "missing_vision",
      "samples": 4659,
      "invalid_generations": 0,
      "out_of_range_generations": 1,
      "mean_router_weights": [
        0.5275357365608215,
        0.4724627435207367,
        0.0
      ]
    },
    "audio_snr_20": {
      "Has0_acc_2": 0.8568,
      "Has0_F1_score": 0.8516,
      "Non0_acc_2": 0.7628,
      "Non0_F1_score": 0.7658,
      "Mult_acc_5": 0.5699,
      "Mult_acc_7": 0.5501,
      "MAE": 0.5121999979019165,
      "Corr": 0.7777,
      "condition": "audio_snr_20",
      "samples": 4659,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.4950708746910095,
        0.23925988376140594,
        0.2656693160533905
      ]
    },
    "audio_snr_10": {
      "Has0_acc_2": 0.8568,
      "Has0_F1_score": 0.8514,
      "Non0_acc_2": 0.7565,
      "Non0_F1_score": 0.7593,
      "Mult_acc_5": 0.5697,
      "Mult_acc_7": 0.5501,
      "MAE": 0.5131000280380249,
      "Corr": 0.777,
      "condition": "audio_snr_10",
      "samples": 4659,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.49504995346069336,
        0.23926745355129242,
        0.2656826674938202
      ]
    },
    "audio_snr_0": {
      "Has0_acc_2": 0.856,
      "Has0_F1_score": 0.8504,
      "Non0_acc_2": 0.7521,
      "Non0_F1_score": 0.7548,
      "Mult_acc_5": 0.5709,
      "Mult_acc_7": 0.5514,
      "MAE": 0.5133000016212463,
      "Corr": 0.7766,
      "condition": "audio_snr_0",
      "samples": 4659,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.49504026770591736,
        0.2392713576555252,
        0.2656882107257843
      ]
    },
    "vision_mask_25": {
      "Has0_acc_2": 0.8562,
      "Has0_F1_score": 0.8509,
      "Non0_acc_2": 0.7504,
      "Non0_F1_score": 0.7529,
      "Mult_acc_5": 0.5684,
      "Mult_acc_7": 0.5488,
      "MAE": 0.5152000188827515,
      "Corr": 0.7768,
      "condition": "vision_mask_25",
      "samples": 4659,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.4949548542499542,
        0.23929713666439056,
        0.26574766635894775
      ]
    },
    "vision_mask_50": {
      "Has0_acc_2": 0.8562,
      "Has0_F1_score": 0.851,
      "Non0_acc_2": 0.7413,
      "Non0_F1_score": 0.7435,
      "Mult_acc_5": 0.5669,
      "Mult_acc_7": 0.5478,
      "MAE": 0.5180000066757202,
      "Corr": 0.7747,
      "condition": "vision_mask_50",
      "samples": 4659,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.49487215280532837,
        0.2393246442079544,
        0.26580309867858887
      ]
    },
    "vision_mask_75": {
      "Has0_acc_2": 0.8577,
      "Has0_F1_score": 0.8526,
      "Non0_acc_2": 0.7223,
      "Non0_F1_score": 0.7232,
      "Mult_acc_5": 0.5673,
      "Mult_acc_7": 0.5482,
      "MAE": 0.5212000012397766,
      "Corr": 0.7737,
      "condition": "vision_mask_75",
      "samples": 4659,
      "invalid_generations": 0,
      "out_of_range_generations": 0,
      "mean_router_weights": [
        0.4947585463523865,
        0.23936185240745544,
        0.2658792734146118
      ]
    }
  },
  "temperatures": [
    1.1071734428405762,
    1.1668447256088257,
    1.1151741743087769
  ],
  "elapsed_seconds": 30404.75,
  "final_checkpoint": "/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-router-v2/seed_1111/checkpoints/final.pt",
  "training_settings": {
    "adapter_lr": 0.005,
    "head_router_lr": 0.001,
    "weight_decay": 0.01,
    "adam_epsilon": 0.0001,
    "warmup_fraction": 0.1,
    "gradient_clip": 1.0,
    "accumulation_steps": 4,
    "stage1_max_epochs": 40,
    "stage1_patience": 10,
    "router_max_epochs": 10,
    "router_patience": 3,
    "progress_interval": 100,
    "quality_gate_epoch": 4,
    "quality_gate_max_mae": 0.72,
    "quality_gate_min_corr": 0.3
  },
  "seed": 1111,
  "router_variant": "full",
  "peak_gpu_memory_bytes": 5184145408,
  "peak_gpu_reserved_bytes": 5595201536,
  "max_rss_bytes": 19585847296,
  "environment": {
    "hostname": "gpu3090n1",
    "python": "3.10.12",
    "python_executable": "/gpfs/work/cpt/jiachenhou23/MSE-Adapter/.venv-repro/bin/python",
    "torch": "2.0.1",
    "torch_cuda": "11.7",
    "transformers": "4.36.1",
    "modelscope": "1.10.0",
    "scikit_learn": "1.3.2",
    "scipy": "1.11.4",
    "cuda_visible_devices": "0",
    "gpu": "NVIDIA GeForce RTX 3090",
    "gpu_total_memory_bytes": 25323503616,
    "slurm_job_id": "2883059",
    "slurm_partition": "gpu3090",
    "slurm_qos": "8gpus"
  },
  "provenance": {
    "files": {
      "mse_router/math_utils.py": "921ab0b3841937dc5b90a5061a38474051caae4a4344004135e1e897f72973cc",
      "mse_router/data.py": "4742d82a8809c26f491fdf8becf6f185216f87e4654f471fa7f07c5cf4e2982e",
      "mse_router/model.py": "0e7dde3741c88833d67443d624d17a0aec043dc659df92d133acde742ef08c86",
      "mse_router/trainer.py": "a09e7dfd81d6e283e53396017fc232b039f349b37d5f3c4a4f9cea7f8baeeb81",
      "scripts/run_qwen_mosei_router.py": "3759ba24294e47fc20a49e2fe4ab63f96baa9b5fb1514e22a72ba53351dae7ea"
    },
    "git_commit": "e0617fec510b2f2c4539e7f8b000efb3dbd98740"
  }
}
```

<!-- END ARTIFACT: outputs/qwen-mosei-router-v2/seed_1111/result.json -->

### 10.8 旧版基线 seed1111 最终结果

路径：`outputs/qwen-mosei-repro/seed_1111/result.json`  
SHA-256：`51367a045b8e4acfd7df0bbbbef9e3219c291b65f54f2b60b5f4bb41ab8aae2d`

<!-- BEGIN ARTIFACT: outputs/qwen-mosei-repro/seed_1111/result.json -->

```json
{
  "status": "ok",
  "seed": 1111,
  "metrics": {
    "Has0_acc_2": 0.8332,
    "Has0_F1_score": 0.828,
    "Non0_acc_2": 0.7067,
    "Non0_F1_score": 0.7085,
    "Mult_acc_5": 0.5289,
    "Mult_acc_7": 0.5098,
    "MAE": 0.5917999744415283,
    "Corr": 0.6999
  },
  "elapsed_seconds": 14750.47,
  "peak_gpu_memory_bytes": 9847915520,
  "peak_gpu_reserved_bytes": 12224299008,
  "max_rss_bytes": 19601211392,
  "checkpoint": "/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-repro/seed_1111/models/cmcm-mosei-regression.pth",
  "upstream_log": "/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-repro/seed_1111/work/logs/cmcm-mosei.log",
  "provenance": {
    "git_commit": "e0617fec510b2f2c4539e7f8b000efb3dbd98740",
    "scientific_tree_status": "",
    "harness_sha256": "0230391738aec42e6f70112eaff54a070962d4719c0e6d469c1dd3d7da58ffa1"
  },
  "files": {
    "dataset": {
      "path": "/gpfs/work/cpt/jiachenhou23/datasets/MSA/data/CMU-MOSEI/Processed/unaligned_50.pkl",
      "size": 13652131313,
      "mtime_ns": 1788282495510194736,
      "sha256": "ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f"
    },
    "model": {
      "path": "/gpfs/work/cpt/jiachenhou23/models/Qwen/Qwen-1_8B",
      "revision": "48d498eecd71554d63a31e2456fac189a09dd976",
      "files": {
        "model-00001-of-00002.safetensors": {
          "size": 2039259008,
          "mtime_ns": 1788380146668453224,
          "sha256": "a28a3ee886cf0ab305dce2e5f3de20c31469e52ac229d3bcd02e5ba0d3532036"
        },
        "model-00002-of-00002.safetensors": {
          "size": 1634419264,
          "mtime_ns": 1788380175092632374,
          "sha256": "252ba3b5c5b63cb4f2021afa8ca926c986b6fe56a2d92ffa4a6178d3031ebf51"
        },
        "model.safetensors.index.json": {
          "size": 14706,
          "mtime_ns": 1788380180287874000,
          "sha256": "5360fbb6cb272649b2ad8c615d2f2f4c5ef666f244e36a03530aae046b5e0536"
        },
        "config.json": {
          "size": 910,
          "mtime_ns": 1788380112467328000,
          "sha256": "5587c8c453793b612e99ca4236ca36746eb61b059766877c046bd3bf5e9d09a7"
        },
        "qwen.tiktoken": {
          "size": 2561218,
          "mtime_ns": 1788380182211237000,
          "sha256": "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186"
        }
      }
    }
  },
  "hyperparameters": {
    "batch_size": 16,
    "learning_rate": 0.005,
    "warm_up_epochs": 30,
    "early_stop": 10,
    "update_epochs": 1,
    "a_lstm_hidden_size": 64,
    "v_lstm_hidden_size": 32,
    "a_lstm_layers": 1,
    "v_lstm_layers": 1,
    "pseudo_tokens": 4,
    "max_new_tokens": 4,
    "task_specific_prompt": "Please predict the sentiment intensity of the above multimodal content in the range [-3.0, +3.0]. Assistant: The sentiment is"
  },
  "environment": {
    "hostname": "gpu3090n2",
    "python": "3.10.12",
    "python_executable": "/gpfs/work/cpt/jiachenhou23/MSE-Adapter/.venv-repro/bin/python",
    "torch": "2.0.1",
    "torch_cuda": "11.7",
    "cudnn": 8500,
    "transformers": "4.36.1",
    "modelscope": "1.10.0",
    "tiktoken": "0.5.2",
    "transformers_stream_generator": "0.0.4",
    "numpy": "1.26.2",
    "pandas": "2.1.4",
    "scikit_learn": "1.3.2",
    "cuda_visible_devices": "0",
    "visible_gpu_count": 1,
    "gpu": "NVIDIA GeForce RTX 3090",
    "gpu_total_memory_bytes": 25323503616,
    "slurm_job_id": "2826880",
    "slurm_job_name": "sys/dashboard/sys/jupyter",
    "slurm_partition": "gpu3090",
    "slurm_qos": "8gpus",
    "slurm_gpus_on_node": "1",
    "slurm_mem_per_node_mib": "20000",
    "slurm_cpus_on_node": "4",
    "slurm_job_end_time": "1788463190"
  }
}
```

<!-- END ARTIFACT: outputs/qwen-mosei-repro/seed_1111/result.json -->

## 11. 完整源代码

这里的“完整源代码”指本实验自有实现、提交/聚合脚本、测试，以及此次执行路径直接调用的本地上游配置、数据加载、指标与 Storage 工具。外部框架和 Qwen 第三方实现不重复复制。

### 11.1 `mse_router/__init__.py`

SHA-256：`68ff95818d4ca73b5bf86d6cf494c7d41d8c68bf162d85251affde929a33af9f`

<!-- BEGIN FILE: mse_router/__init__.py -->

```python
"""Conflict- and uncertainty-aware multimodal routing for Qwen/MOSEI."""

from .math_utils import (
    masked_softmax,
    normalized_entropy,
    normalized_js_divergence,
    soft_ordinal_targets,
)

__all__ = [
    "masked_softmax",
    "normalized_entropy",
    "normalized_js_divergence",
    "soft_ordinal_targets",
]
```

<!-- END FILE: mse_router/__init__.py -->

### 11.2 `mse_router/math_utils.py`

SHA-256：`921ab0b3841937dc5b90a5061a38474051caae4a4344004135e1e897f72973cc`

<!-- BEGIN FILE: mse_router/math_utils.py -->

```python
"""Small, dependency-light mathematical helpers used by the router."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


ANCHORS = torch.arange(-3.0, 4.0)


def soft_ordinal_targets(
    labels: torch.Tensor, anchors: torch.Tensor | None = None
) -> torch.Tensor:
    """Linearly interpolate a continuous target between adjacent ordinal anchors."""
    if anchors is None:
        anchors = ANCHORS.to(device=labels.device, dtype=labels.dtype)
    if anchors.ndim != 1 or anchors.numel() < 2:
        raise ValueError("anchors must be a one-dimensional tensor with >= 2 entries")
    if not bool(torch.all(anchors[1:] > anchors[:-1])):
        raise ValueError("anchors must be strictly increasing")

    values = labels.reshape(-1).to(dtype=anchors.dtype)
    values = values.clamp(float(anchors[0]), float(anchors[-1]))
    upper = torch.searchsorted(anchors, values, right=False)
    upper = upper.clamp(1, anchors.numel() - 1)
    lower = upper - 1
    exact_low = values <= anchors[0]
    exact_high = values >= anchors[-1]
    lower = torch.where(exact_low, torch.zeros_like(lower), lower)
    upper = torch.where(exact_low, torch.zeros_like(upper), upper)
    top_index = torch.full_like(lower, anchors.numel() - 1)
    lower = torch.where(exact_high, top_index, lower)
    upper = torch.where(exact_high, top_index, upper)

    lower_anchor = anchors[lower]
    upper_anchor = anchors[upper]
    denominator = (upper_anchor - lower_anchor).clamp_min(
        torch.finfo(anchors.dtype).eps
    )
    upper_weight = torch.where(
        lower == upper,
        torch.zeros_like(values),
        (values - lower_anchor) / denominator,
    )
    result = torch.zeros(
        values.shape[0], anchors.numel(), device=values.device, dtype=values.dtype
    )
    result.scatter_add_(1, lower[:, None], (1.0 - upper_weight)[:, None])
    result.scatter_add_(1, upper[:, None], upper_weight[:, None])
    return result


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"shape mismatch: logits={logits.shape}, targets={targets.shape}")
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def normalized_entropy(probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy normalized to [0, 1] for a categorical distribution."""
    count = probabilities.shape[-1]
    if count < 2:
        raise ValueError("entropy requires at least two classes")
    p = probabilities.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1) / math.log(count)


def normalized_js_divergence(
    first: torch.Tensor, second: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Jensen-Shannon divergence normalized by log(2), hence bounded by one."""
    if first.shape != second.shape:
        raise ValueError(f"shape mismatch: first={first.shape}, second={second.shape}")
    p = first.clamp_min(eps)
    q = second.clamp_min(eps)
    midpoint = 0.5 * (p + q)
    kl_pm = (p * (p.log() - midpoint.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - midpoint.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm) / math.log(2.0)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over present modalities with exact zeros for absent modalities."""
    if logits.shape != mask.shape:
        raise ValueError(f"shape mismatch: logits={logits.shape}, mask={mask.shape}")
    present = mask.to(dtype=torch.bool)
    if bool((present.sum(dim=-1) == 0).any()):
        raise ValueError("each sample must contain at least one modality")
    masked_logits = logits.masked_fill(~present, torch.finfo(logits.dtype).min)
    weights = F.softmax(masked_logits, dim=-1)
    weights = weights * present.to(weights.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).eps
    )
```

<!-- END FILE: mse_router/math_utils.py -->

### 11.3 `mse_router/data.py`

SHA-256：`4742d82a8809c26f491fdf8becf6f185216f87e4654f471fa7f07c5cf4e2982e`

<!-- BEGIN FILE: mse_router/data.py -->

```python
"""Group-safe calibration split, batching, and modality corruptions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset


MODALITY_INDEX = {"text": 0, "audio": 1, "vision": 2}


def video_group(sample_id: Any) -> str:
    if isinstance(sample_id, bytes):
        sample_id = sample_id.decode("utf-8")
    return str(sample_id).split("$_$", maxsplit=1)[0]


@dataclass(frozen=True)
class CalibrationSplit:
    train_indices: list[int]
    calibration_indices: list[int]
    train_groups: int
    calibration_groups: int
    fingerprint: str


def make_calibration_split(
    dataset: Dataset, fraction: float = 0.10, seed: int = 20260903
) -> CalibrationSplit:
    """Reserve a video-group-disjoint, nearest-anchor-stratified train subset."""
    if not 0.0 < fraction < 0.5:
        raise ValueError("calibration fraction must be between zero and 0.5")
    ids = np.asarray(dataset.ids)
    labels = np.asarray(dataset.labels["M"]).reshape(-1)
    groups = np.asarray([video_group(item) for item in ids])
    strata = np.clip(np.rint(labels), -3, 3).astype(np.int64)
    n_splits = int(round(1.0 / fraction))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    target_count = len(dataset) * fraction
    candidates = list(splitter.split(np.zeros(len(dataset)), strata, groups))
    train_indices, calibration_indices = min(
        candidates, key=lambda pair: abs(len(pair[1]) - target_count)
    )
    train_groups = set(groups[train_indices])
    calibration_groups = set(groups[calibration_indices])
    overlap = train_groups & calibration_groups
    if overlap:
        raise RuntimeError(f"video group leakage in calibration split: {sorted(overlap)[:3]}")
    digest = hashlib.sha256()
    for index in sorted(int(item) for item in calibration_indices):
        digest.update(f"{index}:{ids[index]}\n".encode("utf-8"))
    return CalibrationSplit(
        train_indices=[int(item) for item in train_indices],
        calibration_indices=[int(item) for item in calibration_indices],
        train_groups=len(train_groups),
        calibration_groups=len(calibration_groups),
        fingerprint=digest.hexdigest(),
    )


def build_router_dataloaders(
    upstream: dict[str, DataLoader],
    microbatch_size: int,
    num_workers: int,
    split_seed: int = 20260903,
) -> tuple[dict[str, DataLoader], CalibrationSplit]:
    split = make_calibration_split(upstream["train"].dataset, seed=split_seed)
    datasets: dict[str, Dataset] = {
        "train": Subset(upstream["train"].dataset, split.train_indices),
        "calibration": Subset(
            upstream["train"].dataset, split.calibration_indices
        ),
        "valid": upstream["valid"].dataset,
        "test": upstream["test"].dataset,
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=microbatch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        for name, dataset in datasets.items()
    }
    return loaders, split


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("text", "audio", "vision", "text_lengths", "audio_lengths", "vision_lengths"):
        moved[key] = batch[key].to(device, non_blocking=True)
    moved["labels"] = dict(batch["labels"])
    moved["labels"]["M"] = batch["labels"]["M"].view(-1).to(
        device, non_blocking=True
    )
    return moved


def _add_audio_noise(
    audio: torch.Tensor,
    lengths: torch.Tensor,
    selected: torch.Tensor,
    snr_low: float,
    snr_high: float,
) -> None:
    for index in torch.nonzero(selected, as_tuple=False).flatten().tolist():
        length = int(lengths[index].item())
        valid = audio[index, :length]
        signal_power = valid.float().pow(2).mean()
        if not bool(torch.isfinite(signal_power)) or float(signal_power) <= 0.0:
            continue
        snr = torch.empty((), device=audio.device).uniform_(snr_low, snr_high)
        noise_power = signal_power / torch.pow(10.0, snr / 10.0)
        noise = torch.randn_like(valid) * noise_power.sqrt().to(valid.dtype)
        audio[index, :length] = valid + noise


def _mask_vision_span(
    vision: torch.Tensor,
    lengths: torch.Tensor,
    selected: torch.Tensor,
    fraction_low: float,
    fraction_high: float,
) -> None:
    for index in torch.nonzero(selected, as_tuple=False).flatten().tolist():
        length = max(1, int(lengths[index].item()))
        fraction = float(
            torch.empty((), device=vision.device).uniform_(
                fraction_low, fraction_high
            )
        )
        span = max(1, min(length, int(round(length * fraction))))
        max_start = length - span
        start = int(
            torch.randint(max_start + 1, (), device=vision.device).item()
        )
        vision[index, start : start + span] = 0


def augment_modalities(
    batch: dict[str, Any],
    modality_drop_probability: float = 0.30,
    audio_noise_probability: float = 0.20,
    vision_mask_probability: float = 0.20,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Apply the stage-training corruptions and return an explicit presence mask."""
    augmented = dict(batch)
    augmented["audio"] = batch["audio"].clone()
    augmented["vision"] = batch["vision"].clone()
    batch_size = batch["text"].shape[0]
    device = batch["text"].device
    presence = torch.ones(batch_size, 3, device=device, dtype=torch.float32)
    drop_selected = torch.rand(batch_size, device=device) < modality_drop_probability
    dropped_modality = torch.randint(3, (batch_size,), device=device)
    rows = torch.nonzero(drop_selected, as_tuple=False).flatten()
    if rows.numel():
        presence[rows, dropped_modality[rows]] = 0.0

    audio_selected = (
        (torch.rand(batch_size, device=device) < audio_noise_probability)
        & presence[:, MODALITY_INDEX["audio"]].bool()
    )
    _add_audio_noise(
        augmented["audio"],
        batch["audio_lengths"],
        audio_selected,
        5.0,
        20.0,
    )
    vision_selected = (
        (torch.rand(batch_size, device=device) < vision_mask_probability)
        & presence[:, MODALITY_INDEX["vision"]].bool()
    )
    _mask_vision_span(
        augmented["vision"],
        batch["vision_lengths"],
        vision_selected,
        0.10,
        0.30,
    )
    return augmented, presence


def robustness_condition(
    batch: dict[str, Any], condition: str
) -> tuple[dict[str, Any], torch.Tensor]:
    """Create a deterministic evaluation corruption for one named condition."""
    changed = dict(batch)
    changed["audio"] = batch["audio"].clone()
    changed["vision"] = batch["vision"].clone()
    batch_size = batch["text"].shape[0]
    device = batch["text"].device
    presence = torch.ones(batch_size, 3, device=device)
    if condition == "clean":
        return changed, presence
    if condition.startswith("missing_"):
        modality = condition.removeprefix("missing_")
        if modality not in MODALITY_INDEX:
            raise ValueError(f"unknown missing-modality condition: {condition}")
        presence[:, MODALITY_INDEX[modality]] = 0.0
        return changed, presence
    if condition.startswith("audio_snr_"):
        snr = float(condition.removeprefix("audio_snr_"))
        selected = torch.ones(batch_size, device=device, dtype=torch.bool)
        _add_audio_noise(
            changed["audio"], batch["audio_lengths"], selected, snr, snr
        )
        return changed, presence
    if condition.startswith("vision_mask_"):
        percent = float(condition.removeprefix("vision_mask_"))
        selected = torch.ones(batch_size, device=device, dtype=torch.bool)
        fraction = percent / 100.0
        _mask_vision_span(
            changed["vision"],
            batch["vision_lengths"],
            selected,
            fraction,
            fraction,
        )
        return changed, presence
    raise ValueError(f"unknown robustness condition: {condition}")


ROBUSTNESS_CONDITIONS = (
    "clean",
    "missing_text",
    "missing_audio",
    "missing_vision",
    "audio_snr_20",
    "audio_snr_10",
    "audio_snr_0",
    "vision_mask_25",
    "vision_mask_50",
    "vision_mask_75",
)
```

<!-- END FILE: mse_router/data.py -->

### 11.4 `mse_router/model.py`

SHA-256：`0e7dde3741c88833d67443d624d17a0aec043dc659df92d133acde742ef08c86`

<!-- BEGIN FILE: mse_router/model.py -->

```python
"""Qwen-1.8B multimodal pseudo-token router.

The language model is shared by the three diagnostic passes and the final
generative pass.  Its parameters stay frozen, while gradients through it train
the modality adapters and (during stage 1) the shared ordinal head.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .math_utils import (
    masked_softmax,
    normalized_entropy,
    normalized_js_divergence,
    soft_cross_entropy,
    soft_ordinal_targets,
)


LOGGER = logging.getLogger("mse_router")
MODALITIES = ("text", "audio", "vision")
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.to(dtype=torch.bool)
        if bool((valid.sum(dim=1) == 0).any()):
            raise ValueError("every text sample must contain at least one valid token")
        scores = self.score(tokens).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


class PackedLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        safe_lengths = lengths.reshape(-1).long().clamp(min=1, max=sequence.shape[1])
        packed = pack_padded_sequence(
            sequence,
            safe_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.rnn(packed)
        # Indexing rather than squeeze preserves the batch dimension for B=1.
        return self.projection(self.dropout(hidden[-1]))


class MultiScaleProjector(nn.Module):
    """The original MSE-Adapter multi-scale projector, instantiated per modality."""

    def __init__(
        self, input_size: int = 256, hidden_size: int = 2048, pseudo_tokens: int = 4
    ) -> None:
        super().__init__()
        scale_hidden = 256
        self.scale1 = nn.Sequential(
            nn.Linear(input_size, hidden_size // 8),
            nn.GELU(),
            nn.Linear(hidden_size // 8, scale_hidden),
        )
        self.scale2 = nn.Sequential(
            nn.Linear(input_size, hidden_size // 32),
            nn.GELU(),
            nn.Linear(hidden_size // 32, scale_hidden),
        )
        self.scale3 = nn.Sequential(
            nn.Linear(input_size, hidden_size // 16),
            nn.GELU(),
            nn.Linear(hidden_size // 16, scale_hidden),
        )
        self.integrating = nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1)
        self.multi_scale_projector = nn.Linear(scale_hidden, hidden_size)
        self.token_projector = nn.Linear(1, pseudo_tokens)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scales = torch.stack(
            [self.scale1(features), self.scale2(features), self.scale3(features)], dim=2
        )
        integrated = self.integrating(scales.unsqueeze(1)).squeeze(3).squeeze(1)
        hidden = self.multi_scale_projector(integrated)
        return self.token_projector(hidden.unsqueeze(-1)).permute(0, 2, 1)


class Router(nn.Module):
    def __init__(self, input_size: int = 30, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        return masked_softmax(self.network(features), presence)


class QwenMseRouter(nn.Module):
    """Conflict-aware pseudo-token fusion while preserving generative supervision."""

    def __init__(self, args: Any) -> None:
        super().__init__()
        from modelscope import AutoModelForCausalLM, AutoTokenizer

        self.hidden_size = int(getattr(args, "router_hidden_size", 2048))
        self.pseudo_tokens = int(getattr(args, "pseudo_tokens", 4))
        self.max_new_tokens = int(getattr(args, "max_new_tokens", 4))
        self.aux_weight = float(getattr(args, "router_aux_weight", 0.3))
        self.router_variant = str(getattr(args, "router_variant", "full"))
        self.task_prompt = str(
            getattr(
                args,
                "task_specific_prompt",
                "Please predict the sentiment intensity of the above multimodal "
                "content in the range [-3.0, +3.0]. Assistant: The sentiment is",
            )
        )
        diagnostic_prompt = str(
            getattr(
                args,
                "diagnostic_prompt",
                "Predict this modality's sentiment intensity from -3 to +3.",
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrain_LM, padding_side="left", trust_remote_code=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            args.pretrain_LM, trust_remote_code=True, torch_dtype=torch.float16
        ).half()
        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        self.llm.config.use_cache = False
        if bool(getattr(args, "gradient_checkpointing", True)):
            self.llm.gradient_checkpointing_enable()

        actual_hidden = int(self.llm.config.hidden_size)
        if actual_hidden != self.hidden_size:
            raise ValueError(
                f"Qwen hidden size is {actual_hidden}, expected {self.hidden_size}"
            )
        eos = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        bos = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.tokenizer.pad_token_id = eos
        self.tokenizer.bos_token_id = bos
        self.eos_token_id = int(eos)
        self.bos_token_id = int(bos)

        text_dim, audio_dim, vision_dim = args.feature_dims
        if int(text_dim) != self.hidden_size:
            raise ValueError(
                f"token embedding dimension is {text_dim}, expected {self.hidden_size}"
            )
        self.text_pool = MaskedAttentionPool(self.hidden_size)
        self.text_projection = nn.Sequential(
            nn.Linear(self.hidden_size, 256), nn.GELU()
        )
        self.text_adapter = MultiScaleProjector(
            256, self.hidden_size, self.pseudo_tokens
        )
        self.audio_encoder = PackedLSTMEncoder(
            int(audio_dim),
            int(getattr(args, "a_lstm_hidden_size", 64)),
            num_layers=int(getattr(args, "a_lstm_layers", 1)),
            dropout=float(getattr(args, "a_lstm_dropout", 0.0)),
        )
        self.audio_adapter = MultiScaleProjector(
            256, self.hidden_size, self.pseudo_tokens
        )
        self.vision_encoder = PackedLSTMEncoder(
            int(vision_dim),
            int(getattr(args, "v_lstm_hidden_size", 32)),
            num_layers=int(getattr(args, "v_lstm_layers", 1)),
            dropout=float(getattr(args, "v_lstm_dropout", 0.0)),
        )
        self.vision_adapter = MultiScaleProjector(
            256, self.hidden_size, self.pseudo_tokens
        )
        self.modality_embeddings = nn.Parameter(torch.empty(3, self.hidden_size))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)

        self.ordinal_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 7),
        )
        self.router = Router(30, 64)
        self.log_temperatures = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.register_buffer("anchors", torch.arange(-3.0, 4.0), persistent=True)

        self._register_prompt("bos_ids", [self.bos_token_id])
        self._register_prompt("wrapper_before_ids", self._token_ids("<Multimodal>"))
        self._register_prompt("wrapper_after_ids", self._token_ids("</Multimodal>"))
        self._register_prompt("diagnostic_prompt_ids", self._token_ids(diagnostic_prompt))
        self._register_prompt("task_prompt_ids", self._token_ids(self.task_prompt))

    def _token_ids(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        return encoded["input_ids"].reshape(-1).tolist()

    def _register_prompt(self, name: str, ids: Iterable[int]) -> None:
        self.register_buffer(
            name, torch.tensor(list(ids), dtype=torch.long), persistent=False
        )

    @property
    def temperatures(self) -> torch.Tensor:
        return self.log_temperatures.exp().clamp(0.05, 10.0)

    def set_temperatures(self, temperatures: torch.Tensor | Iterable[float]) -> None:
        values = torch.as_tensor(
            temperatures,
            device=self.log_temperatures.device,
            dtype=self.log_temperatures.dtype,
        )
        if values.shape != (3,) or not bool(torch.isfinite(values).all()):
            raise ValueError("temperatures must contain three finite values")
        if bool(((values < 0.05) | (values > 10.0)).any()):
            raise ValueError("temperatures must lie in [0.05, 10.0]")
        with torch.no_grad():
            self.log_temperatures.copy_(values.log())

    def _embedding_layer(self) -> nn.Module:
        return self.llm.base_model.get_input_embeddings()

    def _expanded_prompt(self, ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        return self._embedding_layer()(ids.unsqueeze(0).expand(batch_size, -1))

    def encode_modalities(
        self,
        text: tuple[torch.Tensor, torch.Tensor],
        audio: tuple[torch.Tensor, torch.Tensor],
        vision: tuple[torch.Tensor, torch.Tensor],
        presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_tensor, _ = text
        audio_tensor, audio_lengths = audio
        vision_tensor, vision_lengths = vision
        token_ids = text_tensor[:, 0, :].long()
        text_mask = text_tensor[:, 1, :].long()
        text_embeddings = self._embedding_layer()(token_ids)
        text_features = self.text_projection(
            self.text_pool(text_embeddings, text_mask)
        )
        audio_features = self.audio_encoder(audio_tensor, audio_lengths)
        vision_features = self.vision_encoder(vision_tensor, vision_lengths)
        pseudo = torch.stack(
            [
                self.text_adapter(text_features),
                self.audio_adapter(audio_features),
                self.vision_adapter(vision_features),
            ],
            dim=1,
        )
        type_embeddings = self.modality_embeddings.to(dtype=pseudo.dtype)
        pseudo = pseudo + type_embeddings[None, :, None, :]
        if presence is not None:
            pseudo = pseudo * presence[:, :, None, None].to(dtype=pseudo.dtype)
        return pseudo

    def _wrapped_prefix(self, pseudo_tokens: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        batch_size = pseudo_tokens.shape[0]
        dtype = pseudo_tokens.dtype
        parts = [
            self._expanded_prompt(self.bos_ids, batch_size).to(dtype=dtype),
            self._expanded_prompt(self.wrapper_before_ids, batch_size).to(dtype=dtype),
            pseudo_tokens,
            self._expanded_prompt(self.wrapper_after_ids, batch_size).to(dtype=dtype),
            self._expanded_prompt(prompt, batch_size).to(dtype=dtype),
        ]
        return torch.cat(parts, dim=1)

    def diagnostic_logits(self, pseudo: torch.Tensor) -> torch.Tensor:
        batch_size = pseudo.shape[0]
        flattened = pseudo.reshape(
            batch_size * 3, self.pseudo_tokens, self.hidden_size
        )
        sequence = self._wrapped_prefix(flattened, self.diagnostic_prompt_ids)
        attention_mask = torch.ones(
            sequence.shape[:2], device=sequence.device, dtype=torch.long
        )
        outputs = self.llm.transformer(
            inputs_embeds=sequence,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        last_valid = outputs.last_hidden_state[:, -1, :]
        return self.ordinal_head(last_valid).reshape(batch_size, 3, 7)

    def _router_statistics(
        self, logits: torch.Tensor, presence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = F.softmax(
            logits.float() / self.temperatures[None, :, None], dim=-1
        )
        present = presence.to(dtype=torch.bool)
        uniform = torch.full_like(probabilities, 1.0 / probabilities.shape[-1])
        probabilities = torch.where(present[:, :, None], probabilities, uniform)
        entropy = normalized_entropy(probabilities)
        pairs = ((0, 1), (0, 2), (1, 2))
        conflicts = []
        for first, second in pairs:
            pair_present = present[:, first] & present[:, second]
            divergence = normalized_js_divergence(
                probabilities[:, first], probabilities[:, second]
            )
            conflicts.append(divergence * pair_present.to(divergence.dtype))
        conflict = torch.stack(conflicts, dim=-1)
        entropy = entropy * presence.to(entropy.dtype)
        features = torch.cat(
            [
                probabilities.reshape(probabilities.shape[0], -1),
                conflict,
                entropy,
                presence.float(),
            ],
            dim=-1,
        )
        return probabilities, conflict, entropy, features

    def _apply_router_variant(self, features: torch.Tensor) -> torch.Tensor:
        features = features.clone()
        if self.router_variant == "full":
            return features
        if self.router_variant == "no_conflict":
            features[:, 21:24] = 0
        elif self.router_variant == "no_uncertainty":
            features[:, 24:27] = 0
        elif self.router_variant == "predictions_only":
            features[:, 21:27] = 0
        elif self.router_variant == "uncertainty_only":
            features[:, :24] = 0
        elif self.router_variant != "uniform":
            raise ValueError(f"unknown router variant: {self.router_variant}")
        return features

    def route(
        self, logits: torch.Tensor, presence: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probabilities, conflicts, entropy, raw_features = self._router_statistics(
            logits, presence
        )
        router_features = self._apply_router_variant(raw_features.detach())
        if self.router_variant == "uniform":
            weights = presence.float() / presence.float().sum(dim=-1, keepdim=True)
        else:
            weights = self.router(router_features, presence)
        return weights, {
            "probabilities": probabilities,
            "conflicts": conflicts,
            "entropy": entropy,
            "router_features": router_features,
        }

    def gated_pseudo_tokens(
        self, pseudo: torch.Tensor, weights: torch.Tensor, presence: torch.Tensor
    ) -> torch.Tensor:
        present_count = presence.float().sum(dim=-1, keepdim=True)
        scales = present_count * weights
        gated = pseudo * scales[:, :, None, None].to(dtype=pseudo.dtype)
        return gated.reshape(pseudo.shape[0], 3 * self.pseudo_tokens, self.hidden_size)

    def gated_non_text_pseudo_tokens(
        self, pseudo: torch.Tensor, weights: torch.Tensor, presence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate audio/vision tokens while keeping natural text as a residual path.

        Multiplication by the number of present modalities preserves unit scale
        for a uniform router. The returned mask removes tokens belonging to a
        deliberately dropped modality from the final Qwen attention graph.
        """
        present_count = presence.float().sum(dim=-1, keepdim=True)
        scales = present_count * weights[:, 1:]
        gated = pseudo[:, 1:] * scales[:, :, None, None].to(dtype=pseudo.dtype)
        batch_size = pseudo.shape[0]
        gated = gated.reshape(
            batch_size, 2 * self.pseudo_tokens, self.hidden_size
        )
        token_mask = presence[:, 1:, None].expand(-1, -1, self.pseudo_tokens)
        return gated, token_mask.reshape(batch_size, -1).long()

    def final_prefix(
        self,
        pseudo: torch.Tensor,
        weights: torch.Tensor,
        presence: torch.Tensor,
        text_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build v2 input: gated A/V pseudo tokens plus original text tokens."""
        batch_size = pseudo.shape[0]
        gated_av, av_mask = self.gated_non_text_pseudo_tokens(
            pseudo, weights, presence
        )
        text_ids = text_tensor[:, 0, :].long()
        text_mask = text_tensor[:, 1, :].long()
        text_mask = text_mask * presence[:, 0, None].long()
        raw_text = self._embedding_layer()(text_ids).to(dtype=pseudo.dtype)
        ones = lambda length: torch.ones(
            batch_size, length, device=pseudo.device, dtype=torch.long
        )
        bos = self._expanded_prompt(self.bos_ids, batch_size).to(dtype=pseudo.dtype)
        before = self._expanded_prompt(self.wrapper_before_ids, batch_size).to(
            dtype=pseudo.dtype
        )
        after = self._expanded_prompt(self.wrapper_after_ids, batch_size).to(
            dtype=pseudo.dtype
        )
        task = self._expanded_prompt(self.task_prompt_ids, batch_size).to(
            dtype=pseudo.dtype
        )
        embeddings = torch.cat([bos, before, gated_av, after, raw_text, task], dim=1)
        attention_mask = torch.cat(
            [
                ones(bos.shape[1]),
                ones(before.shape[1]),
                av_mask,
                ones(after.shape[1]),
                text_mask,
                ones(task.shape[1]),
            ],
            dim=1,
        )
        return embeddings, attention_mask

    def _teacher_forcing_loss(
        self,
        prefix: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        label_strings = [
            f"{float(value):+.1f}" for value in labels.detach().float().clamp(-3, 3)
        ]
        encoded = self.tokenizer(
            label_strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(prefix.device)
        label_ids = encoded["input_ids"]
        label_mask = encoded["attention_mask"].long()
        label_embeddings = self._embedding_layer()(label_ids).to(dtype=prefix.dtype)
        inputs = torch.cat([prefix, label_embeddings], dim=1)
        ignore = torch.full(
            prefix.shape[:2], -100, device=prefix.device, dtype=torch.long
        )
        targets = label_ids.masked_fill(label_mask == 0, -100)
        targets = torch.cat([ignore, targets], dim=1)
        attention_mask = torch.cat([prefix_attention_mask, label_mask], dim=1)
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        output = self.llm(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=targets,
            use_cache=False,
            return_dict=True,
        )
        return output.loss

    def forward(
        self,
        labels: torch.Tensor,
        text: tuple[torch.Tensor, torch.Tensor],
        audio: tuple[torch.Tensor, torch.Tensor],
        vision: tuple[torch.Tensor, torch.Tensor],
        presence: torch.Tensor | None = None,
        stage: str = "stage1",
    ) -> dict[str, torch.Tensor]:
        batch_size = labels.reshape(-1).shape[0]
        if presence is None:
            presence = torch.ones(batch_size, 3, device=labels.device)
        presence = presence.to(device=labels.device, dtype=torch.float32)
        pseudo = self.encode_modalities(text, audio, vision, presence)
        logits = self.diagnostic_logits(pseudo)
        weights, diagnostics = self.route(logits, presence)
        prefix, prefix_attention_mask = self.final_prefix(
            pseudo, weights, presence, text[0]
        )
        generation_loss = self._teacher_forcing_loss(
            prefix, prefix_attention_mask, labels.reshape(-1)
        )

        ordinal_targets = soft_ordinal_targets(labels, self.anchors)
        per_modality = soft_cross_entropy(
            logits.float() / self.temperatures[None, :, None],
            ordinal_targets[:, None, :].expand(-1, 3, -1),
        )
        auxiliary_loss = (
            per_modality * presence
        ).sum() / presence.sum().clamp_min(1.0)
        loss = generation_loss
        if stage == "stage1":
            loss = loss + self.aux_weight * auxiliary_loss
        elif stage != "router":
            raise ValueError(f"unknown training stage: {stage}")
        return {
            "Loss": loss,
            "GenerationLoss": generation_loss,
            "AuxiliaryLoss": auxiliary_loss,
            "OrdinalLogits": logits,
            "Weights": weights,
            "Presence": presence,
            **diagnostics,
        }

    @staticmethod
    def parse_responses(responses: list[str]) -> tuple[list[float], dict[str, Any]]:
        values: list[float] = []
        invalid: list[int] = []
        out_of_range: list[int] = []
        for index, response in enumerate(responses):
            normalized = response.replace("–", "-").replace("−", "-")
            match = NUMBER_PATTERN.search(normalized)
            if match is None:
                values.append(0.0)
                invalid.append(index)
                continue
            try:
                value = float(match.group(0))
            except ValueError:
                values.append(0.0)
                invalid.append(index)
                continue
            if not math.isfinite(value) or value < -3.0 or value > 3.0:
                values.append(0.0)
                out_of_range.append(index)
            else:
                values.append(value)
        return values, {
            "raw_responses": responses,
            "invalid_indices": invalid,
            "out_of_range_indices": out_of_range,
            "invalid_count": len(invalid),
            "out_of_range_count": len(out_of_range),
        }

    @torch.no_grad()
    def diagnose(
        self,
        text: tuple[torch.Tensor, torch.Tensor],
        audio: tuple[torch.Tensor, torch.Tensor],
        vision: tuple[torch.Tensor, torch.Tensor],
        presence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = text[0].shape[0]
        if presence is None:
            presence = torch.ones(batch_size, 3, device=text[0].device)
        presence = presence.float()
        pseudo = self.encode_modalities(text, audio, vision, presence)
        logits = self.diagnostic_logits(pseudo)
        weights, diagnostics = self.route(logits, presence)
        return {"logits": logits, "weights": weights, **diagnostics}

    @torch.no_grad()
    def generate(
        self,
        text: tuple[torch.Tensor, torch.Tensor],
        audio: tuple[torch.Tensor, torch.Tensor],
        vision: tuple[torch.Tensor, torch.Tensor],
        presence: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> list[float] | tuple[list[float], dict[str, Any]]:
        batch_size = text[0].shape[0]
        if presence is None:
            presence = torch.ones(batch_size, 3, device=text[0].device)
        presence = presence.float()
        pseudo = self.encode_modalities(text, audio, vision, presence)
        logits = self.diagnostic_logits(pseudo)
        weights, router_diagnostics = self.route(logits, presence)
        prefix, prefix_attention_mask = self.final_prefix(
            pseudo, weights, presence, text[0]
        )
        # Qwen's GenerationMixin path assumes a cache when inputs_embeds are
        # supplied.  This explicit loop keeps use_cache=False while still
        # appending every generated token to the next full-prefix pass.
        running_embeddings = prefix
        running_attention_mask = prefix_attention_mask
        generated: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, device=prefix.device, dtype=torch.bool)
        for _ in range(self.max_new_tokens):
            position_ids = running_attention_mask.cumsum(dim=-1) - 1
            position_ids.masked_fill_(running_attention_mask == 0, 0)
            output = self.llm(
                inputs_embeds=running_embeddings,
                attention_mask=running_attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            next_token = output.logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, self.eos_token_id), next_token
            )
            generated.append(next_token)
            finished = finished | (next_token == self.eos_token_id)
            next_embedding = self._embedding_layer()(next_token[:, None]).to(
                dtype=running_embeddings.dtype
            )
            running_embeddings = torch.cat(
                [running_embeddings, next_embedding], dim=1
            )
            running_attention_mask = torch.cat(
                [
                    running_attention_mask,
                    torch.ones(
                        batch_size, 1, device=prefix.device, dtype=torch.long
                    ),
                ],
                dim=1,
            )
            if bool(finished.all()):
                break
        outputs = torch.stack(generated, dim=1)
        responses = self.tokenizer.batch_decode(
            outputs,
            add_special_tokens=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        values, parsing = self.parse_responses(responses)
        if not return_diagnostics:
            return values
        diagnostics: dict[str, Any] = {
            **parsing,
            "weights": weights.detach().float().cpu(),
            "probabilities": router_diagnostics["probabilities"].detach().cpu(),
            "conflicts": router_diagnostics["conflicts"].detach().cpu(),
            "entropy": router_diagnostics["entropy"].detach().cpu(),
            "temperatures": self.temperatures.detach().cpu(),
        }
        return values, diagnostics

    def set_stage(self, stage: str) -> None:
        if stage not in {"stage1", "calibration", "router", "eval"}:
            raise ValueError(f"unknown stage: {stage}")
        for name, parameter in self.named_parameters():
            if name.startswith("llm.") or name == "log_temperatures":
                parameter.requires_grad = False
            elif stage == "stage1":
                parameter.requires_grad = True
            elif stage == "router":
                parameter.requires_grad = name.startswith("router.") and (
                    self.router_variant != "uniform"
                )
            else:
                parameter.requires_grad = False

    def adapter_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.text_pool,
            self.text_projection,
            self.text_adapter,
            self.audio_encoder,
            self.audio_adapter,
            self.vision_encoder,
            self.vision_adapter,
        )
        parameters = [parameter for module in modules for parameter in module.parameters()]
        parameters.append(self.modality_embeddings)
        return parameters

    def experiment_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only experiment parameters/buffers, never frozen Qwen weights."""
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith("llm.")
        }

    def load_experiment_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_non_llm = [
            name for name in incompatible.missing_keys if not name.startswith("llm.")
        ]
        if unexpected or missing_non_llm:
            raise RuntimeError(
                f"incompatible router checkpoint: missing={missing_non_llm}, "
                f"unexpected={unexpected}"
            )
```

<!-- END FILE: mse_router/model.py -->

### 11.5 `mse_router/trainer.py`

SHA-256：`a09e7dfd81d6e283e53396017fc232b039f349b37d5f3c4a4f9cea7f8baeeb81`

<!-- BEGIN FILE: mse_router/trainer.py -->

```python
"""Three-stage optimization and evaluation for the Qwen/MOSEI router."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup

from .data import (
    ROBUSTNESS_CONDITIONS,
    augment_modalities,
    move_batch,
    robustness_condition,
)
from .math_utils import soft_ordinal_targets
from .model import QwenMseRouter


LOGGER = logging.getLogger("mse_router")


class CalibrationError(RuntimeError):
    """Raised when temperature fitting fails the predeclared validity checks."""


class UnderperformingRunError(RuntimeError):
    """Raised when the seed-one quality gate rejects a collapsed configuration."""


@dataclass(frozen=True)
class TrainingSettings:
    adapter_lr: float = 5e-3
    head_router_lr: float = 1e-3
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-4
    warmup_fraction: float = 0.10
    gradient_clip: float = 1.0
    accumulation_steps: int = 4
    stage1_max_epochs: int = 40
    stage1_patience: int = 10
    router_max_epochs: int = 10
    router_patience: int = 3
    progress_interval: int = 100
    quality_gate_epoch: int = 4
    quality_gate_max_mae: float = 0.72
    quality_gate_min_corr: float = 0.30


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class RouterTrainer:
    def __init__(
        self,
        args: Any,
        output_dir: Path,
        settings: TrainingSettings,
    ) -> None:
        from utils.metricsTop import MetricsTop

        self.args = args
        self.output_dir = output_dir
        self.settings = settings
        self.metrics = MetricsTop(args).getMetics(args.datasetName)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _unpack(
        self, batch: dict[str, Any]
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        labels = batch["labels"]["M"].view(-1)
        text = (batch["text"], batch["text_lengths"])
        audio = (batch["audio"], batch["audio_lengths"])
        vision = (batch["vision"], batch["vision_lengths"])
        return labels, text, audio, vision

    def _save_checkpoint(
        self,
        model: QwenMseRouter,
        path: Path,
        stage: str,
        epoch: int,
        valid_metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "format_version": 1,
            "stage": stage,
            "epoch": epoch,
            "valid_metrics": valid_metrics,
            "router_variant": model.router_variant,
            "temperatures": model.temperatures.detach().cpu(),
            "model": model.experiment_state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @staticmethod
    def _load_checkpoint(model: QwenMseRouter, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu")
        model.load_experiment_state_dict(checkpoint["model"])
        return checkpoint

    def _optimizer(self, model: QwenMseRouter, stage: str) -> torch.optim.Optimizer:
        if stage == "stage1":
            adapter_parameters = [
                parameter
                for parameter in model.adapter_parameters()
                if parameter.requires_grad
            ]
            head_router_parameters = [
                parameter
                for module in (model.ordinal_head, model.router)
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            groups = [
                {"params": adapter_parameters, "lr": self.settings.adapter_lr},
                {
                    "params": head_router_parameters,
                    "lr": self.settings.head_router_lr,
                },
            ]
        elif stage == "router":
            groups = [
                {
                    "params": [
                        parameter
                        for parameter in model.router.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": self.settings.head_router_lr,
                }
            ]
        else:
            raise ValueError(f"unknown optimizer stage: {stage}")
        if not any(group["params"] for group in groups):
            raise ValueError(f"stage {stage} has no trainable parameters")
        return torch.optim.AdamW(
            groups,
            eps=self.settings.adam_epsilon,
            weight_decay=self.settings.weight_decay,
        )

    @staticmethod
    def _set_training_mode(model: QwenMseRouter, stage: str) -> None:
        model.train()
        if stage == "router":
            # Only the Router's dropout is active after stage 1. Frozen feature
            # extractors and the ordinal head remain deterministic.
            for module in (
                model.llm,
                model.text_pool,
                model.text_projection,
                model.text_adapter,
                model.audio_encoder,
                model.audio_adapter,
                model.vision_encoder,
                model.vision_adapter,
                model.ordinal_head,
            ):
                module.eval()

    def _train_epoch(
        self,
        model: QwenMseRouter,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.cuda.amp.GradScaler,
        stage: str,
        epoch: int,
        max_epochs: int,
    ) -> dict[str, float]:
        self._set_training_mode(model, stage)
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "generation": 0.0, "auxiliary": 0.0}
        batches = len(loader)
        epoch_started = time.time()
        for step, cpu_batch in enumerate(loader, start=1):
            batch = move_batch(cpu_batch, self.args.device)
            augmented, presence = augment_modalities(batch)
            labels, text, audio, vision = self._unpack(augmented)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(
                    labels,
                    text,
                    audio,
                    vision,
                    presence=presence,
                    stage=stage,
                )
                scaled_loss = output["Loss"] / self.settings.accumulation_steps
            if not bool(torch.isfinite(output["Loss"]).item()):
                raise RuntimeError(
                    f"non-finite {stage} loss at batch {step}: {output['Loss'].item()}"
                )
            scaler.scale(scaled_loss).backward()
            should_step = (
                step % self.settings.accumulation_steps == 0 or step == batches
            )
            if should_step:
                scale_before = float(scaler.get_scale())
                scaler.unscale_(optimizer)
                trainable = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                gradient_norm = clip_grad_norm_(trainable, self.settings.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                optimizer.zero_grad(set_to_none=True)
                if bool(torch.isfinite(gradient_norm).item()):
                    scheduler.step()
                elif scale_after < scale_before:
                    LOGGER.warning(
                        "%s gradient overflow at batch %d; GradScaler reduced "
                        "scale from %.0f to %.0f and skipped the update",
                        stage,
                        step,
                        scale_before,
                        scale_after,
                    )
                else:
                    raise RuntimeError(
                        f"non-finite {stage} gradient norm at batch {step} "
                        "without a GradScaler reduction"
                    )
            totals["loss"] += float(output["Loss"].detach())
            totals["generation"] += float(output["GenerationLoss"].detach())
            totals["auxiliary"] += float(output["AuxiliaryLoss"].detach())
            if step % self.settings.progress_interval == 0 or step == batches:
                elapsed = time.time() - epoch_started
                eta = elapsed / step * (batches - step)
                learning_rates = [group["lr"] for group in optimizer.param_groups]
                LOGGER.info(
                    "%s epoch %d/%d batch %d/%d (%.1f%%) loss=%.4f "
                    "gen=%.4f aux=%.4f lr=%s scale=%.0f eta=%.0fs",
                    stage,
                    epoch,
                    max_epochs,
                    step,
                    batches,
                    100.0 * step / batches,
                    totals["loss"] / step,
                    totals["generation"] / step,
                    totals["auxiliary"] / step,
                    [f"{value:.3e}" for value in learning_rates],
                    scaler.get_scale(),
                    eta,
                )
        return {key: value / max(1, batches) for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(
        self,
        model: QwenMseRouter,
        loader: torch.utils.data.DataLoader,
        mode: str,
        condition: str = "clean",
    ) -> dict[str, Any]:
        model.eval()
        predictions: list[torch.Tensor] = []
        truths: list[torch.Tensor] = []
        invalid = 0
        out_of_range = 0
        raw_count = 0
        weight_sum = torch.zeros(3)
        evaluation_started = time.time()
        for step, cpu_batch in enumerate(loader, start=1):
            batch = move_batch(cpu_batch, self.args.device)
            changed, presence = robustness_condition(batch, condition)
            labels, text, audio, vision = self._unpack(changed)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                values, diagnostics = model.generate(
                    text,
                    audio,
                    vision,
                    presence=presence,
                    return_diagnostics=True,
                )
            predictions.append(torch.tensor(values, dtype=torch.float32))
            truths.append(labels.detach().float().cpu())
            invalid += int(diagnostics["invalid_count"])
            out_of_range += int(diagnostics["out_of_range_count"])
            raw_count += len(values)
            weight_sum += diagnostics["weights"].sum(dim=0)
            if step % self.settings.progress_interval == 0 or step == len(loader):
                elapsed = time.time() - evaluation_started
                eta = elapsed / step * (len(loader) - step)
                LOGGER.info(
                    "%s %s batch %d/%d (%.1f%%) invalid=%d "
                    "out_of_range=%d eta=%.0fs",
                    mode,
                    condition,
                    step,
                    len(loader),
                    100.0 * step / len(loader),
                    invalid,
                    out_of_range,
                    eta,
                )
        prediction = torch.cat(predictions)
        truth = torch.cat(truths)
        metrics = {
            key: float(value) for key, value in self.metrics(prediction, truth).items()
        }
        result = {
            **metrics,
            "condition": condition,
            "samples": raw_count,
            "invalid_generations": invalid,
            "out_of_range_generations": out_of_range,
            "mean_router_weights": (weight_sum / max(1, raw_count)).tolist(),
        }
        LOGGER.info(
            "%s %s MAE=%.4f Corr=%.4f invalid=%d out_of_range=%d weights=%s",
            mode,
            condition,
            result["MAE"],
            result["Corr"],
            invalid,
            out_of_range,
            [round(item, 4) for item in result["mean_router_weights"]],
        )
        return result

    @torch.no_grad()
    def _collect_calibration_logits(
        self,
        model: QwenMseRouter,
        loader: torch.utils.data.DataLoader,
    ) -> tuple[np.ndarray, np.ndarray]:
        model.set_stage("calibration")
        model.eval()
        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, self.args.device)
            labels, text, audio, vision = self._unpack(batch)
            presence = torch.ones(labels.shape[0], 3, device=labels.device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pseudo = model.encode_modalities(text, audio, vision, presence)
                logits = model.diagnostic_logits(pseudo)
            targets = soft_ordinal_targets(labels, model.anchors)
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(targets.detach().float().cpu())
        return torch.cat(all_logits).numpy(), torch.cat(all_targets).numpy()

    def calibrate_temperatures(
        self,
        model: QwenMseRouter,
        loader: torch.utils.data.DataLoader,
    ) -> dict[str, Any]:
        LOGGER.info("calibration: collecting clean modality logits")
        logits, targets = self._collect_calibration_logits(model, loader)
        LOGGER.info("calibration: fitting three bounded scalar temperatures")

        def nll(modality: int, temperature: float) -> float:
            scaled = logits[:, modality, :] / temperature
            log_probabilities = scaled - logsumexp(scaled, axis=-1, keepdims=True)
            return float(-(targets * log_probabilities).sum(axis=-1).mean())

        temperatures: list[float] = []
        records: list[dict[str, Any]] = []
        valid = True
        for modality in range(3):
            baseline = nll(modality, 1.0)
            result = minimize_scalar(
                lambda temperature: nll(modality, float(temperature)),
                bounds=(0.05, 10.0),
                method="bounded",
                options={"xatol": 1e-5, "maxiter": 200},
            )
            temperature = float(result.x)
            calibrated = nll(modality, temperature)
            at_bound = temperature <= 0.0501 or temperature >= 9.999
            modality_valid = bool(
                result.success
                and math.isfinite(temperature)
                and math.isfinite(calibrated)
                and not at_bound
                and calibrated <= baseline + 1e-7
            )
            valid = valid and modality_valid
            temperatures.append(temperature)
            records.append(
                {
                    "modality": ("text", "audio", "vision")[modality],
                    "temperature": temperature,
                    "tau1_nll": baseline,
                    "calibrated_nll": calibrated,
                    "optimizer_success": bool(result.success),
                    "at_bound": at_bound,
                    "valid": modality_valid,
                    "message": str(result.message),
                }
            )
        report = {
            "valid": valid,
            "samples": int(logits.shape[0]),
            "temperatures": temperatures,
            "modalities": records,
        }
        write_json(self.output_dir / "calibration.json", report)
        if valid:
            model.set_temperatures(temperatures)
        return report

    def _fit_stage(
        self,
        model: QwenMseRouter,
        loaders: dict[str, torch.utils.data.DataLoader],
        stage: str,
        max_epochs: int,
        patience: int,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        model.set_stage(stage)
        optimizer = self._optimizer(model, stage)
        updates_per_epoch = math.ceil(
            len(loaders["train"]) / self.settings.accumulation_steps
        )
        total_updates = max_epochs * updates_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(round(total_updates * self.settings.warmup_fraction)),
            num_training_steps=total_updates,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        best_mae = float("inf")
        best_epoch = 0
        history: list[dict[str, Any]] = []
        for epoch in range(1, max_epochs + 1):
            started = time.time()
            train_losses = self._train_epoch(
                model,
                loaders["train"],
                optimizer,
                scheduler,
                scaler,
                stage,
                epoch,
                max_epochs,
            )
            valid = self.evaluate(model, loaders["valid"], mode=f"{stage}-valid")
            record = {
                "epoch": epoch,
                "elapsed_seconds": round(time.time() - started, 3),
                "train": train_losses,
                "valid": valid,
            }
            history.append(record)
            write_json(self.output_dir / f"{stage}_history.json", history)
            if valid["MAE"] <= best_mae - 1e-6:
                best_mae = float(valid["MAE"])
                best_epoch = epoch
                self._save_checkpoint(
                    model,
                    checkpoint_path,
                    stage,
                    epoch,
                    valid,
                    extra={"training_settings": asdict(self.settings)},
                )
            if (
                stage == "stage1"
                and epoch == self.settings.quality_gate_epoch
                and valid["MAE"] > self.settings.quality_gate_max_mae
                and valid["Corr"] < self.settings.quality_gate_min_corr
            ):
                quality_gate = {
                    "passed": False,
                    "epoch": epoch,
                    "observed_mae": valid["MAE"],
                    "observed_corr": valid["Corr"],
                    "required_mae_at_most": self.settings.quality_gate_max_mae,
                    "required_corr_at_least": self.settings.quality_gate_min_corr,
                }
                write_json(self.output_dir / "quality_gate.json", quality_gate)
                raise UnderperformingRunError(
                    "stage-1 quality gate failed after warmup: "
                    f"MAE={valid['MAE']:.4f}, Corr={valid['Corr']:.4f}"
                )
            if epoch - best_epoch >= patience:
                break
        self._load_checkpoint(model, checkpoint_path)
        return {
            "stage": stage,
            "best_epoch": best_epoch,
            "best_valid_mae": best_mae,
            "epochs_ran": len(history),
            "checkpoint": str(checkpoint_path),
        }

    def fit(
        self,
        model: QwenMseRouter,
        loaders: dict[str, torch.utils.data.DataLoader],
        run_robustness: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        stage1_path = self.output_dir / "checkpoints" / "stage1.pt"
        final_path = self.output_dir / "checkpoints" / "final.pt"
        stage1 = self._fit_stage(
            model,
            loaders,
            "stage1",
            self.settings.stage1_max_epochs,
            self.settings.stage1_patience,
            stage1_path,
        )
        calibration = self.calibrate_temperatures(model, loaders["calibration"])
        if not calibration["valid"]:
            failure = {
                "status": "calibration_invalid",
                "stage1": stage1,
                "calibration": calibration,
            }
            write_json(self.output_dir / "result.json", failure)
            raise CalibrationError(
                "temperature calibration was non-finite, reached a bound, or failed "
                "to improve soft NLL; refusing the remaining multi-seed run"
            )
        self._save_checkpoint(
            model,
            self.output_dir / "checkpoints" / "calibrated.pt",
            "calibration",
            0,
            {},
            extra=calibration,
        )

        if model.router_variant == "uniform":
            initial_valid = self.evaluate(
                model, loaders["valid"], mode="uniform-valid"
            )
            self._save_checkpoint(
                model, final_path, "uniform", 0, initial_valid, extra=calibration
            )
            router_stage = {
                "stage": "uniform",
                "best_epoch": 0,
                "best_valid_mae": initial_valid["MAE"],
                "epochs_ran": 0,
                "checkpoint": str(final_path),
            }
        else:
            router_stage = self._fit_stage(
                model,
                loaders,
                "router",
                self.settings.router_max_epochs,
                self.settings.router_patience,
                final_path,
            )

        test = self.evaluate(model, loaders["test"], mode="test")
        robustness: dict[str, Any] = {}
        if run_robustness:
            for condition in ROBUSTNESS_CONDITIONS:
                if condition == "clean":
                    robustness[condition] = test
                else:
                    robustness[condition] = self.evaluate(
                        model,
                        loaders["test"],
                        mode="robustness",
                        condition=condition,
                    )
            write_json(self.output_dir / "robustness.json", robustness)
        result = {
            "status": "ok",
            "stage1": stage1,
            "calibration": calibration,
            "router_stage": router_stage,
            "test": test,
            "robustness": robustness,
            "temperatures": model.temperatures.detach().cpu().tolist(),
            "elapsed_seconds": round(time.time() - started, 3),
            "final_checkpoint": str(final_path),
            "training_settings": asdict(self.settings),
        }
        write_json(self.output_dir / "result.json", _jsonable(result))
        return result
```

<!-- END FILE: mse_router/trainer.py -->

### 11.6 `scripts/run_qwen_mosei_router.py`

SHA-256：`3759ba24294e47fc20a49e2fe4ab63f96baa9b5fb1514e22a72ba53351dae7ea`

<!-- BEGIN FILE: scripts/run_qwen_mosei_router.py -->

```python
#!/usr/bin/env python3
"""Preflight, train, and aggregate the Qwen-1.8B MOSEI router experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = PROJECT_DIR / "MSE-Qwen-1.8B"
DEFAULT_DATASET = Path(
    "/gpfs/work/cpt/jiachenhou23/datasets/MSA/data/CMU-MOSEI/Processed/unaligned_50.pkl"
)
DEFAULT_MODEL = Path("/gpfs/work/cpt/jiachenhou23/models/Qwen/Qwen-1_8B")
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs" / "qwen-mosei-router-v2"
SEEDS = (1111, 2222, 3333, 4444, 5555)
EXPECTED_SPLITS = {"train": 16_326, "valid": 1_871, "test": 4_659}
EXPECTED_DATASET_SIZE = 13_652_131_313
EXPECTED_DATASET_SHA256 = (
    "ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f"
)
ROUTER_SOURCE_FILES = (
    PROJECT_DIR / "mse_router" / "math_utils.py",
    PROJECT_DIR / "mse_router" / "data.py",
    PROJECT_DIR / "mse_router" / "model.py",
    PROJECT_DIR / "mse_router" / "trainer.py",
    Path(__file__).resolve(),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train", "aggregate"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--preflight-result",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "preflight" / "result.json",
    )
    parser.add_argument(
        "--router-variant",
        choices=(
            "full",
            "no_conflict",
            "no_uncertainty",
            "predictions_only",
            "uncertainty_only",
            "uniform",
        ),
        default="full",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()
    if args.mode == "train" and args.seed is None:
        parser.error("train mode requires --seed")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance() -> dict[str, Any]:
    return {
        "files": {
            str(path.relative_to(PROJECT_DIR)): sha256_file(path)
            for path in ROUTER_SOURCE_FILES
        },
        "git_commit": os.popen(
            f"git -C {PROJECT_DIR} rev-parse HEAD"
        ).read().strip(),
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def environment_info() -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": package_version("transformers"),
        "modelscope": package_version("modelscope"),
        "scikit_learn": package_version("scikit-learn"),
        "scipy": package_version("scipy"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu.name if gpu else None,
        "gpu_total_memory_bytes": gpu.total_memory if gpu else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_qos": os.environ.get("SLURM_JOB_QOS"),
    }


def max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def validate_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("refusing GPU execution outside a Slurm allocation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Slurm allocation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, found {torch.cuda.device_count()}")


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_config(args: argparse.Namespace, microbatch: int, accumulation: int):
    sys.path.insert(0, str(UPSTREAM_DIR))
    from config.config_regression import ConfigRegression

    base = argparse.Namespace(
        is_tune=False,
        tune_mode=False,
        train_mode="regression",
        modelName="cmcm",
        datasetName="mosei",
        root_dataset_dir=str(args.dataset_path.resolve().parent),
        num_workers=args.num_workers,
        model_save_dir=str(args.output_root / "unused"),
        res_save_dir=str(args.output_root / "unused"),
        pretrain_LM=str(args.model_path.resolve()),
        gpu_ids=[0],
    )
    config = ConfigRegression(base).get_config()
    config.modelName = "mse_router"
    config.dataPath = str(args.dataset_path.resolve())
    config.pretrain_LM = str(args.model_path.resolve())
    config.device = torch.device("cuda:0")
    config.batch_size = microbatch
    config.update_epochs = accumulation
    config.router_hidden_size = 2048
    config.router_aux_weight = 0.3
    config.router_variant = args.router_variant
    config.gradient_checkpointing = True
    config.router_architecture = "natural_text_residual_gated_audio_vision_v2"
    config.diagnostic_prompt = (
        "Predict this modality's sentiment intensity from -3 to +3."
    )
    return config


def validate_files(args: argparse.Namespace, hash_dataset: bool) -> dict[str, Any]:
    dataset = args.dataset_path.resolve()
    model = args.model_path.resolve()
    if not dataset.is_file() or dataset.stat().st_size != EXPECTED_DATASET_SIZE:
        raise RuntimeError(f"unexpected or missing dataset: {dataset}")
    dataset_hash = sha256_file(dataset) if hash_dataset else None
    if dataset_hash is not None and dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"dataset SHA-256 mismatch: {dataset_hash}")
    required_model_files = (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "qwen.tiktoken",
    )
    missing = [name for name in required_model_files if not (model / name).is_file()]
    if missing:
        raise RuntimeError(f"missing Qwen files: {missing}")
    return {
        "dataset": {
            "path": str(dataset),
            "size": dataset.stat().st_size,
            "mtime_ns": dataset.stat().st_mtime_ns,
            "sha256": dataset_hash,
        },
        "model": {
            "path": str(model),
            "files": {
                name: {
                    "size": (model / name).stat().st_size,
                    "mtime_ns": (model / name).stat().st_mtime_ns,
                }
                for name in required_model_files
            },
        },
    }


def load_data(config: Any, microbatch: int):
    from data.load_data import MMDataLoader
    from mse_router.data import build_router_dataloaders

    upstream = MMDataLoader(config)
    for split, expected in EXPECTED_SPLITS.items():
        if len(upstream[split].dataset) != expected:
            raise RuntimeError(
                f"unexpected {split} sample count: {len(upstream[split].dataset)}"
            )
    return build_router_dataloaders(
        upstream, microbatch, config.num_workers, split_seed=20260903
    )


def _trainable_summary(model: torch.nn.Module) -> dict[str, Any]:
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "trainable_parameters": sum(parameter.numel() for _, parameter in named),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_tensors": len(named),
        "names": [name for name, _ in named],
    }


def _one_backward(
    model: Any,
    config: Any,
    loader: torch.utils.data.DataLoader,
    accumulation: int,
) -> dict[str, Any]:
    from mse_router.data import augment_modalities, move_batch

    model.set_stage("stage1")
    model.train()
    batch = move_batch(next(iter(loader)), config.device)
    augmented, presence = augment_modalities(batch)
    labels = augmented["labels"]["M"].view(-1)
    text = (augmented["text"], augmented["text_lengths"])
    audio = (augmented["audio"], augmented["audio_lengths"])
    vision = (augmented["vision"], augmented["vision_lengths"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        eps=1e-4,
        weight_decay=0.01,
    )
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    loss_scale_attempts = []
    for attempt in range(1, 17):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(
                labels, text, audio, vision, presence=presence, stage="stage1"
            )
        if not bool(torch.isfinite(output["Loss"]).item()):
            raise RuntimeError(f"non-finite preflight loss: {output['Loss'].item()}")
        scale_before = float(scaler.get_scale())
        scaler.scale(output["Loss"] / accumulation).backward()
        scaler.unscale_(optimizer)
        gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        if not gradients:
            raise RuntimeError("preflight produced no trainable gradients")
        nonfinite = [
            name
            for name, gradient in gradients.items()
            if not bool(torch.isfinite(gradient).all().item())
        ]
        categories = {
            "adapter": any(
                not name.startswith(("ordinal_head.", "router."))
                for name in gradients
            ),
            "ordinal_head": any(
                name.startswith("ordinal_head.") for name in gradients
            ),
            "router": any(name.startswith("router.") for name in gradients),
        }
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        loss_scale_attempts.append(
            {
                "attempt": attempt,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "nonfinite_gradient_names": nonfinite,
            }
        )
        if not nonfinite:
            if not all(categories.values()):
                raise RuntimeError(f"missing gradient categories: {categories}")
            break
        if scale_after >= scale_before:
            raise RuntimeError(
                "non-finite gradients did not trigger a GradScaler reduction: "
                f"{nonfinite}"
            )
    else:
        raise RuntimeError("dynamic loss scaling did not stabilize within 16 attempts")
    torch.cuda.synchronize()
    return {
        "batch_size": int(labels.shape[0]),
        "effective_batch_size": int(labels.shape[0] * accumulation),
        "loss": float(output["Loss"].detach()),
        "generation_loss": float(output["GenerationLoss"].detach()),
        "auxiliary_loss": float(output["AuxiliaryLoss"].detach()),
        "gradient_tensors": len(gradients),
        "gradient_categories": categories,
        "loss_scale_attempts": loss_scale_attempts,
        "elapsed_seconds": round(time.time() - started, 3),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


@torch.no_grad()
def _one_generate(model: Any, config: Any, loader: torch.utils.data.DataLoader):
    from mse_router.data import move_batch

    model.eval()
    batch = move_batch(next(iter(loader)), config.device)
    text = (batch["text"][:1], batch["text_lengths"][:1])
    audio = (batch["audio"][:1], batch["audio_lengths"][:1])
    vision = (batch["vision"][:1], batch["vision_lengths"][:1])
    with torch.cuda.amp.autocast(dtype=torch.float16):
        values, diagnostics = model.generate(
            text, audio, vision, return_diagnostics=True
        )
    return {
        "values": values,
        "raw_responses": diagnostics["raw_responses"],
        "weights": diagnostics["weights"].tolist(),
        "invalid_count": diagnostics["invalid_count"],
        "out_of_range_count": diagnostics["out_of_range_count"],
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    validate_allocation()
    setup_seed(1111)
    files = validate_files(args, hash_dataset=True)
    attempts = []
    selected = None
    model = None
    split = None
    loaders = None
    for microbatch, accumulation in ((4, 4), (2, 8), (1, 16)):
        try:
            config = build_config(args, microbatch, accumulation)
            if loaders is None or loaders["train"].batch_size != microbatch:
                loaders, split = load_data(config, microbatch)
            if model is None:
                from mse_router.model import QwenMseRouter

                model = QwenMseRouter(config).to(config.device)
            backward = _one_backward(model, config, loaders["train"], accumulation)
            attempts.append(
                {
                    "microbatch": microbatch,
                    "accumulation": accumulation,
                    "status": "ok",
                    **backward,
                }
            )
            selected = {
                "microbatch": microbatch,
                "accumulation": accumulation,
                "effective_batch": 16,
            }
            break
        except torch.cuda.OutOfMemoryError as error:
            attempts.append(
                {
                    "microbatch": microbatch,
                    "accumulation": accumulation,
                    "status": "oom",
                    "error": str(error),
                }
            )
            if model is not None:
                model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    if selected is None or model is None or loaders is None or split is None:
        raise RuntimeError(f"all microbatch preflight attempts failed: {attempts}")
    generated = _one_generate(model, config, loaders["valid"])
    trainable = _trainable_summary(model)
    if any(parameter.requires_grad for parameter in model.llm.parameters()):
        raise RuntimeError("Qwen is not completely frozen")
    return {
        "status": "ok",
        "mode": "preflight",
        "architecture": "natural_text_residual_gated_audio_vision_v2",
        "selected_batching": selected,
        "attempts": attempts,
        "generation_smoke": generated,
        "trainable": trainable,
        "split": {
            "optimization_samples": len(split.train_indices),
            "calibration_samples": len(split.calibration_indices),
            "optimization_groups": split.train_groups,
            "calibration_groups": split.calibration_groups,
            "fingerprint": split.fingerprint,
            "seed": 20260903,
        },
        "files": files,
        "provenance": source_provenance(),
        "environment": environment_info(),
        "max_rss_bytes": max_rss_bytes(),
    }


def verify_preflight(args: argparse.Namespace) -> dict[str, Any]:
    with args.preflight_result.resolve().open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") != "ok":
        raise RuntimeError(f"preflight did not succeed: {args.preflight_result}")
    if result["provenance"] != source_provenance():
        raise RuntimeError("router source changed after preflight")
    current = validate_files(args, hash_dataset=False)
    previous = result["files"]
    for section in ("dataset",):
        for key in ("path", "size", "mtime_ns"):
            if current[section][key] != previous[section][key]:
                raise RuntimeError(f"{section} changed after preflight ({key})")
    for name, record in previous["model"]["files"].items():
        if current["model"]["files"][name] != record:
            raise RuntimeError(f"Qwen file changed after preflight: {name}")
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_allocation()
    preflight_result = verify_preflight(args)
    setup_seed(args.seed)
    batching = preflight_result["selected_batching"]
    config = build_config(
        args, batching["microbatch"], batching["accumulation"]
    )
    loaders, split = load_data(config, batching["microbatch"])
    if split.fingerprint != preflight_result["split"]["fingerprint"]:
        raise RuntimeError("calibration split differs from preflight")
    from mse_router.model import QwenMseRouter
    from mse_router.trainer import RouterTrainer, TrainingSettings, write_json

    seed_dir = args.output_root.resolve() / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "seed": args.seed,
        "router_variant": args.router_variant,
        "architecture": config.router_architecture,
        "batching": batching,
        "split": preflight_result["split"],
        "files": preflight_result["files"],
        "provenance": preflight_result["provenance"],
        "environment": environment_info(),
        "started_unix": time.time(),
    }
    write_json(seed_dir / "manifest.json", manifest)
    model = QwenMseRouter(config).to(config.device)
    settings = TrainingSettings(accumulation_steps=batching["accumulation"])
    trainer = RouterTrainer(config, seed_dir, settings)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.fit(
        model, loaders, run_robustness=not args.skip_robustness
    )
    torch.cuda.synchronize()
    result.update(
        {
            "seed": args.seed,
            "router_variant": args.router_variant,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "max_rss_bytes": max_rss_bytes(),
            "environment": environment_info(),
            "provenance": source_provenance(),
        }
    )
    write_json(seed_dir / "result.json", result)
    manifest["status"] = "ok"
    manifest["finished_unix"] = time.time()
    manifest["result"] = str(seed_dir / "result.json")
    write_json(seed_dir / "manifest.json", manifest)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    results = []
    for seed in SEEDS:
        path = args.output_root.resolve() / f"seed_{seed}" / "result.json"
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("status") != "ok":
            raise RuntimeError(f"seed {seed} is not complete: {path}")
        results.append(result)
    metric_names = list(results[0]["test"].keys())
    numeric_metrics = {}
    for metric in metric_names:
        values = [result["test"].get(metric) for result in results]
        if all(isinstance(value, (int, float)) for value in values):
            numeric_metrics[metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "values": values,
            }
    summary = {
        "status": "ok",
        "seeds": list(SEEDS),
        "primary_metric": "MAE",
        "metrics": numeric_metrics,
        "success_mean_mae": numeric_metrics["MAE"]["mean"],
    }
    output = args.output_root.resolve() / "five_seed_summary.json"
    from mse_router.trainer import write_json

    write_json(output, summary)
    return summary


def configure_logging(args: argparse.Namespace) -> None:
    if args.mode == "train":
        directory = args.output_root.resolve() / f"seed_{args.seed}"
    else:
        directory = args.output_root.resolve() / args.mode
    directory.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(directory / "run.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(PROJECT_DIR))
    configure_logging(args)
    if args.mode == "preflight":
        output_path = args.output_root.resolve() / "preflight" / "result.json"
    elif args.mode == "train":
        output_path = args.output_root.resolve() / f"seed_{args.seed}" / "result.json"
    else:
        output_path = args.output_root.resolve() / "five_seed_summary.json"
    started = time.time()
    try:
        if args.mode == "preflight":
            result = preflight(args)
        elif args.mode == "train":
            result = train(args)
        else:
            result = aggregate(args)
        if args.mode == "preflight":
            from mse_router.trainer import write_json

            write_json(output_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        failure = {
            "status": "error",
            "mode": args.mode,
            "seed": args.seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "environment": environment_info(),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        raise


if __name__ == "__main__":
    main()
```

<!-- END FILE: scripts/run_qwen_mosei_router.py -->

### 11.7 `scripts/qwen_mosei_router_preflight.slurm`

SHA-256：`ca2feae6bf3a54e47d1934367063d1e2f6d75a087849bed6b319ad7a14b683b3`

<!-- BEGIN FILE: scripts/qwen_mosei_router_preflight.slurm -->

```bash
#!/usr/bin/env bash
#SBATCH --job-name=qwen-router-v2-preflight
#SBATCH --account=yushanpan_surf
#SBATCH --partition=gpu3090
#SBATCH --qos=gpudebug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-router-v2/slurm/%x-%j.out
#SBATCH --error=/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-router-v2/slurm/%x-%j.err

set -euo pipefail

project_dir=/gpfs/work/cpt/jiachenhou23/MSE-Adapter
python_bin="$project_dir/.venv-repro/bin/python"
torch_env=/gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/envs/torch-2.0-env

cd "$project_dir"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export LD_LIBRARY_PATH="$torch_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$python_bin" -u scripts/run_qwen_mosei_router.py preflight
```

<!-- END FILE: scripts/qwen_mosei_router_preflight.slurm -->

### 11.8 `scripts/qwen_mosei_router.slurm`

SHA-256：`df70ab68aa38d622914c03e64838adcfb0412a83f8f1fe7362a185eaade493e6`

<!-- BEGIN FILE: scripts/qwen_mosei_router.slurm -->

```bash
#!/usr/bin/env bash
#SBATCH --job-name=qwen-mosei-router-v2
#SBATCH --account=yushanpan_surf
#SBATCH --partition=gpu3090
#SBATCH --qos=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-router-v2/slurm/%x-%A_%a.out
#SBATCH --error=/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-router-v2/slurm/%x-%A_%a.err

set -euo pipefail

project_dir=/gpfs/work/cpt/jiachenhou23/MSE-Adapter
python_bin="$project_dir/.venv-repro/bin/python"
torch_env=/gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/envs/torch-2.0-env
seeds=(1111 2222 3333 4444 5555)
task_id=${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}
seed=${seeds[$task_id]:?Invalid array task ID: $task_id}

cd "$project_dir"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export LD_LIBRARY_PATH="$torch_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$python_bin" -u scripts/run_qwen_mosei_router.py train --seed "$seed"
```

<!-- END FILE: scripts/qwen_mosei_router.slurm -->

### 11.9 `scripts/qwen_mosei_router_aggregate.slurm`

SHA-256：`de6590e80368c93d65cbfc7fcdd5e7082ec97e9439f2bc66fdfe18e686ee13b0`

<!-- BEGIN FILE: scripts/qwen_mosei_router_aggregate.slurm -->

```bash
#!/usr/bin/env bash
#SBATCH --job-name=qwen-router-v2-summary
#SBATCH --account=yushanpan_surf
#SBATCH --partition=cpudebug
#SBATCH --qos=cpudebug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-router-v2/slurm/%x-%j.out
#SBATCH --error=/gpfs/work/cpt/jiachenhou23/MSE-Adapter/outputs/qwen-mosei-router-v2/slurm/%x-%j.err

set -euo pipefail

project_dir=/gpfs/work/cpt/jiachenhou23/MSE-Adapter
cd "$project_dir"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

"$project_dir/.venv-repro/bin/python" -u scripts/run_qwen_mosei_router.py aggregate
```

<!-- END FILE: scripts/qwen_mosei_router_aggregate.slurm -->

### 11.10 `scripts/submit_qwen_mosei_router.sh`

SHA-256：`1bde39e414f31a523554f911b7ebfbd28267710751562a61894f8f352b3c4fc6`

<!-- BEGIN FILE: scripts/submit_qwen_mosei_router.sh -->

```bash
#!/usr/bin/env bash
set -euo pipefail

project_dir=/gpfs/work/cpt/jiachenhou23/MSE-Adapter
cd "$project_dir"
mkdir -p outputs/qwen-mosei-router-v2/slurm

preflight_job=$(sbatch --parsable scripts/qwen_mosei_router_preflight.slurm)
seed_one_job=$(sbatch --parsable --dependency="afterok:$preflight_job" --array=0 scripts/qwen_mosei_router.slurm)
# The remaining four seeds cannot start unless seed 1111 completes with valid
# temperature calibration.  CalibrationError gives seed 1111 a non-zero exit.
remaining_job=$(sbatch --parsable --dependency="afterok:$seed_one_job" --array=1-4%4 scripts/qwen_mosei_router.slurm)
summary_job=$(sbatch --parsable --dependency="afterok:$seed_one_job:$remaining_job" scripts/qwen_mosei_router_aggregate.slurm)

printf 'preflight_job=%s\n' "$preflight_job"
printf 'seed_1111_job=%s\n' "$seed_one_job"
printf 'remaining_seeds_job=%s\n' "$remaining_job"
printf 'summary_job=%s\n' "$summary_job"
```

<!-- END FILE: scripts/submit_qwen_mosei_router.sh -->

### 11.11 `tests/test_mse_router.py`

SHA-256：`2c5a017f618971d8ef06a1e4bcfe7beb700f7aa449b584f116ea44a4ef8e287e`

<!-- BEGIN FILE: tests/test_mse_router.py -->

```python
from __future__ import annotations

import unittest

import numpy as np
import torch

from mse_router.data import augment_modalities, make_calibration_split
from mse_router.math_utils import (
    masked_softmax,
    normalized_entropy,
    normalized_js_divergence,
    soft_ordinal_targets,
)
from mse_router.model import (
    MultiScaleProjector,
    PackedLSTMEncoder,
    QwenMseRouter,
    Router,
)


class FakeDataset:
    def __init__(self) -> None:
        ids = []
        labels = []
        for group_index in range(70):
            label = (group_index % 7) - 3
            for clip in range(2):
                ids.append(f"video_{group_index}$_${clip}")
                labels.append([float(label)])
        self.ids = np.asarray(ids)
        self.labels = {"M": np.asarray(labels, dtype=np.float32)}

    def __len__(self) -> int:
        return len(self.ids)


class RouterMathTests(unittest.TestCase):
    def test_soft_ordinal_targets_interpolate_and_clip(self) -> None:
        labels = torch.tensor([-4.0, -2.5, 0.25, 3.0, 4.0])
        targets = soft_ordinal_targets(labels)
        self.assertTrue(torch.allclose(targets.sum(dim=-1), torch.ones(5)))
        self.assertEqual(targets[0].argmax().item(), 0)
        self.assertTrue(torch.allclose(targets[1, :2], torch.tensor([0.5, 0.5])))
        self.assertTrue(torch.allclose(targets[2, 3:5], torch.tensor([0.75, 0.25])))
        self.assertEqual(targets[-1].argmax().item(), 6)

    def test_entropy_and_js_are_normalized(self) -> None:
        uniform = torch.full((1, 7), 1.0 / 7.0)
        concentrated = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0]])
        opposite = torch.tensor([[0.0, 1, 0, 0, 0, 0, 0]])
        self.assertAlmostEqual(normalized_entropy(uniform).item(), 1.0, places=6)
        self.assertLess(normalized_entropy(concentrated).item(), 1e-5)
        self.assertLess(
            normalized_js_divergence(uniform, uniform).abs().item(), 1e-6
        )
        self.assertAlmostEqual(
            normalized_js_divergence(concentrated, opposite).item(), 1.0, places=5
        )

    def test_masked_softmax_excludes_missing_modalities(self) -> None:
        logits = torch.zeros(2, 3)
        mask = torch.tensor([[1, 1, 1], [1, 0, 1]])
        weights = masked_softmax(logits, mask)
        self.assertTrue(torch.allclose(weights[0], torch.full((3,), 1.0 / 3.0)))
        self.assertTrue(torch.allclose(weights[1], torch.tensor([0.5, 0.0, 0.5])))

    def test_zero_initialized_router_starts_uniform(self) -> None:
        router = Router()
        weights = router(torch.randn(3, 30), torch.ones(3, 3))
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0)))


class RouterComponentTests(unittest.TestCase):
    def test_projector_and_lstm_preserve_batch_one(self) -> None:
        encoder = PackedLSTMEncoder(5, 4)
        encoded = encoder(torch.randn(1, 6, 5), torch.tensor([4]))
        self.assertEqual(tuple(encoded.shape), (1, 256))
        projector = MultiScaleProjector(256, 32, 4)
        pseudo = projector(encoded)
        self.assertEqual(tuple(pseudo.shape), (1, 4, 32))

    def test_response_parser_rejects_invalid_and_out_of_range(self) -> None:
        values, diagnostics = QwenMseRouter.parse_responses(
            ["+1.2", "sentiment=-0.7", "unknown", "+4.0"]
        )
        self.assertEqual(values, [1.2, -0.7, 0.0, 0.0])
        self.assertEqual(diagnostics["invalid_indices"], [2])
        self.assertEqual(diagnostics["out_of_range_indices"], [3])

    def test_gate_preserves_uniform_scale(self) -> None:
        instance = object.__new__(QwenMseRouter)
        torch.nn.Module.__init__(instance)
        instance.pseudo_tokens = 4
        instance.hidden_size = 8
        pseudo = torch.randn(2, 3, 4, 8)
        presence = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32)
        pseudo[1, 1] = 0
        weights = torch.tensor(
            [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.0, 0.5]], dtype=torch.float32
        )
        gated = QwenMseRouter.gated_pseudo_tokens(
            instance, pseudo, weights, presence
        ).reshape_as(pseudo)
        self.assertTrue(torch.allclose(gated, pseudo))

    def test_v2_gate_keeps_uniform_audio_visual_scale_and_masks_missing(self) -> None:
        instance = object.__new__(QwenMseRouter)
        torch.nn.Module.__init__(instance)
        instance.pseudo_tokens = 4
        instance.hidden_size = 8
        pseudo = torch.randn(2, 3, 4, 8)
        presence = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32)
        pseudo[1, 1] = 0
        weights = torch.tensor(
            [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.0, 0.5]], dtype=torch.float32
        )
        gated, mask = QwenMseRouter.gated_non_text_pseudo_tokens(
            instance, pseudo, weights, presence
        )
        self.assertTrue(torch.allclose(gated.reshape(2, 2, 4, 8), pseudo[:, 1:]))
        self.assertTrue(torch.equal(mask[0], torch.ones(8, dtype=torch.long)))
        self.assertTrue(
            torch.equal(mask[1], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
        )

    def test_router_has_30_features_and_detaches_diagnostic_inputs(self) -> None:
        instance = object.__new__(QwenMseRouter)
        torch.nn.Module.__init__(instance)
        instance.log_temperatures = torch.nn.Parameter(
            torch.zeros(3), requires_grad=False
        )
        instance.router_variant = "full"
        instance.router = Router()
        torch.nn.init.normal_(instance.router.network[-1].weight)
        logits = torch.randn(2, 3, 7, requires_grad=True)
        presence = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32)
        weights, diagnostics = QwenMseRouter.route(instance, logits, presence)
        self.assertEqual(tuple(diagnostics["router_features"].shape), (2, 30))
        (weights * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
        self.assertIsNone(logits.grad)


class RouterDataTests(unittest.TestCase):
    def test_calibration_split_is_deterministic_and_group_safe(self) -> None:
        dataset = FakeDataset()
        first = make_calibration_split(dataset)
        second = make_calibration_split(dataset)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.calibration_indices, second.calibration_indices)
        train_groups = {
            dataset.ids[index].split("$_$")[0] for index in first.train_indices
        }
        calibration_groups = {
            dataset.ids[index].split("$_$")[0]
            for index in first.calibration_indices
        }
        self.assertFalse(train_groups & calibration_groups)

    def test_augmentation_drops_exactly_one_when_forced(self) -> None:
        batch_size = 32
        batch = {
            "text": torch.zeros(batch_size, 3, 5),
            "audio": torch.ones(batch_size, 6, 2),
            "vision": torch.ones(batch_size, 6, 2),
            "audio_lengths": torch.full((batch_size,), 6),
            "vision_lengths": torch.full((batch_size,), 6),
        }
        _, presence = augment_modalities(
            batch,
            modality_drop_probability=1.0,
            audio_noise_probability=0.0,
            vision_mask_probability=0.0,
        )
        self.assertTrue(torch.equal(presence.sum(dim=-1), torch.full((32,), 2.0)))


if __name__ == "__main__":
    unittest.main()
```

<!-- END FILE: tests/test_mse_router.py -->

### 11.12 `MSE-Qwen-1.8B/config/config_regression.py`

SHA-256：`4bb970d3818ceb7e866ff7e23717eafd74b5b740db47cd75afdc431b8c740cc0`

<!-- BEGIN FILE: MSE-Qwen-1.8B/config/config_regression.py -->

```python
import os
import argparse

from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'cmcm': self.__CMCM
        }
        # hyper parameters for datasets
        self.root_dataset_dir = args.root_dataset_dir
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = self.root_dataset_dir
        tmp = {
            'mosi':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (2048, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE'
                }
            },
            'mosei':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (2048, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE'
                }
            },


            'simsv2': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS_V2/ch-simsv2s.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (50, 925, 232),  # (text, audio, video)
                    'feature_dims': (2048, 25, 177),  # (text, audio, video)
                    'train_samples': 2722,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'MAE',
                }
            }
        }
        return tmp

    def __CMCM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_label_prefix':True,
                'need_normalized': False,
                'use_PLM': True,
                'save_labels': False,
            },
            # dataset
            'datasetParas':{
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'task_specific_prompt': 'Please predict the sentiment intensity of the above multimodal content in the range [-3.0, +3.0]. Assistant: The sentiment is',
                    'max_new_tokens': 4,
                    'pseudo_tokens': 4,
                    'batch_size': 16,
                    'learning_rate': 5e-3,
                    # feature subNets
                    'a_lstm_hidden_size': 64,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    'warm_up_epochs':30,
                    #loss weight   best：1
                    'gamma':1,
                    'update_epochs': 1,
                    'early_stop': 10,     #10和8没啥区别
                    # res
                    'H': 3.0,
                },

                'simsv2': {
                    # the batch_size of each epoch is update_epochs * batch_size
                    'max_new_tokens': 4,
                    'pseudo_tokens': 4,
                    'task_specific_prompt': '请对上述多模态内容的情感强度进行预测，范围在[-1.0, +1.0]之间。响应: 情感为',
                    'batch_size': 16,
                    'learning_rate': 5e-4,   #5e -4 较好
                    # feature subNets
                    'a_lstm_hidden_size': 64,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    'warm_up_epochs': 30,  # 不太确定是30还是40，先跑一把
                    'update_epochs': 1,
                    'early_stop': 10,
                    # loss weight  best：0.25
                    'gamma': 1,
                    # res
                    'H': 1.0
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args
```

<!-- END FILE: MSE-Qwen-1.8B/config/config_regression.py -->

### 11.13 `MSE-Qwen-1.8B/data/load_data.py`

SHA-256：`cf49bba620205b40799779aa7e6b919de48d37c213e1ee1be84cc15f6d7189f6`

<!-- BEGIN FILE: MSE-Qwen-1.8B/data/load_data.py -->

```python
import os
import logging
import pickle
import json
import numpy as np
import pandas as pd
import torch
import gzip
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from modelscope import AutoTokenizer, AutoModel
from operator import itemgetter
from torch.nn.utils.rnn import pad_sequence

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'simsv2': self.__init_simsv2,
            'meld': self.__init_meld,
            'iemocap': self.__init_iemocap,
            'cherma': self.__init_cherma,

        }
        DATA_MAP[args.datasetName]()

    def __init_meld(self):
        data_path = os.path.join(self.args.dataPath, self.args.datasetName + '_' + self.mode + '.pkl')
        label_index_mapping = self.args.label_index_mapping
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.vision = np.array(list(map(lambda item: item['features']['video'], data))).astype(np.float32)
            self.audio = np.array(list(map(lambda item: item['features']['audio'], data))).astype(np.float32)
            self.rawText = np.array(list(map(lambda item: item['features']['text'], data)))

            # self.labels = {
            #     'M': list(map(lambda item: item['label'], data))
            # }
            self.labels = {
                'M': list(map(lambda item: label_index_mapping.get(item['label'],-1), data))
            }
            if self.args.use_PLM:
                self.text = self.PLM_tokenizer(self.rawText)

        # label_mapping

        # self.labels['M']  = [label_index_mapping.get(label, -1) for label in self.labels['M']]

        if not self.args.need_data_aligned:
            self.audio_lengths = np.array(list(map(lambda item: item['features']['audio_len'], data)))
            self.vision_lengths = np.array(list(map(lambda item: item['features']['video_len'], data)))

    def __init_iemocap(self):
        return self.__init_meld()

    def __init_cherma(self):
        return self.__init_meld()

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
            if self.args.use_PLM:
                self.text = data[self.mode]['raw_text']
                self.text = self.PLM_tokenizer(self.text)

        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }

        if self.args.need_label_prefix:
            labels = self.labels['M']
            label_prefix = []
            for i in range(len(labels)):
                if labels[i] < 0:
                    label_prefix.append(f'negative,{labels[i].item():.{1}f}')
                elif labels[i] > 0:
                    label_prefix.append(f'positive,{labels[i].item():.{1}f}')
                else:
                    label_prefix.append(f'neutral,{labels[i].item():.{1}f}')
            self.labels_prefix = label_prefix

        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
            self.text_lengths = self.args.seq_lens[0]
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision != self.vision] = 0

        if  self.args.need_normalized:
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __init_simsv2(self):
        return self.__init_mosi()

    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        text_length, audio_length, video_length = self.args.seq_lens
        self.vision = Truncated(self.vision, video_length)
        self.text = Truncated(self.text, text_length)
        self.audio = Truncated(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

        # 这里text.shape是三维矩阵[sample_num,tokenizer_output,length]
        # tokenizer_output的3个维度分别是token_ids,mask(识别句子中padding的位置),segment_ids
    def get_seq_len(self):
        return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def PLM_tokenizer (self, rawtexts):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.pretrain_LM,
            padding_side='left',
            trust_remote_code=True
        )
        # self.pad_token_id = self.tokenizer.convert_tokens_to_ids('<|extra_0|>')
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        self.tokenizer.pad_token_id = self.eos_token_id
        token_list = []
        for text in rawtexts:
            text_tokenizer = self.tokenizer(text,
                                 padding='max_length',  # 如果样本长度不满足最大长度则填充
                                 truncation=True,  # 截断至最大长度
                                 max_length=self.args.seq_lens[0],
                                 return_tensors = 'pt',
                                 add_special_tokens=False
                                )

            token_ids = text_tokenizer['input_ids'].squeeze(0)  # tensor of token ids  torch.Size([max_len])
            attn_masks = text_tokenizer['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values  torch.Size([max_len])
            token_type_ids = [0] * len(token_ids)               #不区分上下句

            #调整维度
            input_ids = np.expand_dims(token_ids, 1)
            input_mask = np.expand_dims(attn_masks, 1)
            segment_ids = np.expand_dims(token_type_ids, 1)

            text_pretrain = np.concatenate([input_ids, input_mask, segment_ids], axis=1).T
            token_list.append(text_pretrain)

        # x_dimensions = [array.shape[1] for array in token_list]
        # # 计算 x 维度的平均值
        # average_x = np.mean(x_dimensions)
        # median_x = np.median(x_dimensions)
        token_list = np.array(token_list)
        return token_list


    def __getitem__(self, index):
        if self.args.train_mode == 'regression':
            sample = {
                'raw_text': self.rawText[index],
                'text': torch.Tensor(self.text[index]),
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'index': index,
                'id': self.ids[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
                'labels_prefix': self.labels_prefix[index]
            }
        else:
            sample = {
                'raw_text': self.rawText[index],
                'text': torch.Tensor(self.text[index]),
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'index': index,
                'labels': {k: v[index] for k, v in self.labels.items()}
                # 'labels': {torch.Tensor(self.labels)},
            }

        if not self.args.need_data_aligned:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
            sample['text_lengths'] = self.args.seq_lens[0]

        return sample



def MMDataLoader(args):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader
```

<!-- END FILE: MSE-Qwen-1.8B/data/load_data.py -->

### 11.14 `MSE-Qwen-1.8B/utils/metricsTop.py`

SHA-256：`3db5bb146dcea31c6ef43c5aa6cf99735d9953f09ed262d4ab3bd6210d5ad7f2`

<!-- BEGIN FILE: MSE-Qwen-1.8B/utils/metricsTop.py -->

```python
import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import r2_score
from itertools import chain
__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self, args):
        if args.train_mode == "regression":
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_regression,
                'MOSEI': self.__eval_mosei_regression,
                'SIMS': self.__eval_sims_regression,
                'SIMSV2': self.__eval_simsv2_regression
            }
        else:
            self.metrics_dict = {
                'IEMOCAP': self.__eval_iemocap_classification,
                'MELD': self.__eval_meld_classification,
                'CHERMA': self.__eval_cherma_classification
            }
            self.label_index_mapping = args.label_index_mapping

    def __eval_iemocap_classification(self, results, truths):
        # label_index_mapping = self.label_index_mapping
        # # 主要通过混淆矩阵来计算
        # results_indices = [label_index_mapping.get(label, label_index_mapping.get('neu')) for label in results]
        # truths_indices = [label_index_mapping.get(label, -1) for label in truths]
        # acc = accuracy_score(truths_indices, results_indices)
        # weight_F1 = f1_score(truths_indices, results_indices, average='weighted')
        acc = accuracy_score(truths, results)
        weight_F1 = f1_score(truths, results, average='weighted')

        eval_result = {
            'acc': acc,
            'weight_F1': weight_F1
        }
        return eval_result

    def __eval_cherma_classification(self, results, truths):
        acc = accuracy_score(truths, results)
        weight_F1 = f1_score(truths, results, average='weighted')
        eval_result = {
            'acc': acc,
            'weight_F1': weight_F1
        }
        return eval_result

    def __eval_meld_classification(self, results, truths):
        acc = accuracy_score(truths, results)
        weight_F1 = f1_score(truths, results, average='weighted')


        eval_result = {
            'acc': acc,
            'weight_F1': weight_F1
        }
        return eval_result




    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))


    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)


        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')
        
        eval_results = {
            "Has0_acc_2":  round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2":  round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }
        return eval_results


    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # weak sentiment two classes{[-0.6, 0.0], (0.0, 0.6]}
        ms_2 = [-1.01, 0.0, 1.01]
        weak_index_l = np.where(test_truth >= -0.4)[0]
        weak_index_r = np.where(test_truth <= 0.4)[0]
        weak_index = [x for x in weak_index_l if x in weak_index_r]
        test_preds_weak = test_preds[weak_index]
        test_truth_weak = test_truth[weak_index]
        test_preds_a2_weak = test_preds_weak.copy()
        test_truth_a2_weak = test_truth_weak.copy()
        for i in range(2):
            test_preds_a2_weak[np.logical_and(test_preds_weak > ms_2[i], test_preds_weak <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2_weak[np.logical_and(test_truth_weak > ms_2[i], test_truth_weak <= ms_2[i + 1])] = i

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i
        
        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i
 
        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a2_weak = self.__multiclass_acc(test_preds_a2_weak, test_truth_a2_weak)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')
        r2 = r2_score(test_truth, test_preds)
        eval_results = {
            "Mult_acc_2": mult_a2,
            "F1_score": f_score,
            "Mult_acc_2_weak": mult_a2_weak,
            "MAE": mae,
            "Corr": corr,  # Correlation Coefficient
            "Mult_acc_3": mult_a3,
            "Mult_acc_5": mult_a5,
            "R_squre": r2
        }
        return eval_results

    def __eval_simsv2_regression(self, y_pred, y_true):
        return self.__eval_sims_regression(y_pred, y_true)
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]
```

<!-- END FILE: MSE-Qwen-1.8B/utils/metricsTop.py -->

### 11.15 `MSE-Qwen-1.8B/utils/functions.py`

SHA-256：`fcb6340636efcec28c735ea28a75ec9e205010786185b29bfb9df13749277c0b`

<!-- BEGIN FILE: MSE-Qwen-1.8B/utils/functions.py -->

```python
def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key] if key in self else False
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"

```

<!-- END FILE: MSE-Qwen-1.8B/utils/functions.py -->

### 11.16 `requirements-qwen-repro.txt`

SHA-256：`b0915d3c214257763ece25e9c8f4730049c0b9e555525cb6702c6ea533989f8b`

<!-- BEGIN FILE: requirements-qwen-repro.txt -->

```text
-r requirements-repro.txt

# Extra runtime dependencies imported by the Qwen-1.8B remote model code.
tiktoken==0.5.2
transformers-stream-generator==0.0.4
```

<!-- END FILE: requirements-qwen-repro.txt -->

## 12. 审查者快速索引

- 模态增强、grouped calibration split：`mse_router/data.py`
- Adapter、诊断分布、JS/熵、Router、token 门控、Qwen 输入：`mse_router/model.py`
- AdamW、loss、early stopping、温度拟合、评估与鲁棒性：`mse_router/trainer.py`
- seed、路径校验、哈希锁定、训练入口、五 seed 聚合：`scripts/run_qwen_mosei_router.py`
- 资源、依赖与 array 映射：`scripts/*.slurm` 和 `scripts/submit_qwen_mosei_router.sh`
- 精确结果：第 10 节；当前判断：第 1、6、7 节。
