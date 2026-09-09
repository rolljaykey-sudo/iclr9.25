# U-HERA

固定预算、效用对齐的层级证据路由器。首验为本地 Qwen-1.8B / MOSI /
seed 1111。代码和输出独立于之前的 MSE-Router 实验。

## Architecture

Text token embeddings → 256D projection; audio/vision → full LSTM sequences
(64/32 hidden units, projected to 256D). Each modality gets sinusoidal positions
and one independent Pre-LN relation block (4 heads, FFN 512, dropout 0.1).
Eight shared, sample-conditioned queries read all three modalities with
single-head evidence attention. A content router allocates each slot to
T/A/V/Null. Non-affine candidate LayerNorm, modality types, a bias-free up
projection and learned base prefixes produce exactly eight LLM embeddings.

`z_k = b_k + 0.1 * W_up(sum_m g[k,m] * (LN(e[k,m]) + type[m]))`

The frozen LLM receives BOS, multimodal wrappers, eight evidence tokens, the
original text, and the unchanged English task prompt. It uses greedy decoding
with four generated tokens. There are no ordinal, calibration, entropy,
conflict or diagnostic-LLM passes. The LLM stays in eval mode with frozen
parameters; input gradients remain enabled for student training. Qwen's local
RoPE implementation requires contiguous valid sequences; ordered compaction
preserves prefix/target alignment.

Stored A/V lengths are authoritative. Nonzero values outside those lengths
are masked explicitly; source feature arrays are not modified. Empty modalities
have zero attention and gate mass; all-empty input uses the base prefix.

## Training

Official MOSI train/valid/test: 1284/229/686. All train samples are used; no
calibration split, augmentation or extra losses. Stage limits/patience:

| Stage | Trainable modules | Max epochs / patience |
| --- | --- | --- |
| Evidence warmup | All evidence modules; uniform real-modality gates, Null off | 40 / 10 |
| Reference/cache | Immutable best warmup checkpoint; no optimization | Once |
| Router warmup | Only slot modality scoring, Null enabled | 10 / 3 |
| Joint | All external modules | 40 / 10 |

All external groups use AdamW LR 5e-4, epsilon 1e-4, decay 0.01, default betas,
10% warmup and cosine scheduling per stage, clip 1. Microbatch 4 × accumulation
4; the final accumulation window uses its actual sample count. External
parameters are FP32 under FP16 AMP; GradScaler starts at 1024. Reader dot
products/aggregation, softmax, CE and KL use FP32. Qwen checkpointing and flash
attention are disabled in this validated first implementation.

The reference computes four per-sample mean target-token CEs on each training
sample: full and direct T/A/V evidence deletion. Each deletion reuses the full
encoding, gates, context, positions and raw text. Utility targets are
`softmax([loss_minus_m - loss_full, 0] / tau_u)`, masking missing modalities.
`tau_u = max(population_std(valid_train_deltas), 0.001)`; Null's zero is a
reference convention. Learned stages use `L_gen + 0.1 * KL(p* || mean_slots(g))`.
The fixed train-only cache has unique IDs and reference/data/source identities.

Validation uses unrounded generated-score MAE, with 1e-6 improvement tolerance.
Joint epoch zero retains the already trained router checkpoint. Final selection
compares Router and Joint stages, keeping the earlier candidate on ties; the
uniform reference is reported independently. Final test predictions are reused
on resume only when model identity and sample IDs match. Checkpoints exclude
frozen LLM weights and record stage, optimizer, scheduler, scaler and RNG state.
Resume is supported at completed epoch boundaries; unfinished epochs are replayed.

## Evidence interpretation and analysis

`J[m,i] = mean_k(g[k,m] * alpha[k,m,i])`, so `sum_i J[m,i] = G[m]`.
G/J are auxiliary evidence budgets. Raw text bypasses the router, and shared
context can carry information across modalities. These scores and the deletion
utilities are not complete modality or raw-frame causal contributions.

After checkpoint selection, validation analysis removes direct modalities and
the highest/lowest/reproducible-random 25% of valid reading positions in each
modality. All three position choices remove the same number of positions;
ties use original order. Local deletion freezes full-pass LayerNorm statistics:
`sum_removed alpha_i * ((V_i - mean(e))/std(e) + type)` is subtracted from the
injection. Removing every position equals direct modality evidence deletion.
No attention/gate renormalization or prefix-length change occurs.

## Commands and artifacts

Interpreter: `/gpfs/work/cpt/jiachenhou23/MSE-Router/.venv-repro/bin/python`.
Configuration: `configs/qwen_mosi.json`. No new dependencies are required.

```bash
bash scripts/check.sh
python run.py validate
# The following modes require one assigned visible GPU in a RUNNING Slurm job.
python run.py preflight
python run.py train --resume
python run.py evaluate
python run.py analyze
python run.py report
```

`scripts/train.slurm` first runs a bounded GPU preflight in a separate process,
then starts/resumes production only when the preflight matches the immutable
source/configuration/data/model identity. Resource selection must be revalidated
before submission. `UHERA_SOURCE_DIR` can identify an archived code snapshot.

Production: `outputs/qwen_mosi/seed_1111/`. Key files are `manifest.json`,
`resolved_config.json`, `train.log`, `checkpoints/reference.pt`,
`utility_cache.pt`, stage `best.pt`/`last.pt`/`done.pt`, `checkpoints/final.pt`,
`predictions/test.jsonl`, `analysis/summary.json`, `result.json` and `report.md`.
Predictions include raw responses, parsing flags, gates and final alpha/J maps.
`run.py report` independently recomputes test metrics from exported predictions.

The GPU preflight uses eight training samples and four validation samples, two
optimizer updates per stage, and never uses the official test set. CPU tests
exercise the actual local Qwen implementation at tiny dimensions.
