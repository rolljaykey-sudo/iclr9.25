# NAACL-10-12 — Original Router V2, no input augmentation

Complete original Router V2 no-augmentation source for multimodal sentiment analysis.
The original model, losses, optimizer settings, calibration and training stages are
preserved byte-for-byte. `SOURCE_MANIFEST.json` records the exported source checksums.

## Source and entry points

| Dataset | Backbone | Original revision | Original training seeds |
|---|---|---|---|
| CMU-MOSEI | Qwen-1.8B | `MSE-Router/revisions/no_augmentation_v1` | 1111, 2222, 3333, 4444, 5555 |
| CMU-MOSEI | Llama-2-7b-hf | same | 4444, 5555 |
| CMU-MOSEI | ChatGLM3-6B-base | same | 3333, 4444, 5555 |
| CH-SIMS v2 | Qwen-1.8B | `MSE-Router/revisions/simsv2_no_augmentation_v1` | 1111, 2222, 3333 |
| CH-SIMS v2 | ChatGLM3-6B-base | same | 1111, 2222, 3333 |

The per-backbone seed restrictions above are part of the original launchers and
are retained. The original SIMS v2 release has no Llama training entry point.

- `mse_router/model.py`: natural-text residual path, modality encoders and adapters,
  seven-anchor ordinal head, conflict/uncertainty features and softmax Router.
- `mse_router/backbone_model.py`: frozen Llama2 and ChatGLM3 interfaces.
- `mse_router/data.py`: video-group-disjoint calibration split and batching.
- `mse_router/trainer.py`: training, temperature calibration, validation and testing.
- Revision `scripts/` and `tests/`: original launchers, Slurm scripts and checks.
- Root `MSE-Router/mse_router/`: original V2 reference modules required by the
  no-augmentation source-integrity tests.
- `MSE-Adapter/`: bundled original configuration, data loader, metrics and required
  GLM model implementation, with upstream licenses.

## Original protocol

Disable modality dropping, audio noise and visual masking in both training stages.
Retain model-internal Dropout and the original stage-dependent train/eval modes.
Frozen LLM weights are FP16; trainable parameters stay FP32, with CUDA autocast and
GradScaler. Microbatch 4 × gradient accumulation 4 gives effective batch 16.

The original pipeline is:

1. Joint adapter / encoder / ordinal-head / Router training: at most 40 epochs,
   patience 10, generation loss plus auxiliary ordinal loss weighted by 0.3.
2. Fit three temperatures on the original approximately 10% video-disjoint training
   holdout (split seed 20260903).
3. Router-only training: at most 10 epochs, patience 3. Select by validation MAE
   within this stage, then evaluate the clean test set.

Original peak learning rates are **adapter 0.005, head / Router 0.001**, with AdamW,
10% warmup, cosine decay and gradient clipping. These are the original experiment
settings; the later LR/dropout joint-only tuning study is a separate release.
MOSEI uses labels on [-3, 3]; SIMS v2 uses native labels and anchors on [-1, 1].
The original numerical/calibration and performance-gate behavior is preserved.

## Environment

Python 3.10 and a CUDA-compatible PyTorch environment are required. The dependency
pins record the environment used by the original experiments:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r MSE-Router/requirements-qwen-repro.txt
```

Supply the original local Qwen / Llama / ChatGLM checkpoints and processed MOSEI or
SIMS v2 pickle separately. No checkpoint weights, processed datasets, training
outputs, virtual environments or credentials are included in this source repository.

The expected processed files are:

- MOSEI `unaligned_50.pkl`: SHA256
  `ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f`.
- SIMS v2 `ch-simsv2s.pkl`: SHA256
  `f8fd9a1dd070588a714ff357b00f31d5bb93277b4e502cf2dc4ff718a95fc49b`.

Dataset download references are in the original
[MSE-Adapter repository](https://github.com/AZYoung233/MSE-Adapter) and
[SIMS v2 processed dataset](https://huggingface.co/datasets/AZYoung/SIMSV2_processed).

## Portable launch wrapper

`run_noaug.py` is a thin wrapper around the unchanged original entry points. It
supplies explicit model/data paths, selects the bundled Adapter dependency, and
always passes `--training-augmentation none --router-variant full --skip-robustness`.
It does not change the architecture or training algorithm.

A CPU-only configuration / dependency check does not load model weights or data:

```bash
python run_noaug.py --dataset simsv2 --backbone qwen --mode preflight \
  --model-path /path/to/Qwen-1_8B --dataset-path /path/to/ch-simsv2s.pkl \
  --check-config
```

In a Slurm allocation with one visible GPU, run the same command without
`--check-config` for the required GPU preflight. Once it succeeds, use
`--mode train --seed 1111`. Each seed starts from fresh trainable modules;
preflight-updated weights are discarded.

Outputs remain under `MSE-Router/outputs/<backbone>-<dataset>-router-v2-no-augmentation/`.
Each seed has checkpoints, stage histories, calibration, `manifest.json`, `run.log`
and `result.json`. The launchers reject existing output directories. For a retry,
choose a new child directory with `--output-root` and run its matching preflight.
The Qwen MOSEI comparison command additionally requires original augmented-control
results; SIMS v2 aggregation requires all three completed matching seeds.

For XJTLU Slurm, `scripts/train_noaug.slurm` is a portable single-GPU example:

```bash
export REPO_DIR="$PWD"
export PYTHON_BIN="$PWD/.venv/bin/python"
export DATASET=simsv2 BACKBONE=qwen SEED=1111
export MODEL_PATH=/path/to/Qwen-1_8B DATASET_PATH=/path/to/ch-simsv2s.pkl
mkdir -p logs
# Choose live available resources and an appropriate wall time.
MODE=preflight sbatch --time=01:00:00 --output="$PWD/logs/%x-%j.out" \
  --error="$PWD/logs/%x-%j.err" scripts/train_noaug.slurm
# Replace PREFLIGHT_JOB_ID with the actual successful submission ID.
MODE=train sbatch --dependency=afterok:PREFLIGHT_JOB_ID \
  --output="$PWD/logs/%x-%j.out" --error="$PWD/logs/%x-%j.err" scripts/train_noaug.slurm
```

The original revision Slurm files preserve their historical HPC paths for source
reproducibility. Use the root portable wrapper/template for a different checkout.
Do not run GPU training on a login node.

## Validation

```bash
python scripts/check_source.py
python scripts/test_source.py
```

These run integrity checks and portable CPU unit tests in separate processes for
the two dataset implementations. Historical integration tests that require local
model files, datasets or old training outputs remain in the original test files;
the portable test command reports those exclusions explicitly. Upload preparation
does not launch new training or claim fresh GPU validation of the exported wrapper.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency attribution.
