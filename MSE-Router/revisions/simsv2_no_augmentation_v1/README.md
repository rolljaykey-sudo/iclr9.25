# ChatGLM3 Router V2 on CH-SIMS v2, no input augmentation

User-requested fresh seeds: **1111, 2222, 3333**. Prefer RTX 4090, one GPU
per independent seed. This isolated revision preserves existing MOSEI/Qwen/GLM
source files and results. Initial source hashes are in `upstream_snapshot.json`.

## Dataset and protocol

- Supervised `ch-simsv2s.pkl`, from the dataset link in MSE-Adapter's README:
  https://huggingface.co/datasets/AZYoung/SIMSV2_processed
- Pinned dataset revision: `0bb42223a6f6099680865fbd807ef26804c43c5d`.
- Expected size: 3,620,390,813 bytes; SHA-256:
  `f8fd9a1dd070588a714ff357b00f31d5bb93277b4e502cf2dc4ff718a95fc49b`.
- The Slurm CPU inspection verifies the checksum, dimensions, label bounds,
  sample IDs, lengths and split counts, and writes `dataset_identity.json`.
- Use SIMS v2's native [-1, 1] labels, Chinese task prompt and upstream metrics.
  Use its feature dimensions (audio 25, vision 177) and native encoder sizes.
- Preserve the V2 seven-output ordinal head and 30-input Router. Rescale the
  seven evenly spaced ordinal anchors from [-3, 3] to [-1, 1]; align auxiliary
  targets, calibration strata, generation targets and parser range accordingly.
- Reserve about 10% of training samples for temperature calibration, disjoint
  by source video, with fixed split seed 20260903. Validation/test remain intact.

## Training

- Frozen ChatGLM3-6B-base, fresh trainable modules for each seed, FP16.
- Full Router V2; microbatch 4, accumulation 4, effective batch 16.
- Both stages disable modality dropping, audio noise and visual masking.
  Model-internal Dropout retains the original stage-specific behavior.
- Adapter LR 0.005, head/Router LR 0.001; auxiliary weight 0.3.
- Original AdamW, warmup/cosine schedule, clipping and early stopping:
  Stage 1 at most 40 epochs/patience 10; Router at most 10/patience 3.
- Checkpoint selection uses best validation MAE within the Router-only stage.
- Clean validation/test only; skip the additional robustness evaluation.
- Two CPUs, 48 GiB RAM, zero DataLoader workers per GPU; 12-hour training limit.

## Execution and outputs

Run CPU checks with `.venv-repro/bin/python -m unittest discover -s tests -v`
from this revision (use the Router-root interpreter and explicit library path).
Run a bounded GPU preflight in a validated Slurm allocation before training.
`scripts/chatglm3_simsv2.slurm preflight` supports Slurm resource overrides.
Training is array `0-2`; tasks map to seeds 1111, 2222, 3333. Select concurrency
from live user/QoS headroom. Successful preflight is required by each task.

Outputs are under `MSE-Router/outputs/chatglm3-simsv2-router-v2-no-augmentation/`:
`preflight/`, `seed_1111/`, `seed_2222/`, `seed_3333/`, `slurm/`, `submission/`.
Each seed writes checkpoints, both stage histories, calibration, manifest and
`result.json`. The last successful seed writes `three_seed_summary.json` and
`three_seed_summary.md`, using sample standard deviation (ddof=1).

The matched Qwen-1.8B run uses `scripts/qwen_simsv2.slurm` with the same
SIMS v2 split, seeds, batching, no-augmentation protocol, training stages and
clean evaluation. Its outputs are isolated under
`MSE-Router/outputs/qwen-simsv2-router-v2-no-augmentation/`.
