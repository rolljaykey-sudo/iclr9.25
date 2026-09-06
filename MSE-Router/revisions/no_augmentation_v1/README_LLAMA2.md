# Llama2 MOSEI Router V2: no training-input augmentation

Fresh user-requested seeds: **4444, 5555**, each on one RTX 4090. Batch settings
explicitly match original V2 seeds **1111 and 2222**: microbatch **4**, gradient
accumulation **4**, effective batch **16**. Do not inherit seed 3333's batch 64.

Both Stage1 and Router-only training disable modality dropping, audio noise
and visual span masking. Model Dropout, FP16, frozen Llama-2-7b-hf, full V2
architecture, losses, auxiliary weight 0.3, learning rates 0.005/0.001,
optimizer/scheduler, calibration split (seed 20260903), early stopping and
best-Router-stage validation-MAE checkpoint selection are unchanged. Only
clean validation/test are run; robustness stress evaluation is skipped.

`scripts/run_llama2_noaug.py` reuses the immutable no-augmentation harness and
trainer, plus the byte-identical original V2 backbone implementation. Only
new Llama2 entry-point/script/test/docs files are added. No original Router,
Adapter, active Qwen/GLM source or historical result is changed.

The CLI permits only preflight/train, seeds 4444/5555, augmentation none and
the full router. It rejects batch settings other than 4 x 4 and output paths
outside `MSE-Router/outputs/llama2-mosei-router-v2-no-augmentation/`. Runs are
atomically reserved to prevent overwrites. Training requires its own passing
GPU preflight, including exact source fingerprints, data/model identity and
batching. Preflight weights are discarded; each seed initializes afresh.

After xjtlu-hpc live allocation/resource checks and CPU tests, submit the GPU
preflight with `scripts/llama2_noaug.slurm preflight`, partition gpu4090, QoS
gpudebug and a one-hour limit. After success, submit
`sbatch --partition=gpu4090 --qos=8gpus --dependency= --array=0-1%2
scripts/llama2_noaug.slurm train`. Task 0 = 4444, task 1 = 5555. Each requests
one GPU, four CPUs, 64 GiB and 96 hours. No seed depends on another seed.
Recheck queue/output existence before submission; never submit duplicates.

Outputs include per-seed `run.log`, `manifest.json`, stage histories,
`checkpoints/` and `result.json`, plus `preflight/`, `slurm/` and `submission/`.
Do not modify preflight-fingerprinted sources once jobs are submitted.

These two seeds currently lack same-seed, same-batch augmented V2 controls.
No paired ablation report or additional baseline training is auto-generated.
Do not merge differently augmented/batched runs into a single protocol mean.

## Submission on 2026-09-05

- 43 CPU tests, CLI validation and shell syntax checks passed. Original V2,
  active Qwen and active GLM source fingerprints remain unchanged.
- GPU preflight **2902087** completed successfully in 63 seconds on
  **gpu4090n8 / RTX 4090**, with finite optimizer/gradient checks in both stages
  and all modalities present. The untrained decode was nonnumeric (recorded);
  this is an execution smoke test, not a predictive-accuracy evaluation.
- Preflight data/model identity, calibration split and exact **4 x 4** batching
  match original V2 seeds **1111/2222**. Both new seeds also passed the actual
  preflight verifier before submission, including their source fingerprints.
- Training array **2902111** was submitted at 20:23 CST to **gpu4090/8gpus**:
  **2902111_0 = 4444**, **2902111_1 = 5555**. No dependencies; concurrency 2.
  Both were initially pending for Priority. Consult Slurm for current state.
- Exact submission commands, resource requests, source hashes, log paths and
  expected per-seed result paths are recorded under the output root's
  `submission/preflight.json` and `submission/training.json`.
