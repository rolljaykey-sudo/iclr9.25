# ChatGLM3 MOSEI Router V2: no training-input augmentation

User-requested seeds: **3333, 4444, 5555**, all fresh runs. Both Stage1 and the
Router-only stage disable audio noise, visual masking and modality dropping.
Model Dropout, FP16, microbatch 4, accumulation 4, effective batch 16, full V2
architecture, objectives, learning rates, calibration split, early stopping
and best-Router-stage validation-MAE checkpoint selection stay unchanged.
Only clean validation/test are run; robustness stress evaluation is skipped.

The implementation reuses the frozen Qwen no-augmentation harness and trainer,
but loads **ChatGLM3-6B-base**, GLM configuration/metrics, and the byte-identical
original V2 `backbone_model.py`. The new GLM launcher adjusts its harness only
inside the new process. Existing Qwen source fingerprints and all original V2
files remain unchanged; no Adapter files or historical results are modified.

Entry point: `scripts/run_chatglm3_noaug.py {preflight,train}`. It restricts
training seeds to the three requested values, the router to `full`, augmentation
to `none`, and output paths to the new GLM no-augmentation experiment. Each
run is atomically reserved; preflight/train output is never overwritten.
Training rechecks the GLM model, dataset, source hashes and settings against
its own successful GPU preflight. It cannot reuse a Qwen/legacy preflight.

After live cluster/allocation checks, submit the preflight with
`scripts/chatglm3_noaug.slurm preflight` and a one-hour debug allocation. After
it succeeds, submit `sbatch --partition=gpu3090 --qos=8gpus --dependency=
--array=0-2%3 scripts/chatglm3_noaug.slurm train` after the live resource checks.
The explicit partition override implements the user's RTX 3090 preference;
do not edit the Slurm template because it is covered by the preflight hashes.
Do not resubmit seeds that already have jobs or output directories.
Each task requests one GPU, four CPUs, 64 GiB and up to 96 hours, matching the
original GLM resource request. No seed depends on any other seed.

Outputs: `MSE-Router/outputs/chatglm3-mosei-router-v2-no-augmentation/`, with
`preflight/`, `seed_3333/`, `seed_4444/`, `seed_5555/`, `slurm/` and submission
records. Per-seed manifests record actual hardware and unchanged protocol.

Do not mix these runs with original augmented GLM seeds 1111/2222 to report
a single five-seed protocol. Matching augmented V2 seeds 3333/4444/5555 do not
currently exist, so no paired noise-ablation comparison is auto-generated.
No additional seeds or baseline runs are authorized by this submission.

## Submission on 2026-09-05

- 35 CPU tests and shell syntax checks passed. GLM model/data/split/batching
  match the original V2, and all non-augmentation training settings match.
- GPU preflight **2900832** completed successfully in 70 seconds on
  gpudebug/RTX 3090. Both stages had finite gradients after normal dynamic
  loss scaling; all modalities were present; the smoke decode was valid.
- Training array **2900851**: task 0 = 3333, task 1 = 4444, task 2 = 5555.
  Submitted to gpu3090,gpu4090/8gpus, with concurrency 3 and no dependencies.
  sat8gpus was already at its eight-job/GPU user limit, so it was not used.
- Exact commands, resources, seed mapping and expected artifacts are recorded
  in the new output root's `submission/` directory.
- At 17:41 CST, the user requested RTX 3090 directly. Pending tasks
  2900851_1 (4444) and 2900851_2 (5555) were updated in place to gpu3090 only.
  Task 2900851_0 (3333) had already started on gpu4090n3 at 17:35:52 and was
  preserved pending user direction about stopping/restarting it. No jobs were
  cancelled or resubmitted and no preflight-fingerprinted files were changed.
