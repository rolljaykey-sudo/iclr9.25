# Qwen V2: no training augmentation

This isolated copy tests only removal of training-input augmentation. It does
not modify the active V2, stability_v1, the Adapter repository or old results.
`upstream_snapshot.json` records the originals; model.py and math_utils.py are
byte-identical to V2. Use `../../.venv-repro/bin/python` from this revision.

## Fixed experiment

- Fresh initialization, seeds 1111, 2222, 3333, 4444, 5555 (the last two added
  at the user's request on 2026-09-05); frozen Qwen-1.8B, full V2 router.
- Both training stages use `--training-augmentation none`: no audio noise,
  visual span masking or modality dropping. Model Dropout is unchanged.
- FP16, microbatch 4, accumulation 4, effective batch 16; learning rates 0.005
  and 0.001, auxiliary weight 0.3. Original scheduler, clipping, early stopping,
  calibration split and objectives are preserved.
- Original selection remains: best validation MAE within the router-only stage.
- Jobs use `--skip-robustness`; clean validation/test, original metric definitions
  and stage histories remain. Do not compare whole-run time as training speed:
  historical runs additionally performed robustness evaluation.

## Interfaces and safety

`scripts/run_qwen_mosei_router.py` provides `preflight`, `train`, `aggregate`.
`--training-augmentation {standard,none}` defaults to `standard`; standard mode
uses a separate augmentation-control output default. `--seeds` is an explicit
aggregate-only seed list (default 1111 2222 3333). New training cannot reuse a
legacy/mismatching preflight or silently fall back to a different microbatch.

Each preflight/train/aggregate directory is atomically reserved before logging.
Existing directories and reports are never overwritten. Failures are retained;
new attempts require an explicitly chosen new output location and job scripts.
Do not alter source after preflight: each training verifies its hashes.

## Validation and execution

Run `PYTHONDONTWRITEBYTECODE=1 ../../.venv-repro/bin/python -m unittest discover -s tests -v`.
Syntax-check the shell/Slurm scripts with `bash -n`.
Use the xjtlu-hpc resource/allocation checks before GPU work. Each GPU task
requests one GPU, four CPUs and 64 GiB; training has a 48-hour wall limit.
The preflight tests actual optimizer groups, finite backward/updates in both
stages, frozen backbone, matching data split and clean generation. Its weights
are disposable and never used to initialize a seed.

After revalidating availability and selecting NOAUG_TRAIN_PARTITION/QOS and
NOAUG_PREFLIGHT_PARTITION/QOS, run `bash scripts/submit_no_augmentation.sh`.
An already-completed matching preflight Job ID may be passed as its argument.
All three seeds depend only on successful preflight, not on one another. Seed
1111 and the two-task array for seeds 2222/3333 can run concurrently whenever
Slurm assigns resources. IDs are recorded under `outputs/.../submission`.
No automatic precision/LR/batch changes or seed replacement are performed.
The last successful GPU task invokes CPU-only aggregation before releasing its
existing allocation. A file lock prevents duplicate reports when both remaining
seeds finish together. This avoids occupying the user's single cpudebug submission
slot, which currently has the pre-existing path-cleanup job. The standalone CPU
Slurm template is available for a later explicitly scheduled report job.

## Report

Outputs are under the Router root's
`outputs/qwen-mosei-router-v2-no-augmentation/`, with per-seed checkpoints,
histories, manifests and results. The aggregate command compares exactly the
requested seeds against `outputs/qwen-mosei-router-v2` and writes
`three_seed_comparison.json` and `three_seed_comparison.md`.
It checks completion, seed identity, data/model identity, split, batching and
non-augmentation training settings. It reports paired metrics, sample standard
deviations, learning-curve summaries, elapsed time and peak memory; worse results
are kept, and no statistical-significance claim is made from three seeds.

## Submitted on 2026-09-05

- 29 unit tests passed; CLI and shell syntax validated; original source hashes unchanged.
- Preflight 2899013 completed in 54 seconds on gpudebug/RTX 3090. Both stages
  reached finite gradients. The initial untrained decode was invalid (recorded
  explicitly); this smoke test validates execution, not sentiment accuracy.
- Seed 1111: 2899037_0; seeds 2222/3333: 2899038_1 and 2899038_2.
- At the user's request on 2026-09-05, the dependency on seed 1111 was removed
  from the already-submitted 2899038 array, without resubmission or new IDs.
- Training uses sat3090/sat8gpus, selected after live checks; the template's
  gpu4090 defaults were overridden at submission. It retains RTX 3090 hardware.
- Submission metadata and exact IDs are in the output root's `submission/`.
- Reports are generated only after all three seeds finish successfully. Their
  absence while jobs are running or failed is not evidence of an improvement.

## Additional seeds 4444 and 5555

`scripts/qwen_noaug_extra_seeds.slurm` maps array tasks 0/1 to seeds 4444/5555.
It uses the same runner, successful preflight and fixed experiment settings;
no preflight-fingerprinted sources or existing tasks are modified. After live
resource checks, submit with `sbatch --dependency= --array=0-1%2
scripts/qwen_noaug_extra_seeds.slurm`. Both tasks are independent of other seeds.
Do not repeat submission if jobs or output directories already exist.

Submitted on 2026-09-05 as 2899140_0 (4444) and 2899140_1 (5555), without
dependencies. Both initially requested gpu4090/8gpus. After observing priority
queueing, task 0 was updated in place to sat3090/sat8gpus (one remaining
submitted-job slot); task 1 was widened to gpu4090,gpu3090/8gpus. These scheduler
overrides are recorded under `submission/extra_seeds_4444_5555/`.

These seeds write to `seed_4444/` and `seed_5555/` in the same Router-only output
root. Submission metadata records the actual GPU partition and launcher hash;
each training manifest records its GPU hardware. The existing automatic
`three_seed_comparison.*` remains a report of 1111/2222/3333 only, not all five.
A five-seed report should be generated separately after all five complete,
without overwriting or relabeling the original three-seed report.
