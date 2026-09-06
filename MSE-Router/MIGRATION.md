# Router root migration

Migration time: 2026-09-05 (Asia/Shanghai)

All Router implementation files, variants, tests, launch scripts, logs,
checkpoints, and results were moved from `../MSE-Adapter` into this project
root. The official Adapter backbone directories remain in `../MSE-Adapter` as
read-only upstream dependencies.

Jobs `2892711`, `2892712`, `2892923`, `2892924`, and `2892971` were already
running during the move. Validated compatibility links allow those jobs to
finish writing into this directory. Slurm job `2895248` depends on all five and
will remove only those links after every training job has terminated.

The old `scripts/run_backbone_mosei_router.py` compatibility link points to the
immutable launch-time source under `provenance/active-job-launch-sources/` so
the running jobs' final source hashes remain truthful. Future jobs use the
migration-aware entry point under `scripts/`.

Historical result manifests are immutable provenance records and therefore
retain the absolute paths that were valid when their runs started.

The validated Python environment was copied into `.venv-repro` and its text
entry points were rewritten to the new absolute prefix. The Adapter copy was
left intact for Adapter reproduction work and for jobs launched before the
migration.
