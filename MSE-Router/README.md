# MSE-Router experiments

This directory is the independent root for all Router-specific code, launch
scripts, tests, documentation, logs, checkpoints, and results. Router artifacts
must not be added to the upstream `MSE-Adapter` repository.

## Layout

- `mse_router/`: audited Router V2 implementation and design documentation.
- `mse_router_variants/`: independent-gate Router variants.
- `scripts/`: Python entry points, Slurm scripts, and submission helpers.
- `tests/`: Router-only unit tests.
- `outputs/`: all Router preflights, logs, checkpoints, and result summaries.

## External dependencies

The released Adapter implementations remain read-only upstream dependencies at
`../MSE-Adapter/MSE-Qwen-1.8B`, `../MSE-Adapter/MSE-Llama2-7B`, and
`../MSE-Adapter/MSE-ChatGLM3-6B`. Set `MSE_ADAPTER_ROOT` to override the Adapter
repository location.

The `.venv-repro` directory is a Router-local copy of the validated Python
environment. Future Router jobs therefore do not execute from an environment
stored inside the Adapter repository.

## Common commands

```bash
cd /gpfs/work/cpt/jiachenhou23/MSE-Router
.venv-repro/bin/python -m unittest discover -s tests -v
.venv-repro/bin/python scripts/run_qwen_mosei_router.py --help
bash scripts/submit_qwen_mosei_router.sh
```

Historical manifests created before the 2026-09-05 migration retain their
original absolute paths as provenance. New runs use this directory as their
project and output root.
