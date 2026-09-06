# Source export validation

Validated in the existing Python 3.10 / PyTorch 2.0.1 environment on 2026-09-06.

- 88 original files match their source SHA256 and byte length.
- 63 Python files pass syntax parsing.
- 37 portable original MOSEI tests and 9 original SIMS v2 tests pass (46 total).
- Seven original integration checks require historical outputs or local model and
  dataset files. They remain unchanged in the original test sources and are
  explicitly excluded by `scripts/test_source.py` in a source-only checkout.
- CPU configuration/import/provenance checks pass for all five supported
  dataset/backbone combinations using the root wrapper and bundled dependencies.
- Root and historical Slurm/shell scripts pass `bash -n`.
- The export includes no dataset/checkpoint binaries, environment directories or
  authentication files. The original source directories are unchanged.

No fresh GPU training was launched to validate this publication wrapper. GPU
execution still requires the original preflight checks before each training run.
