#!/usr/bin/env bash
set -euo pipefail
project_dir=/gpfs/work/cpt/jiachenhou23/U-HERA
python_bin=/gpfs/work/cpt/jiachenhou23/MSE-Router/.venv-repro/bin/python
cd "$project_dir"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$python_bin" -B -m unittest discover -s tests -v
"$python_bin" -B run.py validate
bash -n scripts/train.slurm
