#!/usr/bin/env python3
"""Fresh Llama2 MOSEI V2 runs without training-input augmentation, batch 16."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import run_qwen_mosei_router as harness


REVISION = Path(__file__).resolve().parents[1]
ROUTER_ROOT = harness.ROUTER_ROOT
UPSTREAM = harness.ADAPTER_DIR / "MSE-Llama2-7B"
MODEL = ROUTER_ROOT.parent / "models/Meta/Llama-2-7b-hf"
OUTPUT_ROOT = ROUTER_ROOT / "outputs/llama2-mosei-router-v2-no-augmentation"
BASELINE_ROOT = ROUTER_ROOT / "outputs/llama2-mosei-router-v2"
SEEDS = (4444, 5555)
BATCHING = {"microbatch": 4, "accumulation": 4, "effective_batch": 16}
REQUIRED_MODEL_FILES = (
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
)
ORIGINAL_BUILD_CONFIG = harness.build_config
ORIGINAL_VERIFY_PREFLIGHT = harness.verify_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--dataset-path", type=Path, default=harness.DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--training-augmentation", choices=("none",), default="none")
    parser.add_argument("--router-variant", choices=("full",), default="full")
    parser.add_argument("--skip-robustness", action="store_true", default=True)
    args = parser.parse_args()
    if args.mode == "train" and args.seed is None:
        parser.error("train mode requires --seed")
    root = args.output_root.resolve()
    if not root.is_relative_to(OUTPUT_ROOT.resolve()):
        parser.error("output root must be inside the Llama2 no-augmentation experiment")
    args.preflight_result = args.preflight_result or root / "preflight/result.json"
    return args


def build_config(args: argparse.Namespace, microbatch: int, accumulation: int) -> Any:
    if (microbatch, accumulation) != (BATCHING["microbatch"], BATCHING["accumulation"]):
        raise ValueError("match Llama2 seeds 1111/2222: microbatch 4 x accumulation 4")
    config = ORIGINAL_BUILD_CONFIG(args, microbatch, accumulation)
    config.router_hidden_size = 4096
    config.router_backbone = "llama2"
    config.router_upstream_dir = str(UPSTREAM.resolve())
    return config


def validate_files(args: argparse.Namespace, hash_dataset: bool) -> dict[str, Any]:
    dataset = args.dataset_path.resolve()
    model = args.model_path.resolve()
    if not dataset.is_file() or dataset.stat().st_size != harness.EXPECTED_DATASET_SIZE:
        raise RuntimeError(f"unexpected or missing dataset: {dataset}")
    dataset_hash = harness.sha256_file(dataset) if hash_dataset else None
    if dataset_hash is not None and dataset_hash != harness.EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"dataset SHA-256 mismatch: {dataset_hash}")
    missing = [name for name in REQUIRED_MODEL_FILES if not (model / name).is_file()]
    if missing:
        raise RuntimeError(f"missing Llama2 files: {missing}")
    return {
        "dataset": {
            "path": str(dataset), "size": dataset.stat().st_size,
            "mtime_ns": dataset.stat().st_mtime_ns, "sha256": dataset_hash,
        },
        "model": {
            "backbone": "llama2", "path": str(model),
            "files": {
                name: {"size": (model / name).stat().st_size,
                       "mtime_ns": (model / name).stat().st_mtime_ns}
                for name in REQUIRED_MODEL_FILES
            },
        },
    }


def verify_preflight(args: argparse.Namespace) -> dict[str, Any]:
    result = ORIGINAL_VERIFY_PREFLIGHT(args)
    current = validate_files(args, hash_dataset=False)
    if result["files"]["model"] != current["model"]:
        raise RuntimeError("Llama2 model identity changed after preflight")
    return result


def configure_harness() -> None:
    harness.UPSTREAM_DIR = UPSTREAM
    harness.DEFAULT_MODEL = MODEL
    harness.DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT
    harness.BASELINE_ROOT = BASELINE_ROOT
    harness.ROUTER_SOURCE_FILES = (
        *(REVISION / "mse_router" / name for name in (
            "math_utils.py", "data.py", "model.py", "backbone_model.py", "trainer.py",
        )),
        REVISION / "scripts/run_qwen_mosei_router.py",
        REVISION / "scripts/llama2_noaug.slurm",
        REVISION / "upstream_snapshot.json",
        Path(__file__).resolve(),
        *(UPSTREAM / name for name in (
            "config/config_regression.py", "data/load_data.py", "utils/metricsTop.py",
        )),
    )
    harness.parse_args = parse_args
    harness.build_config = build_config
    harness.validate_files = validate_files
    harness.verify_preflight = verify_preflight
    # Process-local only: do not edit sources used by active Qwen/GLM jobs.
    import mse_router.model as router_model
    from mse_router.backbone_model import BackboneMseRouter
    router_model.QwenMseRouter = BackboneMseRouter


def main() -> None:
    configure_harness()
    harness.main()


if __name__ == "__main__":
    main()
