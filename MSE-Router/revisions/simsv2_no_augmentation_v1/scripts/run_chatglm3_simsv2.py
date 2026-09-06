#!/usr/bin/env python3
"""ChatGLM3 Router V2 on supervised CH-SIMS v2, without input augmentation."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

import run_qwen_mosei_router as harness


REVISION = Path(__file__).resolve().parents[1]
ROUTER_ROOT = REVISION.parents[1]
UPSTREAM = harness.ADAPTER_DIR / "MSE-ChatGLM3-6B"
MODEL = ROUTER_ROOT.parent / "models/THUDM/chatglm3-6b-base"
DATASET = ROUTER_ROOT.parent / "datasets/MSA/data/SIMS_V2/ch-simsv2s.pkl"
OUTPUT_ROOT = ROUTER_ROOT / "outputs/chatglm3-simsv2-router-v2-no-augmentation"
SEEDS = (1111, 2222, 3333)
REQUIRED_MODEL_FILES = (
    "config.json", "pytorch_model.bin.index.json", "tokenizer.model", "tokenizer_config.json",
    *(f"pytorch_model-{part:05d}-of-00007.bin" for part in range(1, 8)),
)
PROTOCOL = {
    "dataset": "simsv2", "backbone": "chatglm3-6b-base", "training_augmentation": "none",
    "score_range": [-1.0, 1.0], "ordinal_anchors": [-1 + i / 3 for i in range(7)],
    "ordinal_head_outputs": 7, "router_inputs": 30,
    "checkpoint_selection": "best_router_stage_valid_mae", "run_robustness": False,
    "calibration_fraction": 0.10, "calibration_split_seed": 20260903,
    "initialization": "fresh_trainable_modules_and_frozen_pretrained_backbone",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train", "aggregate"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--dataset-path", type=Path, default=DATASET)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--training-augmentation", choices=("none",), default="none")
    parser.add_argument("--router-variant", choices=("full",), default="full")
    parser.add_argument("--skip-robustness", action="store_true", default=True)
    args = parser.parse_args()
    if args.mode == "train" and args.seed is None:
        parser.error("train requires --seed")
    if not args.output_root.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        parser.error("output root must be within the SIMS v2 experiment")
    if args.num_workers != 0:
        parser.error("this validated resource configuration requires --num-workers 0")
    args.preflight_result = args.preflight_result or args.output_root / "preflight/result.json"
    return args


def build_config(args: argparse.Namespace, microbatch: int, accumulation: int) -> Any:
    import sys
    sys.path.insert(0, str(UPSTREAM))
    from config.config_regression import ConfigRegression

    base = argparse.Namespace(
        is_tune=False, tune_mode=False, train_mode="regression", modelName="cmcm",
        datasetName="simsv2", root_dataset_dir=str(args.dataset_path.resolve().parent),
        num_workers=args.num_workers, model_save_dir=str(args.output_root / "unused"),
        res_save_dir=str(args.output_root / "unused"), pretrain_LM=str(args.model_path.resolve()),
        gpu_ids=[0],
    )
    config = ConfigRegression(base).get_config()
    config.modelName = "mse_router"
    config.dataPath = str(args.dataset_path.resolve())
    config.device = torch.device("cuda:0")
    config.batch_size = microbatch
    config.update_epochs = accumulation
    config.router_hidden_size = 4096
    config.router_aux_weight = 0.3
    config.router_variant = "full"
    config.router_backbone = "chatglm3"
    config.router_upstream_dir = str(UPSTREAM.resolve())
    config.training_augmentation = "none"
    config.gradient_checkpointing = True
    config.router_architecture = "natural_text_residual_gated_audio_vision_v2"
    config.diagnostic_prompt = "请预测该模态的情感强度，范围为-1到+1。"
    return config


def dataset_identity() -> dict:
    identity = json.loads((REVISION / "dataset_identity.json").read_text())
    if identity["status"] != "ok" or identity["dataset"] != "simsv2":
        raise ValueError("A successful SIMS v2 data inspection is required")
    return identity


def validate_files(args: argparse.Namespace, hash_dataset: bool) -> dict[str, Any]:
    identity = dataset_identity()
    dataset, model = args.dataset_path.resolve(), args.model_path.resolve()
    if not dataset.is_file() or dataset.stat().st_size != identity["size"]:
        raise ValueError(f"SIMS v2 file identity mismatch: {dataset}")
    digest = harness.sha256_file(dataset) if hash_dataset else None
    if digest is not None and digest != identity["sha256"]:
        raise ValueError("SIMS v2 checksum mismatch")
    missing = [name for name in REQUIRED_MODEL_FILES if not (model / name).is_file()]
    if missing:
        raise ValueError(f"Missing ChatGLM3 model files: {missing}")
    return {
        "dataset": {"path": str(dataset), "size": dataset.stat().st_size,
                    "mtime_ns": dataset.stat().st_mtime_ns, "sha256": digest},
        "model": {"backbone": "chatglm3", "path": str(model), "files": {
            name: {"size": (model / name).stat().st_size,
                   "mtime_ns": (model / name).stat().st_mtime_ns}
            for name in REQUIRED_MODEL_FILES}},
    }


def aggregate_results(root: Path, require_all: bool = True) -> dict | None:
    from mse_router.trainer import write_json
    root.mkdir(parents=True, exist_ok=True)
    with (root / "aggregate.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        paths = [root / f"seed_{seed}" / "result.json" for seed in SEEDS]
        if not all(path.is_file() for path in paths):
            if require_all:
                raise ValueError("All three seed results are required")
            return None
        records = [json.loads(path.read_text()) for path in paths]
        if any(r.get("status") != "ok" for r in records):
            if require_all:
                raise ValueError("All three seeds must have completed successfully")
            return None
        if not require_all and any("dataset_protocol" not in r for r in records):
            return None
        output = root / "three_seed_summary.json"
        if output.exists():
            return json.loads(output.read_text())
        expected_test = dataset_identity()["splits"]["test"]["samples"]
        fields = ("MAE", "Corr", "Mult_acc_2", "Mult_acc_2_weak", "Mult_acc_3",
                  "Mult_acc_5", "F1_score", "R_squre")
        for seed, result in zip(SEEDS, records):
            if result["seed"] != seed or result.get("dataset_protocol") != PROTOCOL:
                raise ValueError(f"Seed/protocol mismatch in {seed}")
            if result["test"]["condition"] != "clean" or result["test"]["samples"] != expected_test:
                raise ValueError("Incomplete or wrong test set")
            if any(not math.isfinite(result["test"][key]) for key in fields):
                raise ValueError(f"Non-finite test metric in {seed}")
        metrics = {}
        for field in fields:
            values = [float(r["test"][field]) for r in records]
            metrics[field] = {"values": values, "mean": statistics.mean(values),
                              "sample_std": statistics.stdev(values)}
        summary = {"status": "ok", "seeds": list(SEEDS), "std_ddof": 1,
                   "dataset_protocol": PROTOCOL, "metrics": metrics,
                   "results": [str(path) for path in paths]}
        write_json(output, summary)
        lines = ["# ChatGLM3 SIMS v2, no augmentation", "", "Seeds: 1111, 2222, 3333. Clean test; sample standard deviation (ddof=1).", "",
                 "| Metric | Mean | Sample std |", "|---|---:|---:|"]
        lines += [f"| {k} | {v['mean']:.6f} | {v['sample_std']:.6f} |" for k, v in metrics.items()]
        with (root / "three_seed_summary.md").open("x") as handle:
            handle.write("\n".join(lines) + "\n")
        return summary


def configure_harness(args: argparse.Namespace) -> None:
    identity = dataset_identity()
    harness.UPSTREAM_DIR = UPSTREAM
    harness.DEFAULT_MODEL = MODEL
    harness.EXPECTED_SPLITS = {k: v["samples"] for k, v in identity["splits"].items()}
    harness.DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT
    harness.ROUTER_SOURCE_FILES = (
        *(REVISION / "mse_router" / name for name in (
            "math_utils.py", "data.py", "model.py", "backbone_model.py", "trainer.py")),
        REVISION / "scripts/run_qwen_mosei_router.py", REVISION / "scripts/chatglm3_simsv2.slurm",
        REVISION / "upstream_snapshot.json", REVISION / "dataset_identity.json", Path(__file__).resolve(),
        *(UPSTREAM / name for name in ("config/config_regression.py", "data/load_data.py",
            "utils/metricsTop.py", "models/ChatGLM3/configuration_chatglm.py",
            "models/ChatGLM3/modeling_chatglm.py", "models/ChatGLM3/tokenization_chatglm.py")),
    )
    harness.parse_args = lambda: args
    harness.build_config = build_config
    harness.validate_files = validate_files
    import mse_router.model as router_model
    from mse_router.backbone_model import BackboneMseRouter
    from mse_router.trainer import write_json
    router_model.QwenMseRouter = BackboneMseRouter
    original_preflight, original_train = harness.preflight, harness.train

    def preflight(parsed):
        result = original_preflight(parsed)
        result["dataset_protocol"] = PROTOCOL
        return result

    def train(parsed):
        checked = json.loads(parsed.preflight_result.read_text())
        if checked.get("dataset_protocol") != PROTOCOL:
            raise ValueError("SIMS v2 preflight protocol mismatch")
        result = original_train(parsed)
        result["dataset_protocol"] = PROTOCOL
        write_json(parsed.output_root / f"seed_{parsed.seed}" / "result.json", result)
        manifest_path = parsed.output_root / f"seed_{parsed.seed}" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["dataset_protocol"] = PROTOCOL
        write_json(manifest_path, manifest)
        return result

    harness.preflight, harness.train = preflight, train


def main():
    args = parse_args()
    if args.mode == "aggregate":
        print(json.dumps(aggregate_results(args.output_root), ensure_ascii=False, indent=2))
        return
    configure_harness(args)
    harness.main()
    if args.mode == "train":
        aggregate_results(args.output_root, require_all=False)


if __name__ == "__main__":
    main()
