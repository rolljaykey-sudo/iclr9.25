#!/usr/bin/env python3
"""Evaluate saved Stage1 checkpoints without further training or calibration."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "outputs" / "qwen-mosei-router-v2-no-augmentation"
SEEDS = (1111, 2222, 3333, 4444, 5555)
METRICS = (
    "MAE", "Corr", "Has0_acc_2", "Has0_F1_score", "Non0_acc_2",
    "Non0_F1_score", "Mult_acc_5", "Mult_acc_7",
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def aggregate(args: argparse.Namespace) -> None:
    results = []
    for seed in SEEDS:
        result = json.loads((args.output_root / f"seed_{seed}" / "result.json").read_text())
        if (
            result["status"] != "ok"
            or result["seed"] != seed
            or result["fusion"] != args.fusion
            or result["test"]["samples"] != 4659
            or result["test"]["condition"] != "clean"
        ):
            raise RuntimeError(f"incompatible result for seed {seed}")
        results.append(result)
    summary = {
        "status": "ok", "seeds": list(SEEDS), "n": len(results),
        "split": "test", "condition": "clean", "samples_per_seed": 4659,
        "checkpoint_selection": "best_stage1_valid_mae",
        "fusion": args.fusion, "temperature_calibration": False,
        "second_stage_training": False, "std_ddof": 1,
        "metrics": {},
        "result_paths": [str(args.output_root / f"seed_{s}" / "result.json") for s in SEEDS],
    }
    rows = [
        "# Qwen V2 no-augmentation Stage1 checkpoint test results", "",
        f"Fusion: {args.fusion}. No temperature calibration or second-stage training.",
        "The source Stage1 checkpoints were jointly trained with a router.",
        "Clean test set: 4659 samples per seed. Sample standard deviation: ddof=1.", "",
        "| Metric | 1111 | 2222 | 3333 | 4444 | 5555 | Mean ± sample std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        values = [result["test"][metric] for result in results]
        mean, std = statistics.mean(values), statistics.stdev(values)
        summary["metrics"][metric] = {"values": values, "mean": mean, "std": std}
        rows.append("| " + " | ".join(
            [metric] + [f"{value:.5f}" for value in values] + [f"{mean:.5f} ± {std:.5f}"]
        ) + " |")
    write_json(args.output_root / "five_seed_summary.json", summary)
    (args.output_root / "five_seed_summary.md").write_text("\n".join(rows) + "\n")
    print(json.dumps(summary, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    import torch
    import run_qwen_mosei_router as runner
    from mse_router.model import QwenMseRouter
    from mse_router.trainer import RouterTrainer, TrainingSettings

    runner.validate_allocation()
    source = SOURCE_ROOT / f"seed_{args.seed}"
    manifest = json.loads((source / "manifest.json").read_text())
    checkpoint_path = source / "checkpoints" / "stage1.pt"
    history = json.loads((source / "stage1_history.json").read_text())
    expected = min(history, key=lambda item: item["valid"]["MAE"])
    output = args.output_root / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
        force=True,
    )
    config_args = argparse.Namespace(
        dataset_path=Path(manifest["files"]["dataset"]["path"]),
        model_path=Path(manifest["files"]["model"]["path"]),
        output_root=output, num_workers=0, router_variant="full",
        training_augmentation="none", preflight_result=SOURCE_ROOT / "preflight" / "result.json",
    )
    preflight = runner.verify_preflight(config_args)
    if manifest["provenance"] != preflight["provenance"]:
        raise RuntimeError("source checkpoint provenance differs from verified preflight")
    runner.setup_seed(args.seed)
    config = runner.build_config(config_args, microbatch=4, accumulation=4)
    started = time.time()
    checkpoint_before = checkpoint_path.stat()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if (
        checkpoint["stage"] != "stage1"
        or checkpoint["epoch"] != expected["epoch"]
        or checkpoint["router_variant"] != "full"
        or checkpoint["valid_metrics"] != expected["valid"]
    ):
        raise RuntimeError("checkpoint does not match the recorded best Stage1 validation epoch")
    checkpoint_sha256 = runner.sha256_file(checkpoint_path)
    checkpoint_after = checkpoint_path.stat()
    if (checkpoint_before.st_size, checkpoint_before.st_mtime_ns) != (
        checkpoint_after.st_size, checkpoint_after.st_mtime_ns
    ):
        raise RuntimeError("source checkpoint changed while being read")
    record = {
        "status": "running", "seed": args.seed, "fusion": args.fusion,
        "source_checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_selection": "best_stage1_valid_mae",
        "checkpoint_valid_metrics": checkpoint["valid_metrics"],
        "source_stage1_includes_trained_router": True,
        "temperature_calibration": False, "second_stage_training": False,
        "batching": {"microbatch": 4, "num_workers": 0},
        "environment": runner.environment_info(), "source_provenance": preflight["provenance"],
        "evaluator_sha256": runner.sha256_file(Path(__file__)),
        "started_unix": started,
    }
    write_json(output / "manifest.json", record)
    loaders, split = runner.load_data(config, microbatch=4)
    if split.fingerprint != preflight["split"]["fingerprint"]:
        raise RuntimeError("dataset split fingerprint differs from training")
    model = QwenMseRouter(config).to(config.device)
    model.load_experiment_state_dict(checkpoint["model"])
    if not torch.equal(model.temperatures.detach().cpu(), torch.ones(3)):
        raise RuntimeError("Stage1 checkpoint has unexpected calibrated temperatures")
    model.set_stage("eval")
    model.eval()
    trainer = RouterTrainer(config, output, TrainingSettings(training_augmentation="none"))
    if args.validate_restore:
        restored = trainer.evaluate(model, loaders["valid"], mode="stage1-restore-valid")
        differences = {
            key: {"expected": expected["valid"][key], "restored": restored[key]}
            for key in METRICS
            if abs(restored[key] - expected["valid"][key]) > 1e-5
        }
        write_json(output / "restore_validation.json", {
            "status": "ok" if not differences else "mismatch",
            "expected": expected["valid"], "restored": restored, "differences": differences,
        })
        if differences:
            raise RuntimeError(f"restored Stage1 checkpoint validation differs: {differences}")
    if args.fusion == "uniform":
        model.router_variant = "uniform"
    metrics = trainer.evaluate(model, loaders["test"], mode="stage1-test", condition="clean")
    if metrics["samples"] != 4659:
        raise RuntimeError("test set sample count differs from the previous full-model evaluation")
    record.update({
        "status": "ok", "test": metrics, "finished_unix": time.time(),
        "elapsed_seconds": round(time.time() - started, 3),
        "temperatures": model.temperatures.detach().cpu().tolist(),
    })
    write_json(output / "result.json", record)
    write_json(output / "manifest.json", record)
    print(json.dumps(record, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate", "aggregate"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--fusion", choices=("checkpoint", "uniform"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-restore", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if args.mode == "evaluate":
        if args.seed is None:
            parser.error("evaluate requires --seed")
        evaluate(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
