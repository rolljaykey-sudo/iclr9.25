#!/usr/bin/env python3
"""Preflight, train, and aggregate the Qwen-1.8B MOSEI router experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import resource
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROUTER_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))
ADAPTER_DIR = Path(
    os.environ.get("MSE_ADAPTER_ROOT", ROUTER_ROOT.parent / "MSE-Adapter")
).expanduser().resolve()
UPSTREAM_DIR = ADAPTER_DIR / "MSE-Qwen-1.8B"
DEFAULT_DATASET = Path(
    "/gpfs/work/cpt/jiachenhou23/datasets/MSA/data/CMU-MOSEI/Processed/unaligned_50.pkl"
)
DEFAULT_MODEL = Path("/gpfs/work/cpt/jiachenhou23/models/Qwen/Qwen-1_8B")
DEFAULT_OUTPUT_ROOT = ROUTER_ROOT / "outputs" / "qwen-mosei-router-v2-no-augmentation"
BASELINE_ROOT = ROUTER_ROOT / "outputs" / "qwen-mosei-router-v2"
COMPARISON_SEEDS = (1111, 2222, 3333)
SEEDS = (1111, 2222, 3333, 4444, 5555)
EXPECTED_SPLITS = {"train": 16_326, "valid": 1_871, "test": 4_659}
EXPECTED_DATASET_SIZE = 13_652_131_313
EXPECTED_DATASET_SHA256 = (
    "ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f"
)
ROUTER_SOURCE_FILES = (
    PROJECT_DIR / "mse_router" / "math_utils.py",
    PROJECT_DIR / "mse_router" / "data.py",
    PROJECT_DIR / "mse_router" / "model.py",
    PROJECT_DIR / "mse_router" / "trainer.py",
    PROJECT_DIR / "mse_router" / "comparison.py",
    PROJECT_DIR / "upstream_snapshot.json",
    PROJECT_DIR / "scripts" / "maybe_aggregate.py",
    PROJECT_DIR / "scripts" / "qwen_noaug.slurm",
    UPSTREAM_DIR / "config" / "config_regression.py",
    UPSTREAM_DIR / "data" / "load_data.py",
    UPSTREAM_DIR / "utils" / "metricsTop.py",
    Path(__file__).resolve(),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train", "aggregate"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--training-augmentation", choices=("standard", "none"), default="standard")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument(
        "--preflight-result",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--router-variant",
        choices=(
            "full",
            "no_conflict",
            "no_uncertainty",
            "predictions_only",
            "uncertainty_only",
            "uniform",
        ),
        default="full",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()
    if args.mode == "train" and args.seed is None:
        parser.error("train mode requires --seed")
    if args.seeds is not None and args.mode != "aggregate":
        parser.error("--seeds is only supported in aggregate mode")
    args.seeds = args.seeds or list(COMPARISON_SEEDS)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("duplicate seeds are not allowed")
    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_ROOT if args.training_augmentation == "none" else (
            ROUTER_ROOT / "outputs" / "qwen-mosei-router-v2-augmentation-control"
        )
    args.preflight_result = args.preflight_result or args.output_root / "preflight" / "result.json"
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance() -> dict[str, Any]:
    def source_name(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_DIR))
        except ValueError:
            return str(Path("upstream_adapter") / path.relative_to(ADAPTER_DIR))

    return {
        "files": {
            source_name(path): sha256_file(path)
            for path in ROUTER_SOURCE_FILES
        },
        "git_commit": os.popen(
            f"git -C {ADAPTER_DIR} rev-parse HEAD"
        ).read().strip(),
        "router_root": str(PROJECT_DIR),
        "adapter_root": str(ADAPTER_DIR),
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def environment_info() -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": package_version("transformers"),
        "modelscope": package_version("modelscope"),
        "scikit_learn": package_version("scikit-learn"),
        "scipy": package_version("scipy"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu.name if gpu else None,
        "gpu_total_memory_bytes": gpu.total_memory if gpu else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_qos": os.environ.get("SLURM_JOB_QOS"),
    }


def max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def validate_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("refusing GPU execution outside a Slurm allocation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Slurm allocation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, found {torch.cuda.device_count()}")


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_config(args: argparse.Namespace, microbatch: int, accumulation: int):
    sys.path.insert(0, str(UPSTREAM_DIR))
    from config.config_regression import ConfigRegression

    base = argparse.Namespace(
        is_tune=False,
        tune_mode=False,
        train_mode="regression",
        modelName="cmcm",
        datasetName="mosei",
        root_dataset_dir=str(args.dataset_path.resolve().parent),
        num_workers=args.num_workers,
        model_save_dir=str(args.output_root / "unused"),
        res_save_dir=str(args.output_root / "unused"),
        pretrain_LM=str(args.model_path.resolve()),
        gpu_ids=[0],
    )
    config = ConfigRegression(base).get_config()
    config.modelName = "mse_router"
    config.dataPath = str(args.dataset_path.resolve())
    config.pretrain_LM = str(args.model_path.resolve())
    config.device = torch.device("cuda:0")
    config.batch_size = microbatch
    config.update_epochs = accumulation
    config.router_hidden_size = 2048
    config.router_aux_weight = 0.3
    config.router_variant = args.router_variant
    config.training_augmentation = args.training_augmentation
    config.gradient_checkpointing = True
    config.router_architecture = "natural_text_residual_gated_audio_vision_v2"
    config.diagnostic_prompt = (
        "Predict this modality's sentiment intensity from -3 to +3."
    )
    return config


def validate_files(args: argparse.Namespace, hash_dataset: bool) -> dict[str, Any]:
    dataset = args.dataset_path.resolve()
    model = args.model_path.resolve()
    if not dataset.is_file() or dataset.stat().st_size != EXPECTED_DATASET_SIZE:
        raise RuntimeError(f"unexpected or missing dataset: {dataset}")
    dataset_hash = sha256_file(dataset) if hash_dataset else None
    if dataset_hash is not None and dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"dataset SHA-256 mismatch: {dataset_hash}")
    required_model_files = (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "qwen.tiktoken",
    )
    missing = [name for name in required_model_files if not (model / name).is_file()]
    if missing:
        raise RuntimeError(f"missing Qwen files: {missing}")
    return {
        "dataset": {
            "path": str(dataset),
            "size": dataset.stat().st_size,
            "mtime_ns": dataset.stat().st_mtime_ns,
            "sha256": dataset_hash,
        },
        "model": {
            "path": str(model),
            "files": {
                name: {
                    "size": (model / name).stat().st_size,
                    "mtime_ns": (model / name).stat().st_mtime_ns,
                }
                for name in required_model_files
            },
        },
    }


def load_data(config: Any, microbatch: int):
    from data.load_data import MMDataLoader
    from mse_router.data import build_router_dataloaders

    upstream = MMDataLoader(config)
    for split, expected in EXPECTED_SPLITS.items():
        if len(upstream[split].dataset) != expected:
            raise RuntimeError(
                f"unexpected {split} sample count: {len(upstream[split].dataset)}"
            )
    return build_router_dataloaders(
        upstream, microbatch, config.num_workers, split_seed=20260903
    )


def _trainable_summary(model: torch.nn.Module) -> dict[str, Any]:
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "trainable_parameters": sum(parameter.numel() for _, parameter in named),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_tensors": len(named),
        "names": [name for name, _ in named],
    }


def _one_backward(
    model: Any,
    config: Any,
    loader: torch.utils.data.DataLoader,
    accumulation: int,
    stage: str = "stage1",
) -> dict[str, Any]:
    from mse_router.data import prepare_training_batch, move_batch
    from mse_router.trainer import RouterTrainer, TrainingSettings

    model.zero_grad(set_to_none=True)
    model.set_stage(stage)
    RouterTrainer._set_training_mode(model, stage)
    batch = move_batch(next(iter(loader)), config.device)
    augmented, presence = prepare_training_batch(batch, config.training_augmentation)
    labels = augmented["labels"]["M"].view(-1)
    text = (augmented["text"], augmented["text_lengths"])
    audio = (augmented["audio"], augmented["audio_lengths"])
    vision = (augmented["vision"], augmented["vision_lengths"])
    trainer = RouterTrainer.__new__(RouterTrainer)
    trainer.settings = TrainingSettings(training_augmentation=config.training_augmentation)
    optimizer = trainer._optimizer(model, stage)
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    loss_scale_attempts = []
    for attempt in range(1, 17):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(
                labels, text, audio, vision, presence=presence, stage=stage
            )
        if not bool(torch.isfinite(output["Loss"]).item()):
            raise RuntimeError(f"non-finite preflight loss: {output['Loss'].item()}")
        scale_before = float(scaler.get_scale())
        scaler.scale(output["Loss"] / accumulation).backward()
        scaler.unscale_(optimizer)
        gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        if not gradients:
            raise RuntimeError("preflight produced no trainable gradients")
        nonfinite = [
            name
            for name, gradient in gradients.items()
            if not bool(torch.isfinite(gradient).all().item())
        ]
        categories = {
            "adapter": any(
                not name.startswith(("ordinal_head.", "router."))
                for name in gradients
            ),
            "ordinal_head": any(
                name.startswith("ordinal_head.") for name in gradients
            ),
            "router": any(name.startswith("router.") for name in gradients),
        }
        if not nonfinite:
            norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                trainer.settings.gradient_clip,
            )
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("non-finite preflight gradient norm")
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        loss_scale_attempts.append(
            {
                "attempt": attempt,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "nonfinite_gradient_names": nonfinite,
            }
        )
        if not nonfinite:
            required_categories = all(categories.values()) if stage == "stage1" else (
                categories["router"] and not categories["adapter"] and not categories["ordinal_head"]
            )
            if not required_categories:
                raise RuntimeError(f"missing gradient categories: {categories}")
            break
        if scale_after >= scale_before:
            raise RuntimeError(
                "non-finite gradients did not trigger a GradScaler reduction: "
                f"{nonfinite}"
            )
    else:
        raise RuntimeError("dynamic loss scaling did not stabilize within 16 attempts")
    torch.cuda.synchronize()
    return {
        "stage": stage,
        "training_augmentation": config.training_augmentation,
        "presence_counts": presence.sum(dim=0).tolist(),
        "batch_size": int(labels.shape[0]),
        "effective_batch_size": int(labels.shape[0] * accumulation),
        "loss": float(output["Loss"].detach()),
        "generation_loss": float(output["GenerationLoss"].detach()),
        "auxiliary_loss": float(output["AuxiliaryLoss"].detach()),
        "gradient_tensors": len(gradients),
        "gradient_categories": categories,
        "loss_scale_attempts": loss_scale_attempts,
        "elapsed_seconds": round(time.time() - started, 3),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


@torch.no_grad()
def _one_generate(model: Any, config: Any, loader: torch.utils.data.DataLoader):
    from mse_router.data import move_batch

    model.eval()
    batch = move_batch(next(iter(loader)), config.device)
    text = (batch["text"][:1], batch["text_lengths"][:1])
    audio = (batch["audio"][:1], batch["audio_lengths"][:1])
    vision = (batch["vision"][:1], batch["vision_lengths"][:1])
    with torch.cuda.amp.autocast(dtype=torch.float16):
        values, diagnostics = model.generate(
            text, audio, vision, return_diagnostics=True
        )
    return {
        "values": values,
        "raw_responses": diagnostics["raw_responses"],
        "weights": diagnostics["weights"].tolist(),
        "invalid_count": diagnostics["invalid_count"],
        "out_of_range_count": diagnostics["out_of_range_count"],
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    validate_allocation()
    setup_seed(1111)
    files = validate_files(args, hash_dataset=True)
    attempts = []
    selected = None
    model = None
    split = None
    loaders = None
    # Do not silently change batching in a matched augmentation ablation.
    for microbatch, accumulation in ((4, 4),):
        try:
            config = build_config(args, microbatch, accumulation)
            if loaders is None or loaders["train"].batch_size != microbatch:
                loaders, split = load_data(config, microbatch)
            if model is None:
                from mse_router.model import QwenMseRouter

                model = QwenMseRouter(config).to(config.device)
            backward = _one_backward(model, config, loaders["train"], accumulation)
            attempts.append(
                {
                    "microbatch": microbatch,
                    "accumulation": accumulation,
                    "status": "ok",
                    **backward,
                }
            )
            selected = {
                "microbatch": microbatch,
                "accumulation": accumulation,
                "effective_batch": 16,
            }
            break
        except torch.cuda.OutOfMemoryError as error:
            attempts.append(
                {
                    "microbatch": microbatch,
                    "accumulation": accumulation,
                    "status": "oom",
                    "error": str(error),
                }
            )
            if model is not None:
                model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    if selected is None or model is None or loaders is None or split is None:
        raise RuntimeError(f"all microbatch preflight attempts failed: {attempts}")
    generated = _one_generate(model, config, loaders["valid"])
    trainable = _trainable_summary(model)
    router_backward = _one_backward(model, config, loaders["train"], 4, stage="router")
    if any(parameter.requires_grad for parameter in model.llm.parameters()):
        raise RuntimeError("Qwen is not completely frozen")
    from mse_router.data import augmentation_config
    from mse_router.trainer import TrainingSettings
    return {
        "status": "ok",
        "mode": "preflight",
        "architecture": "natural_text_residual_gated_audio_vision_v2",
        "selected_batching": selected,
        "router_variant": args.router_variant,
        "training_augmentation": augmentation_config(args.training_augmentation),
        "training_settings": asdict(TrainingSettings(training_augmentation=args.training_augmentation)),
        "router_backward": router_backward,
        "attempts": attempts,
        "generation_smoke": generated,
        "trainable": trainable,
        "split": {
            "optimization_samples": len(split.train_indices),
            "calibration_samples": len(split.calibration_indices),
            "optimization_groups": split.train_groups,
            "calibration_groups": split.calibration_groups,
            "fingerprint": split.fingerprint,
            "seed": 20260903,
        },
        "files": files,
        "provenance": source_provenance(),
        "environment": environment_info(),
        "max_rss_bytes": max_rss_bytes(),
    }


def verify_preflight(args: argparse.Namespace) -> dict[str, Any]:
    from mse_router.data import augmentation_config
    from mse_router.trainer import TrainingSettings
    with args.preflight_result.resolve().open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") != "ok":
        raise RuntimeError(f"preflight did not succeed: {args.preflight_result}")
    if result.get("training_augmentation") != augmentation_config(args.training_augmentation):
        raise RuntimeError("preflight training augmentation mismatch (legacy preflight is not accepted)")
    if result.get("training_settings") != asdict(TrainingSettings(training_augmentation=args.training_augmentation)):
        raise RuntimeError("preflight training settings mismatch")
    if result.get("router_variant") != args.router_variant:
        raise RuntimeError("preflight router variant mismatch")
    if result.get("selected_batching") != {"microbatch": 4, "accumulation": 4, "effective_batch": 16}:
        raise RuntimeError("matched experiment requires microbatch 4 x accumulation 4")
    if result["provenance"] != source_provenance():
        raise RuntimeError("router source changed after preflight")
    current = validate_files(args, hash_dataset=False)
    previous = result["files"]
    for section in ("dataset",):
        for key in ("path", "size", "mtime_ns"):
            if current[section][key] != previous[section][key]:
                raise RuntimeError(f"{section} changed after preflight ({key})")
    for name, record in previous["model"]["files"].items():
        if current["model"]["files"][name] != record:
            raise RuntimeError(f"Qwen file changed after preflight: {name}")
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_allocation()
    preflight_result = verify_preflight(args)
    setup_seed(args.seed)
    batching = preflight_result["selected_batching"]
    config = build_config(
        args, batching["microbatch"], batching["accumulation"]
    )
    loaders, split = load_data(config, batching["microbatch"])
    if split.fingerprint != preflight_result["split"]["fingerprint"]:
        raise RuntimeError("calibration split differs from preflight")
    from mse_router.model import QwenMseRouter
    from mse_router.trainer import RouterTrainer, TrainingSettings, write_json
    from mse_router.data import augmentation_config

    seed_dir = args.output_root.resolve() / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    settings = TrainingSettings(
        accumulation_steps=batching["accumulation"],
        training_augmentation=args.training_augmentation,
    )
    manifest = {
        "status": "running",
        "seed": args.seed,
        "router_variant": args.router_variant,
        "architecture": config.router_architecture,
        "batching": batching,
        "training_augmentation": augmentation_config(args.training_augmentation),
        "training_settings": asdict(settings),
        "run_robustness": not args.skip_robustness,
        "checkpoint_selection": "best_router_stage_valid_mae",
        "split": preflight_result["split"],
        "files": preflight_result["files"],
        "provenance": preflight_result["provenance"],
        "environment": environment_info(),
        "started_unix": time.time(),
    }
    write_json(seed_dir / "manifest.json", manifest)
    model = QwenMseRouter(config).to(config.device)
    trainer = RouterTrainer(config, seed_dir, settings)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.fit(
        model, loaders, run_robustness=not args.skip_robustness
    )
    torch.cuda.synchronize()
    result.update(
        {
            "seed": args.seed,
            "training_augmentation": augmentation_config(args.training_augmentation),
            "run_robustness": not args.skip_robustness,
            "checkpoint_selection": "best_router_stage_valid_mae",
            "router_variant": args.router_variant,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "max_rss_bytes": max_rss_bytes(),
            "environment": environment_info(),
            "provenance": source_provenance(),
        }
    )
    write_json(seed_dir / "result.json", result)
    manifest["status"] = "ok"
    manifest["finished_unix"] = time.time()
    manifest["result"] = str(seed_dir / "result.json")
    write_json(seed_dir / "manifest.json", manifest)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    from mse_router.comparison import build_comparison, render_comparison
    from mse_router.trainer import write_json

    summary = build_comparison(
        args.output_root.resolve(), args.baseline_root.resolve(),
        args.seeds, args.training_augmentation,
    )
    summary["provenance"] = source_provenance()
    summary["report"] = str(args.output_root.resolve() / "three_seed_comparison.md")
    write_json(args.output_root.resolve() / "three_seed_comparison.json", summary)
    (args.output_root.resolve() / "three_seed_comparison.md").write_text(
        render_comparison(summary), encoding="utf-8"
    )
    return summary


def claim_run_directory(args: argparse.Namespace) -> Path:
    """Reserve before logging/exception handling, so collisions cannot overwrite."""
    root = args.output_root.resolve()
    if not root.is_relative_to(ROUTER_ROOT / "outputs"):
        raise ValueError("experiment outputs must remain under MSE-Router/outputs")
    if root == BASELINE_ROOT or root.is_relative_to(BASELINE_ROOT):
        raise ValueError("the historical baseline is read-only")
    if args.mode == "aggregate":
        for name in ("three_seed_comparison.json", "three_seed_comparison.md"):
            if (root / name).exists():
                raise FileExistsError(f"refusing to overwrite {root / name}")
    name = f"seed_{args.seed}" if args.mode == "train" else args.mode
    directory = root / name
    root.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    return directory


def configure_logging(args: argparse.Namespace) -> None:
    if args.mode == "train":
        directory = args.output_root.resolve() / f"seed_{args.seed}"
    else:
        directory = args.output_root.resolve() / args.mode
    directory.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(directory / "run.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(PROJECT_DIR))
    claim_run_directory(args)
    configure_logging(args)
    if args.mode == "preflight":
        output_path = args.output_root.resolve() / "preflight" / "result.json"
    elif args.mode == "train":
        output_path = args.output_root.resolve() / f"seed_{args.seed}" / "result.json"
    else:
        output_path = args.output_root.resolve() / "three_seed_comparison.json"
    started = time.time()
    try:
        if args.mode == "preflight":
            result = preflight(args)
        elif args.mode == "train":
            result = train(args)
        else:
            result = aggregate(args)
        if args.mode == "preflight":
            from mse_router.trainer import write_json

            write_json(output_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        failure = {
            "status": "error",
            "mode": args.mode,
            "seed": args.seed,
            "training_augmentation": args.training_augmentation,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "environment": environment_info(),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        raise


if __name__ == "__main__":
    main()
