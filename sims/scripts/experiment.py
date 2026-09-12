#!/usr/bin/env python3
# 实验主入口：校验数据与环境、执行预检、按种子训练并汇总指标。
"""ChatGLM3 SIMS 的预检、训练与种子结果汇总流程。"""

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
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from mse_router.model import ARCHITECTURE, CONFLICT_METRIC
ADAPTER_DIR = Path(
    os.environ.get("MSE_ADAPTER_ROOT", PROJECT_DIR.parent / "external" / "MSE-Adapter")
).expanduser().resolve()
UPSTREAM_DIR = ADAPTER_DIR / "MSE-ChatGLM3-6B"
DEFAULT_DATASET = PROJECT_DIR / "data" / "dataset.pkl"
DEFAULT_MODEL = Path(os.environ.get("CHATGLM_MODEL_PATH", PROJECT_DIR.parent / "models" / "chatglm3-6b-base"))
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs"
SEEDS = (1111, 1113, 1115)
EXPECTED_SPLITS = {}
EXPECTED_DATASET_SIZE = 0
EXPECTED_DATASET_SHA256 = ""
CONFIG_PATH = None
ROUTER_SOURCE_FILES = (
    PROJECT_DIR / "mse_router" / "math_utils.py",
    PROJECT_DIR / "mse_router" / "sequence.py",
    PROJECT_DIR / "mse_router" / "data.py",
    PROJECT_DIR / "mse_router" / "model.py",
    PROJECT_DIR / "mse_router" / "trainer.py",
    Path(__file__).resolve(),
)


# 解析预检、训练或汇总模式及路径参数；训练模式必须明确随机种子。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train", "aggregate"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--preflight-result",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "preflight" / "result.json",
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
    args = parser.parse_args()
    if args.mode == "train" and args.seed is None:
        parser.error("train mode requires --seed")
    return args


# 按块读取文件计算 SHA-256，避免把大数据文件一次性载入内存。
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# 记录参与实验的源码散列、上游提交号及项目路径，用于预检一致性校验。
def source_provenance() -> dict[str, Any]:
    # 用项目相对路径标识本地源码，为上游适配器文件生成单独的来源路径。
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
        "git_commit": subprocess.run(
            ["git", "-C", str(ADAPTER_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "router_root": str(PROJECT_DIR),
        "adapter_root": str(ADAPTER_DIR),
    }


# 查询已安装依赖版本；未安装时返回明确的缺失标记。
def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


# 记录 Python、依赖、可见 GPU 和 Slurm 作业信息，便于复查实验环境。
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


# 将进程最大常驻内存从 Linux 的 KiB 单位换算为字节。
def max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


# 检查 Slurm 作业标识、CUDA 可用性及单张可见 GPU 的要求。
def validate_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("refusing GPU execution outside a Slurm allocation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Slurm allocation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, found {torch.cuda.device_count()}")


# 统一 Python、NumPy 和 PyTorch 随机种子，并设置 cuDNN 的确定性选项。
def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 在上游配置基础上设置当前实验的数据、骨干维度、批量与路由参数。
def build_config(args: argparse.Namespace, microbatch: int, accumulation: int):
    sys.path.insert(0, str(UPSTREAM_DIR))
    from config.config_regression import ConfigRegression

    base = argparse.Namespace(
        is_tune=False,
        tune_mode=False,
        train_mode="regression",
        modelName="cmcm",
        datasetName="mosi",
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
    config.router_hidden_size = 4096
    config.router_aux_weight = 0.3
    config.router_variant = args.router_variant
    config.gradient_checkpointing = True
    config.router_architecture = ARCHITECTURE
    config.router_conflict_metric = CONFLICT_METRIC
    config.diagnostic_prompt = (
        "Predict this modality's sentiment intensity from -3 to +3."
    )
    return config


# 数据集入口会装配 ChatGLM3 的文件校验函数。
def validate_files(args, hash_dataset):
    raise RuntimeError("请通过本目录的 run_chatglm3 数据集入口执行。")


# 加载上游数据，核对各划分样本数，直接用完整训练集构建 DataLoader。
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
        upstream, microbatch, config.num_workers
    )


# 统计可训练参数总数、张量数量及名称，用于核对冻结范围。
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


# 预检一次真实反向更新；允许动态损失缩放降低倍率，并核对各训练模块均有梯度。
def _one_backward(
    model: Any,
    config: Any,
    loader: torch.utils.data.DataLoader,
    accumulation: int,
) -> dict[str, Any]:
    from mse_router.data import move_batch

    model.set_stage("stage1")
    model.train()
    batch = move_batch(next(iter(loader)), config.device)
    # 预检和正式训练使用同一无扰动输入路径，三模态全部存在。
    labels = batch["labels"]["M"].view(-1)
    presence = torch.ones(labels.shape[0], 3, device=labels.device)
    text = (batch["text"], batch["text_lengths"])
    audio = (batch["audio"], batch["audio_lengths"])
    vision = (batch["vision"], batch["vision_lengths"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        eps=1e-4,
        weight_decay=0.01,
    )
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    loss_scale_attempts = []
    for attempt in range(1, 17):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(
                labels, text, audio, vision, presence=presence, stage="stage1"
            )
        if not bool(torch.isfinite(output["Loss"]).item()):
            raise RuntimeError(f"non-finite preflight loss: {output['Loss'].item()}")
        scale_before = float(scaler.get_scale())
        scaler.scale(output["Loss"] / accumulation).backward()
        # 先取消损失缩放，再裁剪真实梯度；溢出时由 GradScaler 跳过参数更新。
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
            **{
                prefix: any(name.startswith(prefix) and bool(gradient.float().abs().max().item() > 0)
                            for name, gradient in gradients.items())
                for prefix in ("text_pool.", "text_projection.", "text_adapter.", "text_modality_embedding")
            },
        }
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
            if not all(categories.values()):
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


# 用验证集的一个样本检查生成路径，并记录原始输出和路由权重。
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


# 校验执行环境与文件，尝试维持有效批量为 16 的不同微批组合，完成反向和生成预检。
def preflight(args: argparse.Namespace) -> dict[str, Any]:
    from mse_router.data import INPUT_POLICY

    validate_allocation()
    setup_seed(1111)
    files = validate_files(args, hash_dataset=True)
    attempts = []
    selected = None
    model = None
    split = None
    loaders = None
    # 显存不足时减小微批并增加累积次数，保持有效批量 16 不变。
    for microbatch, accumulation in ((4, 4), (2, 8), (1, 16)):
        try:
            config = build_config(args, microbatch, accumulation)
            if loaders is None or loaders["train"].batch_size != microbatch:
                loaders, split = load_data(config, microbatch)
            if model is None:
                from mse_router.model import ChatGLMMseRouter

                model = ChatGLMMseRouter(config).to(config.device)
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
    if any(parameter.requires_grad for parameter in model.llm.parameters()):
        raise RuntimeError("ChatGLM3 is not completely frozen")
    return {
        "status": "ok",
        "mode": "preflight",
        "architecture": ARCHITECTURE,
        "conflict_metric": CONFLICT_METRIC,
        "selected_batching": selected,
        "attempts": attempts,
        "generation_smoke": generated,
        "trainable": trainable,
        "split": {
            "optimization_samples": len(split.train_indices),
            "calibration_samples": 0,
            "optimization_groups": split.train_groups,
            "calibration_groups": 0,
            "fingerprint": split.fingerprint,
            "full_training_split": True,
        },
        "input_policy": dict(INPUT_POLICY),
        "files": files,
        "provenance": source_provenance(),
        "environment": environment_info(),
        "max_rss_bytes": max_rss_bytes(),
    }


# 在训练前核对预检状态、源码散列及数据和模型文件元数据，防止沿用失效预检。
def verify_preflight(args: argparse.Namespace) -> dict[str, Any]:
    with args.preflight_result.resolve().open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") != "ok":
        raise RuntimeError(f"preflight did not succeed: {args.preflight_result}")
    # 注释也会改变源码散列，因此新副本运行训练前需要重新生成自己的预检记录。
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
            raise RuntimeError(f"ChatGLM3 file changed after preflight: {name}")
    return result


# 按指定种子重建数据和模型，校验划分指纹，执行训练并保存运行清单及结果。
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
        raise RuntimeError("full training inventory differs from preflight")
    from mse_router.model import ChatGLMMseRouter
    from mse_router.trainer import RouterTrainer, TrainingSettings, write_json

    seed_dir = args.output_root.resolve() / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "seed": args.seed,
        "router_variant": args.router_variant,
        "architecture": config.router_architecture,
        "conflict_metric": CONFLICT_METRIC,
        "batching": batching,
        "split": preflight_result["split"],
        "input_policy": preflight_result["input_policy"],
        "files": preflight_result["files"],
        "provenance": preflight_result["provenance"],
        "environment": environment_info(),
        "started_unix": time.time(),
    }
    write_json(seed_dir / "manifest.json", manifest)
    model = ChatGLMMseRouter(config).to(config.device)
    settings = TrainingSettings(accumulation_steps=batching["accumulation"])
    trainer = RouterTrainer(config, seed_dir, settings)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.fit(model, loaders)
    torch.cuda.synchronize()
    result.update(
        {
            "seed": args.seed,
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


# 读取约定种子的已完成结果，对数值指标计算均值和样本标准差。
def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    results = []
    for seed in SEEDS:
        path = args.output_root.resolve() / f"seed_{seed}" / "result.json"
        if not path.exists():
            path = args.output_root.resolve() / f"task_{seed}" / f"seed_{seed}" / "result.json"
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("status") != "ok":
            raise RuntimeError(f"seed {seed} is not complete: {path}")
        if result.get("selection", {}).get("metric") != "MAE" or result.get("selection", {}).get("direction") != "min":
            raise RuntimeError(f"seed {seed} was not selected by minimum MAE")
        results.append(result)
    metric_names = list(results[0]["test"].keys())
    numeric_metrics = {}
    for metric in metric_names:
        values = [result["test"].get(metric) for result in results]
        if all(isinstance(value, (int, float)) for value in values):
            numeric_metrics[metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "values": values,
            }
    summary = {
        "status": "ok",
        "seeds": list(SEEDS),
        "primary_metric": "MAE",
        "metrics": numeric_metrics,
        "selection_metric_mean": numeric_metrics["MAE"]["mean"],
    }
    output = args.output_root.resolve() / "seed_summary.json"
    from mse_router.trainer import write_json

    write_json(output, summary)
    return summary


# 按当前运行模式和随机种子创建日志目录，同时写控制台和 run.log。
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


# 分派预检、训练或汇总流程，并统一记录成功结果与异常诊断。
def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(PROJECT_DIR))
    configure_logging(args)
    if args.mode == "preflight":
        output_path = args.output_root.resolve() / "preflight" / "result.json"
    elif args.mode == "train":
        output_path = args.output_root.resolve() / f"seed_{args.seed}" / "result.json"
    else:
        output_path = args.output_root.resolve() / "seed_summary.json"
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
