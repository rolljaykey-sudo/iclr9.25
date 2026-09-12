#!/usr/bin/env python3
# 骨干入口：为统一实验流程装配 Llama2 或 ChatGLM3 的模型、数据和配置接口。
"""Run the ordered-text Wasserstein MOSEI Router V4 with Llama2 or ChatGLM3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import run_qwen_mosei_router as harness


PROJECT_DIR = Path(__file__).resolve().parents[1]
ADAPTER_DIR = harness.ADAPTER_DIR
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
BACKBONES = {
    "llama2": {
        "upstream": ADAPTER_DIR / "MSE-Llama2-7B",
        "model": Path("/gpfs/work/cpt/jiachenhou23/models/Meta/Llama-2-7b-hf"),
        "output": PROJECT_DIR / "outputs" / "llama2-mosei-router-v4-conflict",
        "required": (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ),
    },
    "chatglm3": {
        "upstream": ADAPTER_DIR / "MSE-ChatGLM3-6B",
        "model": Path(
            "/gpfs/work/cpt/jiachenhou23/models/THUDM/chatglm3-6b-base"
        ),
        "output": PROJECT_DIR / "outputs" / "chatglm3-mosei-router-v4-conflict",
        "required": (
            "config.json",
            "pytorch_model.bin.index.json",
            *(f"pytorch_model-{part:05d}-of-00007.bin" for part in range(1, 8)),
            "tokenizer.model",
            "tokenizer_config.json",
        ),
    },
}


# 先提取骨干名称，把其余命令行参数留给通用实验入口解析。
def extract_backbone() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backbone", choices=tuple(BACKBONES), required=True)
    parsed, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return str(parsed.backbone)


# 为通用入口替换骨干路径、配置构造和文件检查，并在当前进程绑定模型实现。
def configure_harness(backbone: str) -> None:
    spec = BACKBONES[backbone]
    upstream = Path(spec["upstream"])
    harness.UPSTREAM_DIR = upstream
    harness.DEFAULT_MODEL = Path(spec["model"])
    harness.DEFAULT_OUTPUT_ROOT = Path(spec["output"])
    backbone_sources = (
        upstream / "config" / "config_regression.py",
        upstream / "data" / "load_data.py",
        upstream / "utils" / "metricsTop.py",
    )
    if backbone == "chatglm3":
        backbone_sources += (
            upstream / "models" / "ChatGLM3" / "configuration_chatglm.py",
            upstream / "models" / "ChatGLM3" / "modeling_chatglm.py",
            upstream / "models" / "ChatGLM3" / "tokenization_chatglm.py",
        )
    harness.ROUTER_SOURCE_FILES = (
        PROJECT_DIR / "mse_router" / "math_utils.py",
        PROJECT_DIR / "mse_router" / "sequence.py",
        PROJECT_DIR / "mse_router" / "data.py",
        PROJECT_DIR / "mse_router" / "model.py",
        PROJECT_DIR / "mse_router" / "backbone_model.py",
        PROJECT_DIR / "mse_router" / "trainer.py",
        PROJECT_DIR / "scripts" / "run_qwen_mosei_router.py",
        Path(__file__).resolve(),
        *backbone_sources,
    )

    original_build_config = harness.build_config

    # 在上游配置基础上设置当前实验的数据、骨干维度、批量与路由参数。
    def build_config(
        args: argparse.Namespace, microbatch: int, accumulation: int
    ) -> Any:
        config = original_build_config(args, microbatch, accumulation)
        config.router_hidden_size = 4096
        config.router_backbone = backbone
        config.router_upstream_dir = str(upstream.resolve())
        return config

    # 检查数据文件大小、可选 SHA-256 及骨干文件完整性，并记录文件元数据。
    def validate_files(
        args: argparse.Namespace, hash_dataset: bool
    ) -> dict[str, Any]:
        dataset = args.dataset_path.resolve()
        model = args.model_path.resolve()
        if not dataset.is_file() or dataset.stat().st_size != harness.EXPECTED_DATASET_SIZE:
            raise RuntimeError(f"unexpected or missing dataset: {dataset}")
        dataset_hash = harness.sha256_file(dataset) if hash_dataset else None
        if (
            dataset_hash is not None
            and dataset_hash != harness.EXPECTED_DATASET_SHA256
        ):
            raise RuntimeError(f"dataset SHA-256 mismatch: {dataset_hash}")
        required = tuple(spec["required"])
        missing = [name for name in required if not (model / name).is_file()]
        if missing:
            raise RuntimeError(f"missing {backbone} files: {missing}")
        return {
            "dataset": {
                "path": str(dataset),
                "size": dataset.stat().st_size,
                "mtime_ns": dataset.stat().st_mtime_ns,
                "sha256": dataset_hash,
            },
            "model": {
                "backbone": backbone,
                "path": str(model),
                "files": {
                    name: {
                        "size": (model / name).stat().st_size,
                        "mtime_ns": (model / name).stat().st_mtime_ns,
                    }
                    for name in required
                },
            },
        }

    harness.build_config = build_config
    harness.validate_files = validate_files

    # 通用入口在执行时导入此模型符号；
    # 当前进程中的替换不会改变其他已运行 Qwen 作业的模型绑定。
    import mse_router.model as router_model
    from mse_router.backbone_model import BackboneMseRouter

    router_model.QwenMseRouter = BackboneMseRouter


# 提取骨干选择、完成入口配置，然后调用通用实验流程。
def main() -> None:
    backbone = extract_backbone()
    configure_harness(backbone)
    harness.main()


if __name__ == "__main__":
    main()
