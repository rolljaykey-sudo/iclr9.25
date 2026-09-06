"""Strict, paired historical-control report for the augmentation experiment."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from .data import augmentation_config


METRICS = (
    "MAE", "Corr", "Has0_acc_2", "Has0_F1_score", "Non0_acc_2",
    "Non0_F1_score", "Mult_acc_5", "Mult_acc_7",
    "invalid_generations", "out_of_range_generations",
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "values": values, "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _completed(root: Path, seed: int) -> tuple[dict, dict]:
    directory = root / f"seed_{seed}"
    result = read_json(directory / "result.json")
    manifest = read_json(directory / "manifest.json")
    if result.get("status") != "ok" or manifest.get("status") != "ok":
        raise ValueError(f"seed {seed} is not complete: {directory}")
    if result.get("seed") != seed or manifest.get("seed") != seed:
        raise ValueError(f"seed identity mismatch: {directory}")
    if result.get("router_variant") != "full" or manifest.get("router_variant") != "full":
        raise ValueError("the paired baseline requires the full V2 router")
    if result["test"].get("condition") != "clean" or result["test"].get("samples") != 4659:
        raise ValueError(f"expected complete clean MOSEI test: {directory}")
    for key in METRICS:
        if not math.isfinite(float(result["test"][key])):
            raise ValueError(f"non-finite {key}: {directory}")
    return result, manifest


def _curve_summary(root: Path, seed: int) -> dict[str, Any]:
    stages = {}
    for stage in ("stage1", "router"):
        history = read_json(root / f"seed_{seed}" / f"{stage}_history.json")
        if not history:
            raise ValueError(f"empty {stage} history for seed {seed}")
        best = min(history, key=lambda item: item["valid"]["MAE"])
        stages[stage] = {
            "epochs": len(history), "best_epoch": best["epoch"],
            "best_valid_mae": best["valid"]["MAE"],
            "last_valid_mae": history[-1]["valid"]["MAE"],
            "first_train_loss": history[0]["train"]["loss"],
            "last_train_loss": history[-1]["train"]["loss"],
            "train_and_validation_seconds": sum(item["elapsed_seconds"] for item in history),
        }
    return stages


def build_comparison(output_root: Path, baseline_root: Path, seeds: list[int], mode: str) -> dict:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("comparison needs an explicit nonempty, unique seed list")
    if output_root.resolve() == baseline_root.resolve():
        raise ValueError("new experiments cannot be their own baseline")
    expected_augmentation = augmentation_config(mode)
    pairs = []
    new_results, old_results = [], []
    for seed in seeds:
        new, new_manifest = _completed(output_root, seed)
        old, old_manifest = _completed(baseline_root, seed)
        if new.get("training_augmentation") != expected_augmentation or new_manifest.get("training_augmentation") != expected_augmentation:
            raise ValueError(f"mixed training augmentation for seed {seed}")
        if new.get("checkpoint_selection") != "best_router_stage_valid_mae":
            raise ValueError("checkpoint selection no longer matches original V2")
        if new.get("run_robustness") is not False or new.get("robustness"):
            raise ValueError("this clean-only experiment must skip robustness evaluation")
        new_settings = dict(new["training_settings"])
        if new_settings.pop("training_augmentation", None) != mode:
            raise ValueError("training settings augmentation mismatch")
        if new_settings != old["training_settings"]:
            raise ValueError(f"other training settings changed for seed {seed}")
        if new_manifest["batching"] != old_manifest["batching"]:
            raise ValueError("baseline batching mismatch")
        if new_manifest["architecture"] != old_manifest["architecture"]:
            raise ValueError("baseline architecture mismatch")
        if new_manifest["split"] != old_manifest["split"]:
            raise ValueError("baseline calibration/data split mismatch")
        if new_manifest["files"] != old_manifest["files"]:
            raise ValueError("baseline dataset/model identity mismatch")
        pairs.append({
            "seed": seed,
            "baseline_result": str(baseline_root / f"seed_{seed}" / "result.json"),
            "new_result": str(output_root / f"seed_{seed}" / "result.json"),
            "baseline": old["test"], "new": new["test"],
            "delta": {key: float(new["test"][key]) - float(old["test"][key]) for key in METRICS},
            "baseline_curves": _curve_summary(baseline_root, seed),
            "new_curves": _curve_summary(output_root, seed),
            "baseline_elapsed_seconds": old["elapsed_seconds"],
            "new_elapsed_seconds": new["elapsed_seconds"],
            "baseline_peak_gpu_memory_bytes": old["peak_gpu_memory_bytes"],
            "new_peak_gpu_memory_bytes": new["peak_gpu_memory_bytes"],
        })
        old_results.append(old)
        new_results.append(new)
    metrics = {
        key: {
            "baseline": summarize([float(result["test"][key]) for result in old_results]),
            "new": summarize([float(result["test"][key]) for result in new_results]),
            "paired_delta": summarize([pair["delta"][key] for pair in pairs]),
        }
        for key in METRICS
    }
    mae_delta = metrics["MAE"]["paired_delta"]["mean"]
    all_improved = all(pair["delta"]["MAE"] < -1e-6 for pair in pairs)
    if all_improved:
        conclusion = "all_seeds_improved"
    elif mae_delta < -1e-6:
        conclusion = "mean_improved_but_not_all_seeds"
    elif mae_delta > 1e-6:
        conclusion = "mean_worsened"
    else:
        conclusion = "no_mean_improvement"
    return {
        "status": "ok", "seeds": seeds, "primary_metric": "MAE",
        "training_augmentation": expected_augmentation,
        "checkpoint_selection": "best_router_stage_valid_mae",
        "metrics": metrics, "pairs": pairs, "conclusion": conclusion,
        "mae_relative_improvement_percent": -100.0 * mae_delta / metrics["MAE"]["baseline"]["mean"],
        "corr_mean_decreased": metrics["Corr"]["paired_delta"]["mean"] < -1e-6,
        "notes": [
            "Historical whole-run times include robustness evaluation; new whole-run times do not.",
            "Stage times include training and validation; compare alongside epochs completed.",
            "All historical/new histories are preserved. No significance claim is made from three seeds.",
            "No performance-dependent seed replacement or test-based checkpoint selection.",
        ],
    }


def render_comparison(summary: dict) -> str:
    labels = {
        "all_seeds_improved": "所有配对 seed 的 MAE 均改善。",
        "mean_improved_but_not_all_seeds": "平均 MAE 改善，但不是所有 seed 都改善。",
        "mean_worsened": "平均 MAE 退化。",
        "no_mean_improvement": "未观察到平均 MAE 改善。",
    }
    lines = [
        "# Qwen V2 无训练扰动对照结果", "", labels[summary["conclusion"]], "",
        "| Seed | 原 MAE | 新 MAE | ΔMAE | 原 Corr | 新 Corr |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pair in summary["pairs"]:
        lines.append(
            f"| {pair['seed']} | {pair['baseline']['MAE']:.4f} | {pair['new']['MAE']:.4f} | "
            f"{pair['delta']['MAE']:+.4f} | {pair['baseline']['Corr']:.4f} | {pair['new']['Corr']:.4f} |"
        )
    lines += ["", "| 指标 | 原均值 ± 标准差 | 新均值 ± 标准差 | 平均变化 |", "|---|---:|---:|---:|"]
    for key, values in summary["metrics"].items():
        old, new = values["baseline"], values["new"]
        lines.append(f"| {key} | {old['mean']:.5f} ± {old['sample_std']:.5f} | {new['mean']:.5f} ± {new['sample_std']:.5f} | {values['paired_delta']['mean']:+.5f} |")
    if summary["corr_mean_decreased"]:
        lines += ["", "注意：平均 Corr 下降，需要与 MAE 的变化一起解释。"]
    lines += ["", "| Seed | 版本 | 阶段 | 轮数 | 最佳验证 MAE | 最后验证 MAE | 首/末训练 loss | 阶段耗时（秒） |", "|---|---|---|---:|---:|---:|---|---:|"]
    for pair in summary["pairs"]:
        for version in ("baseline", "new"):
            for stage, curve in pair[f"{version}_curves"].items():
                lines.append(f"| {pair['seed']} | {version} | {stage} | {curve['epochs']} | {curve['best_valid_mae']:.4f} | {curve['last_valid_mae']:.4f} | {curve['first_train_loss']:.4f} / {curve['last_train_loss']:.4f} | {curve['train_and_validation_seconds']:.1f} |")
    lines += ["", "| Seed | 原/新整轮实验耗时（秒） | 原/新峰值显存（GiB） |", "|---|---|---|"]
    for pair in summary["pairs"]:
        lines.append(f"| {pair['seed']} | {pair['baseline_elapsed_seconds']:.1f} / {pair['new_elapsed_seconds']:.1f} | {pair['baseline_peak_gpu_memory_bytes'] / 2**30:.3f} / {pair['new_peak_gpu_memory_bytes'] / 2**30:.3f} |")
    lines += ["", "旧实验总耗时包含鲁棒性评估，新实验不包含，因此总耗时不能直接解释为训练加速。", "", "训练和验证曲线保留在各 seed 的 history JSON 中。该报告不据三个 seed 宣称统计显著，也不据测试集调整超参数。", ""]
    return "\n".join(lines)
