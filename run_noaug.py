#!/usr/bin/env python3
"""Portable command wrapper around the unchanged original Router launchers."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ENTRYPOINTS = {
    ("mosei", "qwen"): ("no_augmentation_v1", "run_qwen_mosei_router"),
    ("mosei", "llama2"): ("no_augmentation_v1", "run_llama2_noaug"),
    ("mosei", "glm"): ("no_augmentation_v1", "run_chatglm3_noaug"),
    ("simsv2", "qwen"): ("simsv2_no_augmentation_v1", "run_qwen_simsv2"),
    ("simsv2", "glm"): ("simsv2_no_augmentation_v1", "run_chatglm3_simsv2"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mosei", "simsv2"), required=True)
    parser.add_argument("--backbone", choices=("qwen", "llama2", "glm"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "train", "aggregate"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--check-config", action="store_true",
                        help="Validate launcher arguments, config and provenance on CPU, then exit.")
    args = parser.parse_args()
    key = (args.dataset, args.backbone)
    if key not in ENTRYPOINTS:
        parser.error("The original SIMS v2 release provides Qwen and GLM launchers only.")
    if args.mode == "train" and args.seed is None:
        parser.error("--mode train requires --seed")
    if args.dataset == "mosei" and args.backbone != "qwen" and args.mode == "aggregate":
        parser.error("The original MOSEI GLM/Llama launchers provide preflight and train only.")
    if args.mode != "aggregate" and not args.check_config:
        if not os.environ.get("SLURM_JOB_ID"):
            parser.error("GPU work must run inside a Slurm GPU allocation; use --check-config for CPU validation.")
        import subprocess
        import torch
        state = subprocess.check_output(
            ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True
        )
        allocated = next((s for s in state.split() if s.startswith("AllocTRES=")), "")
        if "JobState=RUNNING" not in state or "gres/gpu=" not in allocated:
            parser.error("A running GPU allocation is required.")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            parser.error("This original single-GPU implementation requires exactly one visible CUDA GPU.")
    # The included upstream files are the actual dependencies used by these runs.
    os.environ["MSE_ADAPTER_ROOT"] = str(ROOT / "MSE-Adapter")
    revision, module_name = ENTRYPOINTS[key]
    source = ROOT / "MSE-Router/revisions" / revision
    sys.path[:0] = [str(source / "scripts"), str(source)]
    runner = importlib.import_module(module_name)
    command = [str(source / "scripts" / (module_name + ".py")), args.mode,
               "--model-path", str(args.model_path.expanduser().resolve()),
               "--dataset-path", str(args.dataset_path.expanduser().resolve()),
               "--training-augmentation", "none", "--router-variant", "full",
               "--skip-robustness"]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.output_root is not None:
        command += ["--output-root", str(args.output_root.expanduser().resolve())]
    if args.preflight_result is not None:
        command += ["--preflight-result", str(args.preflight_result.expanduser().resolve())]
    sys.argv = command
    if args.check_config:
        import json
        from dataclasses import asdict
        if key[0] == "mosei" and key[1] != "qwen":
            runner.configure_harness()
        parsed = runner.parse_args()
        if key[0] == "simsv2":
            runner.configure_harness(parsed)
        harness = getattr(runner, "harness", runner)
        config = harness.build_config(parsed, 4, 4)
        from mse_router.trainer import TrainingSettings
        provenance = harness.source_provenance()
        print(json.dumps({"status": "ok", "dataset": args.dataset,
                          "backbone": args.backbone, "seed": parsed.seed,
                          "feature_dims": config.feature_dims,
                          "output_root": str(parsed.output_root),
                          "settings": asdict(TrainingSettings(training_augmentation="none")),
                          "source_files": len(provenance["files"])}, indent=2))
        return
    runner.main()


if __name__ == "__main__":
    main()
