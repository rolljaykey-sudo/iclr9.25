#!/usr/bin/env python3
"""Validate, preflight, train/resume, evaluate, analyze, and summarize U-HERA."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import transformers
from torch.utils.data import Subset

from u_hera import ARCHITECTURE
from u_hera.data import load_datasets, make_loaders, move_batch, unpack
from u_hera.metrics import metrics_from_rows
from u_hera.model import ModelSettings, MODALITIES, STAGES, UHERAModel
from u_hera.trainer import UHERATrainer
from u_hera.utils import object_hash, seed_everything, sha256_file, write_json

LOGGER = logging.getLogger("u_hera")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config["architecture"] != ARCHITECTURE or config["dataset"] != "mosi":
        raise ValueError("this implementation validates u_hera_v1 / MOSI")
    ModelSettings(**config["model"])
    training = config["training"]
    if training["augmentation"] != "none":
        raise ValueError("offline utility requires the configured clean input protocol")
    if config["runtime"] != dict(dtype="float16", gradient_checkpointing=False, use_flash_attn=False):
        raise ValueError("unvalidated LLM runtime settings")
    if min(training["microbatch"], training["accumulation_steps"], training["learning_rate"],
           training["utility_temperature_floor"], training["gradient_clip"], training["progress_interval"]) <= 0:
        raise ValueError("invalid training settings")
    if not 0 < config["analysis"]["deletion_fraction"] <= 1:
        raise ValueError("invalid deletion fraction")
    for stage in STAGES:
        if min(training["stages"][stage].values()) < 1:
            raise ValueError("invalid stage schedule")
    return config


def source_manifest():
    paths = [ROOT / "run.py", *sorted((ROOT / "u_hera").glob("*.py"))]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def identity_manifest(config):
    data_hash = sha256_file(config["dataset_path"])
    if data_hash != config["dataset_sha256"]:
        raise ValueError("MOSI dataset SHA-256 differs from the approved source")
    model_path = Path(config["model_path"])
    model_files = [p for p in model_path.iterdir() if p.suffix in (".py", ".json", ".tiktoken", ".bin", ".safetensors")]
    if not any(p.suffix in (".bin", ".safetensors") for p in model_files):
        raise FileNotFoundError("no local Qwen checkpoint weights")
    manifest = dict(architecture=ARCHITECTURE, config=config, source=source_manifest(), dataset_sha256=data_hash,
                    backbone_files={p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p)) for p in sorted(model_files)},
                    environment=dict(python=platform.python_version(), torch=torch.__version__,
                                     transformers=transformers.__version__, cuda_build=torch.version.cuda))
    return manifest, object_hash(manifest)


def load_tokenizer(config):
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model_path"], padding_side="left",
                                                           trust_remote_code=True, local_files_only=True)
    tokenizer.eos_token_id = int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.bos_token_id = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    encoded = tokenizer([f"{x/10:+.1f}" for x in range(-30, 31)], add_special_tokens=False)["input_ids"]
    if max(map(len, encoded)) > config["model"]["max_new_tokens"]:
        raise ValueError("decoding budget cannot emit every one-decimal MOSI target")
    return tokenizer


def load_model(config, tokenizer):
    llm = transformers.AutoModelForCausalLM.from_pretrained(
        config["model_path"], local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16, fp16=True, bf16=False, fp32=False, use_flash_attn=False,
    )
    llm.gradient_checkpointing_disable()
    model = UHERAModel(llm, tokenizer, ModelSettings(**config["model"]), config["task_prompt"])
    return model.to(torch.device("cuda:0"))


def require_gpu_allocation():
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("GPU modes require a live Slurm allocation")
    result = subprocess.run(["scontrol", "show", "job", job, "-o"], text=True, capture_output=True, check=True)
    fields = dict(part.split("=", 1) for part in result.stdout.split() if "=" in part)
    allocation = fields.get("AllocTRES", "")
    allocated = re.search(r"(?:^|,)gres/gpu=(\d+)", allocation)
    if fields.get("JobState") != "RUNNING" or not allocated or int(allocated[1]) < 1:
        raise RuntimeError("allocation is not RUNNING with assigned GPUs")
    if "login" in socket.gethostname().lower() or fields.get("NodeList") != socket.gethostname().split('.')[0]:
        raise RuntimeError("shell is not on the allocated single compute node")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("U-HERA v1 expects exactly one assigned, visible CUDA device")
    return dict(job_id=job, node=socket.gethostname(), allocated_gpus=int(allocated[1]), visible_gpus=torch.cuda.device_count(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), end_time=fields.get("EndTime"),
                gpu=torch.cuda.get_device_name(0), memory_bytes=torch.cuda.get_device_properties(0).total_memory)


def prepare(config):
    LOGGER.info("checking immutable source, dataset and local model identities")
    manifest, identity = identity_manifest(config)
    tokenizer = load_tokenizer(config)
    datasets = load_datasets(config, tokenizer)
    return manifest, identity, tokenizer, datasets


def preflight(config, output):
    allocation = require_gpu_allocation()
    manifest, base_identity, tokenizer, datasets = prepare(config)
    smoke = copy.deepcopy(config)
    smoke["training"].update(microbatch=2, accumulation_steps=2, progress_interval=1)
    for stage in STAGES:
        smoke["training"]["stages"][stage] = dict(max_epochs=1, patience=1)
    subsets = dict(train=Subset(datasets["train"], list(range(8))), valid=Subset(datasets["valid"], list(range(4))))
    # The official test split is never used in preflight.
    loaders = make_loaders(subsets, smoke["training"])
    smoke_identity = object_hash(dict(base=base_identity, protocol="8_train_4_validation_two_updates_per_stage", config=smoke))
    seed_everything(config["seed"])
    model = load_model(config, tokenizer)
    trainer = UHERATrainer(model, smoke, output, smoke_identity)
    summaries = {}
    gradients = {}
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("llm.")}
    for stage in STAGES:
        summaries[stage] = trainer.fit_stage(loaders, stage)
        history = json.loads((output / f"{stage}_history.json").read_text())
        if history[0]["train"]["updates"] != 2:
            raise RuntimeError(f"preflight {stage} did not complete two optimizer updates")
        gradients[stage] = {n: float(p.grad.float().norm()) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        if not gradients[stage] or not any(v > 0 for v in gradients[stage].values()):
            raise RuntimeError(f"no trainable gradients in {stage}")
        if any(p.requires_grad or p.grad is not None for p in model.llm.parameters()):
            raise RuntimeError("LLM is not frozen")
        if stage == "evidence_warmup":
            reference = output / "checkpoints" / "reference.pt"
            trainer.checkpoint(reference, stage, summaries[stage]["best_epoch"], summaries[stage]["valid_metrics"])
            trainer.cache_reference(loaders["train"], reference)
    model.set_stage("eval")
    batch = move_batch(next(iter(loaders["valid"])), trainer.device)
    _, text, audio, vision = unpack(batch)
    with torch.no_grad(), trainer.amp():
        evidence = model.encode_evidence(text, audio, vision)
        for i, modality in enumerate(MODALITIES):
            if not torch.allclose(evidence["J"][modality].sum(-1), evidence["G"][:, i], atol=1e-5):
                raise AssertionError("hierarchical importance conservation failed")
            deleted = model.evidence.delete_modality(evidence, modality)
            all_local = model.evidence.delete_positions(evidence, modality, evidence["masks"][modality])
            if not torch.allclose(deleted, all_local, atol=2e-4, rtol=2e-3):
                raise AssertionError("local/modal deletion mismatch on GPU")
        empty = model.encode_evidence(text, audio, vision, presence=torch.zeros(len(batch["id"]), 3, device=trainer.device))
        if not torch.isfinite(empty["tokens"]).all() or not torch.equal(empty["tokens"], model.evidence.composer.base_prefix[None].expand_as(empty["tokens"])):
            raise AssertionError("all-null prefix invariant failed")
        before = model.generate(text, audio, vision)
    checkpoint = output / "checkpoints" / "roundtrip.pt"
    trainer.checkpoint(checkpoint, "joint", 1, {})
    trainer.load(checkpoint)
    with torch.no_grad(), trainer.amp():
        after = model.generate(text, audio, vision)
    if before != after:
        raise AssertionError("checkpoint roundtrip generation mismatch")
    changed = [n for n, p in model.named_parameters() if n in initial and not torch.equal(initial[n], p.detach())]
    if not any(n.startswith("evidence.slot_router.") for n in changed) or not any(n.startswith("audio_encoder.") for n in changed):
        raise AssertionError("router or evidence parameters did not update")
    if source_manifest() != manifest["source"]:
        raise RuntimeError("source changed during preflight")
    report = dict(passed=True, base_identity=base_identity, identity=smoke_identity, allocation=allocation,
                  manifest=manifest, train_samples=8, validation_samples=4, official_test_used=False,
                  stages=summaries, gradient_norms=gradients, changed_parameters=changed,
                  trainable_external_parameters=sum(p.numel() for n, p in model.named_parameters() if not n.startswith("llm.")),
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), report_path=str(output / "preflight.json"))
    write_json(output / "preflight.json", report)
    write_json(ROOT / "outputs" / "preflight_latest.json", report)
    LOGGER.info("PREFLIGHT PASSED: %s", output / "preflight.json")
    return report


def write_report(output):
    result = json.loads((output / "result.json").read_text())
    rows = [json.loads(line) for line in (output / "predictions" / "test.jsonl").read_text().splitlines()]
    recomputed = metrics_from_rows(rows)
    for key, value in recomputed.items():
        if isinstance(value, float) and abs(value-result["test"][key]) > 1e-10:
            raise AssertionError(f"exported test metric mismatch: {key}")
    write_json(output / "metrics_recomputed.json", recomputed)
    metrics = result["test"]
    lines = ["# U-HERA / Qwen-1.8B / MOSI / seed 1111", "",
             f"Selected stage: `{result['selection']['stage']}`, epoch {result['selection']['best_epoch']}; validation MAE {result['valid']['MAE']:.6f}.", "",
             "| Test metric | Value |", "| --- | ---: |"]
    for key in ("MAE", "Corr", "Has0_acc_2", "Has0_F1_score", "Non0_acc_2", "Non0_F1_score", "Mult_acc_7", "invalid_generation_rate"):
        value = metrics[key]
        lines.append(f"| {key} | {value:.6f} |" if value is not None else f"| {key} | undefined |")
    lines += ["", "Mean auxiliary budget [T, A, V, Null]: " + str(metrics["mean_G_T_A_V_Null"]), "",
              "All metrics were recomputed from exported predictions. Model selection used validation MAE; the test set was evaluated after selection.", "",
              "G/J describe auxiliary evidence allocation. Reference and local deletion fix the remaining evidence path; they do not estimate complete modality or raw-frame causal effects.", "",
              "| Validation deletion | Δ MAE | Mean Δ target-token CE |", "| --- | ---: | ---: |"]
    if result.get("analysis"):
        for name, metric in result["analysis"]["conditions"].items():
            lines.append(f"| {name} | {metric['delta_MAE']:.6f} | {metric['mean_loss_delta']:.6f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return recomputed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "preflight", "train", "evaluate", "analyze", "report"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "qwen_mosi.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-report", type=Path, default=ROOT / "outputs" / "preflight_latest.json")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or Path(config["output_root"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.mode == "preflight":
        output = args.output or ROOT / "outputs" / "preflight" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output.mkdir(parents=True, exist_ok=False)
    if args.mode == "report":
        print(json.dumps(write_report(output)))
        return
    if args.mode == "validate":
        manifest, identity, _, datasets = prepare(config)
        report = dict(passed=True, identity=identity, manifest=manifest,
                      splits={k: dict(samples=len(d), audit=d.audit) for k, d in datasets.items()})
        write_json(ROOT / "outputs" / "validation.json", report)
        print(json.dumps(dict(passed=True, identity=identity, splits=config["splits"])))
        return
    if args.mode == "preflight":
        handler = logging.FileHandler(output / "preflight.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
        preflight(config, output)
        return
    allocation = require_gpu_allocation()
    manifest, identity, tokenizer, datasets = prepare(config)
    if args.mode == "train":
        report = json.loads(args.preflight_report.read_text())
        if not report.get("passed") or report.get("base_identity") != identity:
            raise RuntimeError("matching completed GPU preflight is required")
        # Do not wait until the first checkpoint read to discover configuration drift.
        previous_manifest = output / "manifest.json"
        if previous_manifest.exists() and object_hash(json.loads(previous_manifest.read_text())) != identity:
            raise ValueError("existing run manifest differs; refusing to overwrite provenance")
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "run.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "train" and (output / "resolved_config.json").exists() and not args.resume:
            raise FileExistsError("run exists; use --resume to continue the same immutable experiment")
        if args.mode == "train" and (output / "result.json").exists():
            existing = json.loads((output / "result.json").read_text())
            if existing["identity"] != identity:
                raise ValueError("completed run identity mismatch")
            write_report(output)
            LOGGER.info("run already complete: %s", output / "report.md")
            return
        handler = logging.FileHandler(output / "train.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
        write_json(output / "resolved_config.json", config)
        write_json(output / "manifest.json", manifest)
        write_json(output / "allocation.json", allocation)
        seed_everything(config["seed"])
        model = load_model(config, tokenizer)
        trainer = UHERATrainer(model, config, output, identity)
        loaders = make_loaders(datasets, config["training"])
        if args.mode == "train":
            trainer.fit(loaders, resume=args.resume)
            if source_manifest() != manifest["source"]:
                raise RuntimeError("source drift occurred during training")
            write_report(output)
        else:
            trainer.load(output / "checkpoints" / "final.pt")
            model.set_stage("eval")
            if args.mode == "evaluate":
                trainer.evaluate(loaders["valid"], "final_valid", detailed=True, reuse=True)
                trainer.evaluate(loaders["test"], "test", detailed=True, reuse=True)
            else:
                from u_hera.analysis import analyze
                analyze(trainer, loaders["valid"])


if __name__ == "__main__":
    main()
