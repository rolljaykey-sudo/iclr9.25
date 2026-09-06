"""Three-stage optimization and evaluation for the Qwen/MOSEI router."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup

from .data import (
    ROBUSTNESS_CONDITIONS,
    augment_modalities,
    move_batch,
    robustness_condition,
)
from .math_utils import soft_ordinal_targets
from .model import QwenMseRouter


LOGGER = logging.getLogger("mse_router")


class CalibrationError(RuntimeError):
    """Raised when temperature fitting fails the predeclared validity checks."""


class UnderperformingRunError(RuntimeError):
    """Raised when the seed-one quality gate rejects a collapsed configuration."""


@dataclass(frozen=True)
class TrainingSettings:
    adapter_lr: float = 5e-3
    head_router_lr: float = 1e-3
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-4
    warmup_fraction: float = 0.10
    gradient_clip: float = 1.0
    accumulation_steps: int = 4
    stage1_max_epochs: int = 40
    stage1_patience: int = 10
    router_max_epochs: int = 10
    router_patience: int = 3
    progress_interval: int = 100
    quality_gate_epoch: int = 4
    quality_gate_max_mae: float = 0.72
    quality_gate_min_corr: float = 0.30


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class RouterTrainer:
    def __init__(
        self,
        args: Any,
        output_dir: Path,
        settings: TrainingSettings,
    ) -> None:
        from utils.metricsTop import MetricsTop

        self.args = args
        self.output_dir = output_dir
        self.settings = settings
        self.metrics = MetricsTop(args).getMetics(args.datasetName)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _unpack(
        self, batch: dict[str, Any]
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        labels = batch["labels"]["M"].view(-1)
        text = (batch["text"], batch["text_lengths"])
        audio = (batch["audio"], batch["audio_lengths"])
        vision = (batch["vision"], batch["vision_lengths"])
        return labels, text, audio, vision

    def _save_checkpoint(
        self,
        model: QwenMseRouter,
        path: Path,
        stage: str,
        epoch: int,
        valid_metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "format_version": 1,
            "stage": stage,
            "epoch": epoch,
            "valid_metrics": valid_metrics,
            "router_variant": model.router_variant,
            "temperatures": model.temperatures.detach().cpu(),
            "model": model.experiment_state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @staticmethod
    def _load_checkpoint(model: QwenMseRouter, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu")
        model.load_experiment_state_dict(checkpoint["model"])
        return checkpoint

    def _optimizer(self, model: QwenMseRouter, stage: str) -> torch.optim.Optimizer:
        if stage == "stage1":
            adapter_parameters = [
                parameter
                for parameter in model.adapter_parameters()
                if parameter.requires_grad
            ]
            head_router_parameters = [
                parameter
                for module in (model.ordinal_head, model.router)
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            groups = [
                {"params": adapter_parameters, "lr": self.settings.adapter_lr},
                {
                    "params": head_router_parameters,
                    "lr": self.settings.head_router_lr,
                },
            ]
        elif stage == "router":
            groups = [
                {
                    "params": [
                        parameter
                        for parameter in model.router.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": self.settings.head_router_lr,
                }
            ]
        else:
            raise ValueError(f"unknown optimizer stage: {stage}")
        if not any(group["params"] for group in groups):
            raise ValueError(f"stage {stage} has no trainable parameters")
        return torch.optim.AdamW(
            groups,
            eps=self.settings.adam_epsilon,
            weight_decay=self.settings.weight_decay,
        )

    @staticmethod
    def _set_training_mode(model: QwenMseRouter, stage: str) -> None:
        model.train()
        if stage == "router":
            # Only the Router's dropout is active after stage 1. Frozen feature
            # extractors and the ordinal head remain deterministic.
            for module in (
                model.llm,
                model.text_pool,
                model.text_projection,
                model.text_adapter,
                model.audio_encoder,
                model.audio_adapter,
                model.vision_encoder,
                model.vision_adapter,
                model.ordinal_head,
            ):
                module.eval()

    def _train_epoch(
        self,
        model: QwenMseRouter,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.cuda.amp.GradScaler,
        stage: str,
        epoch: int,
        max_epochs: int,
    ) -> dict[str, float]:
        self._set_training_mode(model, stage)
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "generation": 0.0, "auxiliary": 0.0}
        batches = len(loader)
        epoch_started = time.time()
        for step, cpu_batch in enumerate(loader, start=1):
            batch = move_batch(cpu_batch, self.args.device)
            augmented, presence = augment_modalities(batch)
            labels, text, audio, vision = self._unpack(augmented)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(
                    labels,
                    text,
                    audio,
                    vision,
                    presence=presence,
                    stage=stage,
                )
                scaled_loss = output["Loss"] / self.settings.accumulation_steps
            if not bool(torch.isfinite(output["Loss"]).item()):
                raise RuntimeError(
                    f"non-finite {stage} loss at batch {step}: {output['Loss'].item()}"
                )
            scaler.scale(scaled_loss).backward()
            should_step = (
                step % self.settings.accumulation_steps == 0 or step == batches
            )
            if should_step:
                scale_before = float(scaler.get_scale())
                scaler.unscale_(optimizer)
                trainable = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                gradient_norm = clip_grad_norm_(trainable, self.settings.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                optimizer.zero_grad(set_to_none=True)
                if bool(torch.isfinite(gradient_norm).item()):
                    scheduler.step()
                elif scale_after < scale_before:
                    LOGGER.warning(
                        "%s gradient overflow at batch %d; GradScaler reduced "
                        "scale from %.0f to %.0f and skipped the update",
                        stage,
                        step,
                        scale_before,
                        scale_after,
                    )
                else:
                    raise RuntimeError(
                        f"non-finite {stage} gradient norm at batch {step} "
                        "without a GradScaler reduction"
                    )
            totals["loss"] += float(output["Loss"].detach())
            totals["generation"] += float(output["GenerationLoss"].detach())
            totals["auxiliary"] += float(output["AuxiliaryLoss"].detach())
            if step % self.settings.progress_interval == 0 or step == batches:
                elapsed = time.time() - epoch_started
                eta = elapsed / step * (batches - step)
                learning_rates = [group["lr"] for group in optimizer.param_groups]
                LOGGER.info(
                    "%s epoch %d/%d batch %d/%d (%.1f%%) loss=%.4f "
                    "gen=%.4f aux=%.4f lr=%s scale=%.0f eta=%.0fs",
                    stage,
                    epoch,
                    max_epochs,
                    step,
                    batches,
                    100.0 * step / batches,
                    totals["loss"] / step,
                    totals["generation"] / step,
                    totals["auxiliary"] / step,
                    [f"{value:.3e}" for value in learning_rates],
                    scaler.get_scale(),
                    eta,
                )
        return {key: value / max(1, batches) for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(
        self,
        model: QwenMseRouter,
        loader: torch.utils.data.DataLoader,
        mode: str,
        condition: str = "clean",
    ) -> dict[str, Any]:
        model.eval()
        predictions: list[torch.Tensor] = []
        truths: list[torch.Tensor] = []
        invalid = 0
        out_of_range = 0
        raw_count = 0
        weight_sum = torch.zeros(3)
        evaluation_started = time.time()
        for step, cpu_batch in enumerate(loader, start=1):
            batch = move_batch(cpu_batch, self.args.device)
            changed, presence = robustness_condition(batch, condition)
            labels, text, audio, vision = self._unpack(changed)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                values, diagnostics = model.generate(
                    text,
                    audio,
                    vision,
                    presence=presence,
                    return_diagnostics=True,
                )
            predictions.append(torch.tensor(values, dtype=torch.float32))
            truths.append(labels.detach().float().cpu())
            invalid += int(diagnostics["invalid_count"])
            out_of_range += int(diagnostics["out_of_range_count"])
            raw_count += len(values)
            weight_sum += diagnostics["weights"].sum(dim=0)
            if step % self.settings.progress_interval == 0 or step == len(loader):
                elapsed = time.time() - evaluation_started
                eta = elapsed / step * (len(loader) - step)
                LOGGER.info(
                    "%s %s batch %d/%d (%.1f%%) invalid=%d "
                    "out_of_range=%d eta=%.0fs",
                    mode,
                    condition,
                    step,
                    len(loader),
                    100.0 * step / len(loader),
                    invalid,
                    out_of_range,
                    eta,
                )
        prediction = torch.cat(predictions)
        truth = torch.cat(truths)
        metrics = {
            key: float(value) for key, value in self.metrics(prediction, truth).items()
        }
        result = {
            **metrics,
            "condition": condition,
            "samples": raw_count,
            "invalid_generations": invalid,
            "out_of_range_generations": out_of_range,
            "mean_router_weights": (weight_sum / max(1, raw_count)).tolist(),
        }
        LOGGER.info(
            "%s %s MAE=%.4f Corr=%.4f invalid=%d out_of_range=%d weights=%s",
            mode,
            condition,
            result["MAE"],
            result["Corr"],
            invalid,
            out_of_range,
            [round(item, 4) for item in result["mean_router_weights"]],
        )
        return result

    @torch.no_grad()
    def _collect_calibration_logits(
        self,
        model: QwenMseRouter,
        loader: torch.utils.data.DataLoader,
    ) -> tuple[np.ndarray, np.ndarray]:
        model.set_stage("calibration")
        model.eval()
        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, self.args.device)
            labels, text, audio, vision = self._unpack(batch)
            presence = torch.ones(labels.shape[0], 3, device=labels.device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pseudo = model.encode_modalities(text, audio, vision, presence)
                logits = model.diagnostic_logits(pseudo)
            targets = soft_ordinal_targets(labels, model.anchors)
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(targets.detach().float().cpu())
        return torch.cat(all_logits).numpy(), torch.cat(all_targets).numpy()

    def calibrate_temperatures(
        self,
        model: QwenMseRouter,
        loader: torch.utils.data.DataLoader,
    ) -> dict[str, Any]:
        LOGGER.info("calibration: collecting clean modality logits")
        logits, targets = self._collect_calibration_logits(model, loader)
        LOGGER.info("calibration: fitting three bounded scalar temperatures")

        def nll(modality: int, temperature: float) -> float:
            scaled = logits[:, modality, :] / temperature
            log_probabilities = scaled - logsumexp(scaled, axis=-1, keepdims=True)
            return float(-(targets * log_probabilities).sum(axis=-1).mean())

        temperatures: list[float] = []
        records: list[dict[str, Any]] = []
        valid = True
        for modality in range(3):
            baseline = nll(modality, 1.0)
            result = minimize_scalar(
                lambda temperature: nll(modality, float(temperature)),
                bounds=(0.05, 10.0),
                method="bounded",
                options={"xatol": 1e-5, "maxiter": 200},
            )
            temperature = float(result.x)
            calibrated = nll(modality, temperature)
            at_bound = temperature <= 0.0501 or temperature >= 9.999
            modality_valid = bool(
                result.success
                and math.isfinite(temperature)
                and math.isfinite(calibrated)
                and not at_bound
                and calibrated <= baseline + 1e-7
            )
            valid = valid and modality_valid
            temperatures.append(temperature)
            records.append(
                {
                    "modality": ("text", "audio", "vision")[modality],
                    "temperature": temperature,
                    "tau1_nll": baseline,
                    "calibrated_nll": calibrated,
                    "optimizer_success": bool(result.success),
                    "at_bound": at_bound,
                    "valid": modality_valid,
                    "message": str(result.message),
                }
            )
        report = {
            "valid": valid,
            "samples": int(logits.shape[0]),
            "temperatures": temperatures,
            "modalities": records,
        }
        write_json(self.output_dir / "calibration.json", report)
        if valid:
            model.set_temperatures(temperatures)
        return report

    def _fit_stage(
        self,
        model: QwenMseRouter,
        loaders: dict[str, torch.utils.data.DataLoader],
        stage: str,
        max_epochs: int,
        patience: int,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        model.set_stage(stage)
        optimizer = self._optimizer(model, stage)
        updates_per_epoch = math.ceil(
            len(loaders["train"]) / self.settings.accumulation_steps
        )
        total_updates = max_epochs * updates_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(round(total_updates * self.settings.warmup_fraction)),
            num_training_steps=total_updates,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        best_mae = float("inf")
        best_epoch = 0
        history: list[dict[str, Any]] = []
        for epoch in range(1, max_epochs + 1):
            started = time.time()
            train_losses = self._train_epoch(
                model,
                loaders["train"],
                optimizer,
                scheduler,
                scaler,
                stage,
                epoch,
                max_epochs,
            )
            valid = self.evaluate(model, loaders["valid"], mode=f"{stage}-valid")
            record = {
                "epoch": epoch,
                "elapsed_seconds": round(time.time() - started, 3),
                "train": train_losses,
                "valid": valid,
            }
            history.append(record)
            write_json(self.output_dir / f"{stage}_history.json", history)
            if valid["MAE"] <= best_mae - 1e-6:
                best_mae = float(valid["MAE"])
                best_epoch = epoch
                self._save_checkpoint(
                    model,
                    checkpoint_path,
                    stage,
                    epoch,
                    valid,
                    extra={"training_settings": asdict(self.settings)},
                )
            if (
                stage == "stage1"
                and epoch == self.settings.quality_gate_epoch
                and valid["MAE"] > self.settings.quality_gate_max_mae
                and valid["Corr"] < self.settings.quality_gate_min_corr
            ):
                quality_gate = {
                    "passed": False,
                    "epoch": epoch,
                    "observed_mae": valid["MAE"],
                    "observed_corr": valid["Corr"],
                    "required_mae_at_most": self.settings.quality_gate_max_mae,
                    "required_corr_at_least": self.settings.quality_gate_min_corr,
                }
                write_json(self.output_dir / "quality_gate.json", quality_gate)
                raise UnderperformingRunError(
                    "stage-1 quality gate failed after warmup: "
                    f"MAE={valid['MAE']:.4f}, Corr={valid['Corr']:.4f}"
                )
            if epoch - best_epoch >= patience:
                break
        self._load_checkpoint(model, checkpoint_path)
        return {
            "stage": stage,
            "best_epoch": best_epoch,
            "best_valid_mae": best_mae,
            "epochs_ran": len(history),
            "checkpoint": str(checkpoint_path),
        }

    def fit(
        self,
        model: QwenMseRouter,
        loaders: dict[str, torch.utils.data.DataLoader],
        run_robustness: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        stage1_path = self.output_dir / "checkpoints" / "stage1.pt"
        final_path = self.output_dir / "checkpoints" / "final.pt"
        stage1 = self._fit_stage(
            model,
            loaders,
            "stage1",
            self.settings.stage1_max_epochs,
            self.settings.stage1_patience,
            stage1_path,
        )
        calibration = self.calibrate_temperatures(model, loaders["calibration"])
        if not calibration["valid"]:
            failure = {
                "status": "calibration_invalid",
                "stage1": stage1,
                "calibration": calibration,
            }
            write_json(self.output_dir / "result.json", failure)
            raise CalibrationError(
                "temperature calibration was non-finite, reached a bound, or failed "
                "to improve soft NLL; refusing the remaining multi-seed run"
            )
        self._save_checkpoint(
            model,
            self.output_dir / "checkpoints" / "calibrated.pt",
            "calibration",
            0,
            {},
            extra=calibration,
        )

        if model.router_variant == "uniform":
            initial_valid = self.evaluate(
                model, loaders["valid"], mode="uniform-valid"
            )
            self._save_checkpoint(
                model, final_path, "uniform", 0, initial_valid, extra=calibration
            )
            router_stage = {
                "stage": "uniform",
                "best_epoch": 0,
                "best_valid_mae": initial_valid["MAE"],
                "epochs_ran": 0,
                "checkpoint": str(final_path),
            }
        else:
            router_stage = self._fit_stage(
                model,
                loaders,
                "router",
                self.settings.router_max_epochs,
                self.settings.router_patience,
                final_path,
            )

        test = self.evaluate(model, loaders["test"], mode="test")
        robustness: dict[str, Any] = {}
        if run_robustness:
            for condition in ROBUSTNESS_CONDITIONS:
                if condition == "clean":
                    robustness[condition] = test
                else:
                    robustness[condition] = self.evaluate(
                        model,
                        loaders["test"],
                        mode="robustness",
                        condition=condition,
                    )
            write_json(self.output_dir / "robustness.json", robustness)
        result = {
            "status": "ok",
            "stage1": stage1,
            "calibration": calibration,
            "router_stage": router_stage,
            "test": test,
            "robustness": robustness,
            "temperatures": model.temperatures.detach().cpu().tolist(),
            "elapsed_seconds": round(time.time() - started, 3),
            "final_checkpoint": str(final_path),
            "training_settings": asdict(self.settings),
        }
        write_json(self.output_dir / "result.json", _jsonable(result))
        return result
