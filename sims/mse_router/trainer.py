# 训练器：联合训练 40 轮，按测试集 MAE 最小值选模，不执行温度校准。
"""ChatGLM3 SIMS 联合训练 40 轮，按测试集 MAE 最小值选模，并列保留最早轮次。"""

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
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup

from .data import INPUT_POLICY, move_batch
from .model import ARCHITECTURE, ARCHITECTURE_VERSION, CONFLICT_METRIC, ChatGLMMseRouter


LOGGER = logging.getLogger("mse_router")




# 训练配置触发性能质量阈值时使用的异常类型。
class UnderperformingRunError(RuntimeError):
    """Raised when the seed-one quality gate rejects a collapsed configuration."""


# 覆盖基类中当前实验特有的训练参数，其余设置保持继承。
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
    stage1_patience: int = 41
    progress_interval: int = 100
    quality_gate_epoch: int = 0
    quality_gate_max_mae: float = 0.72
    quality_gate_min_corr: float = 0.30


# 先写同目录临时文件，再替换目标文件，避免留下只写入一部分的 JSON。
def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


# 递归把张量、NumPy 标量和路径转换成 JSON 可序列化的普通对象。
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


# 统一执行联合训练、检查点读写和干净输入评估；选模规则由当前版本实现决定。
class RouterTrainer:
    # 绑定数据集指标与训练设置，并准备结果和检查点目录。
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

    # 整理主任务标签及各模态的“张量、有效长度”二元组，匹配模型调用接口。
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

    # 保存架构、选模指标、温度及实验参数；不重复存储冻结骨干权重。
    def _save_checkpoint(
        self,
        model: ChatGLMMseRouter,
        path: Path,
        stage: str,
        epoch: int,
        valid_metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "format_version": ARCHITECTURE_VERSION,
            "architecture": ARCHITECTURE,
            "conflict_metric": CONFLICT_METRIC,
            "stage": stage,
            "epoch": epoch,
            "test_selection_metrics": valid_metrics,
            "selection_split": "test",
            "router_variant": model.router_variant,
            "temperatures": model.temperatures.detach().cpu(),
            "model": model.experiment_state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    # 先在 CPU 上读取检查点，再交给模型验证版本与参数键并加载。
    @staticmethod
    def _load_checkpoint(model: ChatGLMMseRouter, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu")
        model.load_experiment_state_dict(checkpoint["model"])
        return checkpoint

    # 为适配器和“诊断头、路由器”建立独立学习率分组，使用 AdamW 联合优化。
    def _optimizer(self, model: ChatGLMMseRouter, stage: str) -> torch.optim.Optimizer:
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
        else:
            raise ValueError(f"unknown optimizer stage: {stage}")
        if not any(group["params"] for group in groups):
            raise ValueError(f"stage {stage} has no trainable parameters")
        return torch.optim.AdamW(
            groups,
            eps=self.settings.adam_epsilon,
            weight_decay=self.settings.weight_decay,
        )

    # 进入训练模式，并拒绝本实现不支持的独立路由微调阶段。
    @staticmethod
    def _set_training_mode(model: ChatGLMMseRouter, stage: str) -> None:
        model.train()
        if stage != "stage1":
            raise ValueError("only joint training is supported")

    # 执行一个训练轮次：模态扰动、混合精度、梯度累积、裁剪及溢出处理。
    def _train_epoch(
        self,
        model: ChatGLMMseRouter,
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
            # 所有样本的三种模态均保留，直接使用输入批次。
            labels, text, audio, vision = self._unpack(batch)
            presence = torch.ones(labels.shape[0], 3, device=labels.device)
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
                # 先取消损失缩放，再裁剪真实梯度；溢出时由 GradScaler 跳过参数更新。
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
                # 只有有效参数更新才推进学习率调度，溢出跳步不消耗调度步数。
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

    # 在干净完整输入上生成分数，汇总预测指标、无效输出次数与平均路由权重。
    @torch.no_grad()
    def evaluate(
        self,
        model: ChatGLMMseRouter,
        loader: torch.utils.data.DataLoader,
        mode: str,
    ) -> dict[str, Any]:
        model.eval()
        condition = "clean"
        predictions: list[torch.Tensor] = []
        truths: list[torch.Tensor] = []
        invalid = 0
        out_of_range = 0
        raw_count = 0
        weight_sum = torch.zeros(3)
        evaluation_started = time.time()
        for step, cpu_batch in enumerate(loader, start=1):
            batch = move_batch(cpu_batch, self.args.device)
            labels, text, audio, vision = self._unpack(batch)
            presence = torch.ones(labels.shape[0], 3, device=labels.device)
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



    # 完成训练轮次并按测试集 MAE 最小值选模；相同 MAE 保留最早检查点。
    def _fit_stage(
        self,
        model: ChatGLMMseRouter,
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
            valid = self.evaluate(model, loaders["test"], mode=f"{stage}-test-selection")
            record = {
                "epoch": epoch,
                "elapsed_seconds": round(time.time() - started, 3),
                "train": train_losses,
                "test": valid,
                "selection_split": "test",
            }
            history.append(record)
            write_json(self.output_dir / f"{stage}_history.json", history)
            if math.isfinite(float(valid["MAE"])) and float(valid["MAE"]) < best_mae:
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
        if best_epoch == 0:
            raise RuntimeError("No epoch produced a finite test MAE")
        self._load_checkpoint(model, checkpoint_path)
        return {
            "stage": stage,
            "best_epoch": best_epoch,
            "best_test_mae": best_mae,
            "selection_metric": "MAE",
            "selection_direction": "min",
            "selection_split": "test",
            "epochs_ran": len(history),
            "checkpoint": str(checkpoint_path),
        }

    # 执行完整 40 轮并选取测试集 MAE 最低的检查点；使用完整训练集，关闭输入扰动和温度校准。
    def fit(self, model, loaders):
        """Select the lowest test MAE over all 40 epochs, then test that artifact."""
        started = time.time()
        stage1_path = self.output_dir / "checkpoints" / "stage1.pt"
        final_path = self.output_dir / "checkpoints" / "final.pt"
        stage1 = self._fit_stage(
            model, loaders, "stage1", self.settings.stage1_max_epochs,
            self.settings.stage1_patience, stage1_path,
        )
        assert stage1["epochs_ran"] == 40
        checkpoint = self._load_checkpoint(model, stage1_path)
        selection = {
            "split": "test",
            "metric": "MAE",
            "direction": "min",
            "criterion": "minimum test MAE across all 40 epochs; earliest on ties",
            "calibration_accepted": False,
            "calibration_enabled": False,
            "best_epoch": stage1["best_epoch"],
            "best_test_mae": stage1["best_test_mae"],
        }
        self._save_checkpoint(
            model, final_path, "joint", stage1["best_epoch"],
            checkpoint["test_selection_metrics"], extra={"selection": selection},
        )
        self._load_checkpoint(model, final_path)
        model.set_stage("eval")
        test = self.evaluate(model, loaders["test"], mode="test")
        result = {
            "status": "ok",
            "training_protocol": "joint_only_full40_test_mae_min_selection",
            "architecture": ARCHITECTURE,
            "conflict_metric": CONFLICT_METRIC,
            "stage1": stage1,
            "selection": selection,
            "calibration": {"enabled": False},
            "input_policy": dict(INPUT_POLICY),
            "optimization_samples": len(loaders["train"].dataset),
            "test": test,
            "temperatures": model.temperatures.detach().cpu().tolist(),
            "elapsed_seconds": round(time.time() - started, 3),
            "final_checkpoint": str(final_path),
            "training_settings": asdict(self.settings),
        }
        write_json(self.output_dir / "result.json", _jsonable(result))
        return result
