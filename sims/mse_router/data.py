# 数据处理：按视频分组划分校准集、构造批次，并实现训练期的模态扰动。
"""Group-safe calibration split, batching, and modality corruptions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset


MODALITY_INDEX = {"text": 0, "audio": 1, "vision": 2}


# 从样本标识中提取视频 ID，使同一视频的片段归入同一数据组。
def video_group(sample_id: Any) -> str:
    if isinstance(sample_id, bytes):
        sample_id = sample_id.decode("utf-8")
    return str(sample_id).split("$_$", maxsplit=1)[0]


# 保存优化集与校准集的索引、视频组数及划分指纹，便于检查数据隔离。
@dataclass(frozen=True)
class CalibrationSplit:
    train_indices: list[int]
    calibration_indices: list[int]
    train_groups: int
    calibration_groups: int
    fingerprint: str


# 按视频组隔离、按最近情感锚点分层，从训练集保留独立校准子集。
def make_calibration_split(
    dataset: Dataset, fraction: float = 0.10, seed: int = 20260903
) -> CalibrationSplit:
    """Reserve a video-group-disjoint, nearest-anchor-stratified train subset."""
    if not 0.0 < fraction < 0.5:
        raise ValueError("calibration fraction must be between zero and 0.5")
    ids = np.asarray(dataset.ids)
    labels = np.asarray(dataset.labels["M"]).reshape(-1)
    groups = np.asarray([video_group(item) for item in ids])
    strata = np.clip(np.rint(labels * 3.0), -3, 3).astype(np.int64)
    n_splits = int(round(1.0 / fraction))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    target_count = len(dataset) * fraction
    candidates = list(splitter.split(np.zeros(len(dataset)), strata, groups))
    train_indices, calibration_indices = min(
        candidates, key=lambda pair: abs(len(pair[1]) - target_count)
    )
    # 按视频级别复核隔离，防止同一视频片段同时进入优化集与校准集。
    train_groups = set(groups[train_indices])
    calibration_groups = set(groups[calibration_indices])
    overlap = train_groups & calibration_groups
    if overlap:
        raise RuntimeError(f"video group leakage in calibration split: {sorted(overlap)[:3]}")
    digest = hashlib.sha256()
    for index in sorted(int(item) for item in calibration_indices):
        digest.update(f"{index}:{ids[index]}\n".encode("utf-8"))
    return CalibrationSplit(
        train_indices=[int(item) for item in train_indices],
        calibration_indices=[int(item) for item in calibration_indices],
        train_groups=len(train_groups),
        calibration_groups=len(calibration_groups),
        fingerprint=digest.hexdigest(),
    )


# 将原训练集拆成优化集和校准集，沿用验证集与测试集；仅打乱优化集。
def build_router_dataloaders(
    upstream: dict[str, DataLoader],
    microbatch_size: int,
    num_workers: int,
    split_seed: int = 20260903,
) -> tuple[dict[str, DataLoader], CalibrationSplit]:
    split = make_calibration_split(upstream["train"].dataset, seed=split_seed)
    datasets: dict[str, Dataset] = {
        "train": Subset(upstream["train"].dataset, split.train_indices),
        "calibration": Subset(
            upstream["train"].dataset, split.calibration_indices
        ),
        "valid": upstream["valid"].dataset,
        "test": upstream["test"].dataset,
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=microbatch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        for name, dataset in datasets.items()
    }
    return loaders, split


# 把各模态张量、长度和主任务标签移到目标设备，保留原批次的其他字段。
def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("text", "audio", "vision", "text_lengths", "audio_lengths", "vision_lengths"):
        moved[key] = batch[key].to(device, non_blocking=True)
    moved["labels"] = dict(batch["labels"])
    moved["labels"]["M"] = batch["labels"]["M"].view(-1).to(
        device, non_blocking=True
    )
    return moved


# 仅对选中样本的有效音频片段，按随机信噪比加入高斯噪声。
def _add_audio_noise(
    audio: torch.Tensor,
    lengths: torch.Tensor,
    selected: torch.Tensor,
    snr_low: float,
    snr_high: float,
) -> None:
    for index in torch.nonzero(selected, as_tuple=False).flatten().tolist():
        length = int(lengths[index].item())
        valid = audio[index, :length]
        signal_power = valid.float().pow(2).mean()
        if not bool(torch.isfinite(signal_power)) or float(signal_power) <= 0.0:
            continue
        snr = torch.empty((), device=audio.device).uniform_(snr_low, snr_high)
        noise_power = signal_power / torch.pow(10.0, snr / 10.0)
        noise = torch.randn_like(valid) * noise_power.sqrt().to(valid.dtype)
        audio[index, :length] = valid + noise


# 在选中样本的有效视觉序列内，随机选取一段连续区间置零。
def _mask_vision_span(
    vision: torch.Tensor,
    lengths: torch.Tensor,
    selected: torch.Tensor,
    fraction_low: float,
    fraction_high: float,
) -> None:
    for index in torch.nonzero(selected, as_tuple=False).flatten().tolist():
        length = max(1, int(lengths[index].item()))
        fraction = float(
            torch.empty((), device=vision.device).uniform_(
                fraction_low, fraction_high
            )
        )
        span = max(1, min(length, int(round(length * fraction))))
        max_start = length - span
        start = int(
            torch.randint(max_start + 1, (), device=vision.device).item()
        )
        vision[index, start : start + span] = 0


# 训练期随机丢弃至多一个模态，并对仍存在的音视频施加扰动，返回显式存在掩码。
def augment_modalities(
    batch: dict[str, Any],
    modality_drop_probability: float = 0.30,
    audio_noise_probability: float = 0.20,
    vision_mask_probability: float = 0.20,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Apply the stage-training corruptions and return an explicit presence mask."""
    augmented = dict(batch)
    # 音视频扰动前复制张量，避免原地修改调用方持有的干净输入。
    augmented["audio"] = batch["audio"].clone()
    augmented["vision"] = batch["vision"].clone()
    batch_size = batch["text"].shape[0]
    device = batch["text"].device
    presence = torch.ones(batch_size, 3, device=device, dtype=torch.float32)
    drop_selected = torch.rand(batch_size, device=device) < modality_drop_probability
    dropped_modality = torch.randint(3, (batch_size,), device=device)
    rows = torch.nonzero(drop_selected, as_tuple=False).flatten()
    if rows.numel():
        presence[rows, dropped_modality[rows]] = 0.0

    audio_selected = (
        (torch.rand(batch_size, device=device) < audio_noise_probability)
        & presence[:, MODALITY_INDEX["audio"]].bool()
    )
    _add_audio_noise(
        augmented["audio"],
        batch["audio_lengths"],
        audio_selected,
        5.0,
        20.0,
    )
    vision_selected = (
        (torch.rand(batch_size, device=device) < vision_mask_probability)
        & presence[:, MODALITY_INDEX["vision"]].bool()
    )
    _mask_vision_span(
        augmented["vision"],
        batch["vision_lengths"],
        vision_selected,
        0.10,
        0.30,
    )
    return augmented, presence
