"""完整训练集的批次加载与设备搬运，不划分校准集、不添加随机扰动。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader


# 随运行清单保存实际输入策略，便于核对本次训练没有启用数据增强。
INPUT_POLICY = {
    "full_training_split": True,
    "modality_dropout": False,
    "audio_noise": False,
    "vision_masking": False,
    "calibration_enabled": False,
    "calibration_samples": 0,
    "temperatures": [1.0, 1.0, 1.0],
}


# 仅记录完整训练集的样本索引和视频组数，不抽取或排除样本。
@dataclass(frozen=True)
class TrainingInventory:
    train_indices: list[int]
    train_groups: int
    fingerprint: str


# 保留数据文件原有 train、valid、test 划分，只让训练集按种子打乱顺序。
def build_router_dataloaders(
    upstream: dict[str, DataLoader], microbatch_size: int, num_workers: int,
) -> tuple[dict[str, DataLoader], TrainingInventory]:
    train_dataset = upstream["train"].dataset
    ids = [item.decode("utf-8") if isinstance(item, bytes) else str(item)
           for item in train_dataset.ids]
    if len(ids) != len(train_dataset):
        raise RuntimeError("training IDs and sample count differ")
    indices = list(range(len(train_dataset)))
    fingerprint = hashlib.sha256(json.dumps(
        {"ids": ids, "train_indices": indices, "policy": INPUT_POLICY},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    inventory = TrainingInventory(
        train_indices=indices,
        train_groups=len({item.split("$_$", maxsplit=1)[0] for item in ids}),
        fingerprint=fingerprint,
    )
    loaders = {
        name: DataLoader(
            upstream[name].dataset,
            batch_size=microbatch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )
        for name in ("train", "valid", "test")
    }
    return loaders, inventory


# 仅搬运张量和标签到目标设备，不修改音视频数值或文本有效长度。
def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("text", "audio", "vision", "text_lengths", "audio_lengths", "vision_lengths"):
        moved[key] = batch[key].to(device, non_blocking=True)
    moved["labels"] = dict(batch["labels"])
    moved["labels"]["M"] = batch["labels"]["M"].view(-1).to(
        device, non_blocking=True
    )
    return moved
