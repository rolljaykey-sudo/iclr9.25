"""Official MOSI splits, raw Qwen tokens, and authoritative stored lengths."""
from __future__ import annotations

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MosiDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.ids = [str(x) for x in data["id"]]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate sample IDs")
        self.raw_text = [str(x) for x in data["raw_text"]]
        tokens = tokenizer(self.raw_text, padding="max_length", truncation=True, max_length=max_length,
                           add_special_tokens=False, return_tensors="pt")
        self.text = torch.stack((tokens["input_ids"], tokens["attention_mask"], torch.zeros_like(tokens["input_ids"])), 1)
        self.labels = torch.from_numpy(np.asarray(data["regression_labels"], dtype=np.float32).reshape(-1).copy())
        if not bool(torch.isfinite(self.labels).all()) or bool((self.labels.abs() > 3).any()):
            raise ValueError("MOSI labels must be finite in [-3,3]")
        self.features, self.lengths, self.audit = {}, {}, {}
        for m in ("audio", "vision"):
            array = np.asarray(data[m], dtype=np.float32)
            lengths = np.asarray(data[m + "_lengths"]).reshape(-1)
            if not np.all(np.isfinite(lengths)) or not np.all(lengths == np.floor(lengths)):
                raise ValueError("nonintegral feature lengths")
            lengths = lengths.astype(np.int64)
            if array.shape[0] != len(self.ids) or np.any(lengths < 0) or np.any(lengths > array.shape[1]):
                raise ValueError("invalid feature shape/lengths")
            if not np.isfinite(array).all():
                raise ValueError("non-finite features require explicit preprocessing")
            mask = np.arange(array.shape[1])[None] < lengths[:, None]
            self.audit[m] = dict(shape=list(array.shape), min_length=int(lengths.min()), max_length=int(lengths.max()),
                                 nonzero_outside_length=int(np.count_nonzero(array[~mask])))
            self.features[m] = torch.from_numpy(array.copy())
            self.lengths[m] = torch.from_numpy(lengths.copy())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return dict(id=self.ids[index], raw_text=self.raw_text[index], text=self.text[index],
                    text_lengths=self.text[index, 1].sum(),
                    audio=self.features["audio"][index], audio_lengths=self.lengths["audio"][index],
                    vision=self.features["vision"][index], vision_lengths=self.lengths["vision"][index],
                    labels={"M": self.labels[index]})


def load_datasets(config, tokenizer):
    with open(config["dataset_path"], "rb") as handle:
        source = pickle.load(handle)
    datasets = {name: MosiDataset(source[name], tokenizer, config["text_max_length"]) for name in ("train", "valid", "test")}
    del source
    seen = set()
    for name, dataset in datasets.items():
        if len(dataset) != config["splits"][name]:
            raise ValueError(f"unexpected {name} split size")
        if seen.intersection(dataset.ids):
            raise ValueError("sample ID overlap between official splits")
        seen.update(dataset.ids)
        for m in ("audio", "vision"):
            if dataset.features[m].shape[-1] != config["model"][m + "_features"]:
                raise ValueError(f"{m} feature dimension mismatch")
    return datasets


def make_loaders(datasets, settings):
    return {name: DataLoader(dataset, batch_size=settings["microbatch"], shuffle=name == "train",
                             num_workers=settings["num_workers"], pin_memory=True)
            for name, dataset in datasets.items()}


def sequential_loader(loader):
    return DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False, num_workers=0,
                      pin_memory=loader.pin_memory)


def move_batch(batch, device):
    result = dict(batch)
    for key in ("text", "audio", "vision", "text_lengths", "audio_lengths", "vision_lengths"):
        result[key] = batch[key].to(device, non_blocking=True)
    result["labels"] = {"M": batch["labels"]["M"].to(device, non_blocking=True)}
    return result


def unpack(batch):
    return (batch["labels"]["M"].reshape(-1), (batch["text"], batch["text_lengths"]),
            (batch["audio"], batch["audio_lengths"]), (batch["vision"], batch["vision_lengths"]))
