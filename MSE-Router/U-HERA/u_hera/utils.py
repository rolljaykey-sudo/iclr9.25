"""Reproducible artifact I/O and train-only utility target construction."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .sequence import masked_softmax


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False, encoding="utf8") as handle:
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False, encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
        temporary = handle.name
    os.replace(temporary, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = handle.name
    torch.save(value, temporary)
    os.replace(temporary, path)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def utility_targets(deltas, presence, floor=.001):
    if deltas.ndim != 2 or deltas.shape != presence.shape or deltas.shape[1] != 3:
        raise ValueError("utility deltas/presence must have shape [N,3]")
    valid = presence.bool()
    if not bool(torch.isfinite(deltas).all()):
        raise ValueError("non-finite reference utility")
    population = deltas[valid].float()
    scale = max(float(population.std(unbiased=False)) if population.numel() else 0., floor)
    scores = torch.cat((deltas.float(), deltas.new_zeros(deltas.shape[0], 1)), -1) / scale
    available = torch.cat((valid, torch.ones_like(valid[:, :1])), -1)
    return masked_softmax(scores, available), scale


def accumulation_windows(loader, steps):
    """Group microbatches so the final window is normalized by its actual size."""
    window = []
    for batch in loader:
        window.append(batch)
        if len(window) == steps:
            yield window
            window = []
    if window:
        yield window
