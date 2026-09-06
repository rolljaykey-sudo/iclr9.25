#!/usr/bin/env python3
"""Verify the upstream SIMS v2 artifact and inspect its supervised splits."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1].parent / "datasets/MSA/data/SIMS_V2/ch-simsv2s.pkl"
EXPECTED_SIZE = 3620390813
EXPECTED_SHA256 = "f8fd9a1dd070588a714ff357b00f31d5bb93277b4e502cf2dc4ff718a95fc49b"


def main():
    partial = DATA.with_name(DATA.name + ".download.part")
    source = DATA if DATA.is_file() else partial
    if source.stat().st_size != EXPECTED_SIZE:
        raise ValueError(f"Wrong file size: {source.stat().st_size}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError("SIMS v2 SHA-256 does not match the publisher's LFS record")
    if source == partial:
        if DATA.exists():
            raise FileExistsError(DATA)
        partial.rename(DATA)
    with DATA.open("rb") as handle:
        data = pickle.load(handle)
    report = {
        "status": "ok", "dataset": "simsv2", "path": str(DATA),
        "size": EXPECTED_SIZE, "sha256": EXPECTED_SHA256,
        "source": "https://huggingface.co/datasets/AZYoung/SIMSV2_processed",
        "source_revision": "0bb42223a6f6099680865fbd807ef26804c43c5d",
        "top_level_keys": list(data), "splits": {},
    }
    split_ids = {}
    for name in ("train", "valid", "test"):
        subset = data[name]
        labels = np.asarray(subset["regression_labels"]).reshape(-1)
        count = len(labels)
        if not np.isfinite(labels).all() or labels.min() < -1 or labels.max() > 1:
            raise ValueError(f"Invalid {name} labels")
        ids = [str(x) for x in subset["id"]]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"Missing or duplicate {name} IDs")
        split_ids[name] = set(ids)
        record = {
            "samples": count, "keys": list(subset),
            "label_min": float(labels.min()), "label_max": float(labels.max()),
            "label_values": np.unique(labels).tolist(), "id_examples": ids[:8],
            "raw_text_examples": [str(x) for x in subset["raw_text"][:2]],
        }
        if len(subset["raw_text"]) != count:
            raise ValueError(f"Wrong {name} text count")
        for modality, dim in (("audio", 25), ("vision", 177)):
            features = np.asarray(subset[modality])
            lengths = np.asarray(subset[modality + "_lengths"]).reshape(-1)
            if features.ndim != 3 or features.shape[0] != count or features.shape[2] != dim:
                raise ValueError(f"Wrong {name} {modality} shape: {features.shape}")
            if len(lengths) != count or lengths.min() < 1 or lengths.max() > features.shape[1]:
                raise ValueError(f"Invalid {name} {modality} lengths")
            record[modality] = {
                "shape": list(features.shape), "dtype": str(features.dtype),
                "min_length": int(lengths.min()), "max_length": int(lengths.max()),
                "nan_count": int(np.isnan(features).sum()),
                "posinf_count": int(np.isposinf(features).sum()),
                "neginf_count": int(np.isneginf(features).sum()),
            }
        report["splits"][name] = record
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if split_ids[a] & split_ids[b]:
            raise ValueError(f"Overlapping IDs: {a}, {b}")
    destination = ROOT / "dataset_identity.json"
    with destination.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
