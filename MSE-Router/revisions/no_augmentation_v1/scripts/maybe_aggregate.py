#!/usr/bin/env python3
"""The last successful training job builds the report without another allocation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path


def complete(root: Path) -> bool:
    for seed in (1111, 2222, 3333):
        for name in ("result.json", "manifest.json"):
            path = root / f"seed_{seed}" / name
            if not path.is_file():
                return False
            with path.open(encoding="utf-8") as handle:
                if json.load(handle).get("status") != "ok":
                    return False
    return True


def main() -> None:
    revision = Path(__file__).resolve().parents[1]
    router_root = revision.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=router_root / "outputs/qwen-mosei-router-v2-no-augmentation")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not complete(root):
        print("Comparison pending: all three seeds must complete successfully.", flush=True)
        return
    # OS releases the lock if the process exits; concurrent finishers serialize.
    with (root / ".aggregation.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        summary = root / "three_seed_comparison.json"
        if summary.exists():
            result = json.loads(summary.read_text(encoding="utf-8"))
            if result.get("status") == "ok" and (root / "three_seed_comparison.md").is_file():
                print("Comparison was already completed by another seed.", flush=True)
                return
            raise RuntimeError("An earlier aggregation failed; preserving its artifacts for diagnosis.")
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        subprocess.run([
            sys.executable, "-u", str(revision / "scripts/run_qwen_mosei_router.py"),
            "aggregate", "--training-augmentation", "none", "--seeds", "1111", "2222", "3333",
            "--output-root", str(root), "--baseline-root", str(router_root / "outputs/qwen-mosei-router-v2"),
        ], check=True, env=environment)


if __name__ == "__main__":
    main()
