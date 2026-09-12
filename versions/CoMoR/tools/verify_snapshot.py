#!/usr/bin/env python3
"""Verify the archived source using only the Python standard library and bash."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    snapshot = json.loads((root / "provenance/source_snapshot.json").read_text())
    for relative, expected in snapshot["files"].items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch: {relative}")
    counts = {"hashed_files": len(snapshot["files"]), "python": 0, "json": 0, "slurm": 0}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
            counts["python"] += 1
        elif path.suffix == ".json":
            json.loads(path.read_text())
            counts["json"] += 1
        elif path.suffix == ".slurm":
            subprocess.run(["bash", "-n", str(path)], check=True)
            counts["slurm"] += 1
    print(json.dumps({"status": "ok", **counts}, indent=2))


if __name__ == "__main__":
    main()
