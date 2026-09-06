#!/usr/bin/env python3
"""Check exported original sources without loading datasets or model weights."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
for name, expected in manifest["files"].items():
    path = ROOT / name
    assert path.is_file(), f"Missing original source: {name}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], name
    assert path.stat().st_size == expected["size"], name
sources = list(ROOT.glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
for directory in ("MSE-Router", "MSE-Adapter"):
    sources += list((ROOT / directory).rglob("*.py"))
for path in sources:
    ast.parse(path.read_text(), filename=str(path))
print(f"Verified {len(manifest['files'])} original files and parsed {len(sources)} Python files.")
