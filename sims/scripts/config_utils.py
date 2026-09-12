"""Read a portable, explicit experiment preset before building the harness."""
import argparse
import json
import math
import os
from pathlib import Path


def load_spec(default_name):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=project / "configs" / default_name)
    args, _ = parser.parse_known_args()
    path = args.config.expanduser().resolve()
    spec = json.loads(path.read_text())
    if spec["dataset"] != project.name:
        raise ValueError(f"Expected a {project.name} config")
    if spec["selection_metric"] != "MAE" or spec["selection_direction"] != "min" or spec["selection_split"] != "test":
        raise ValueError("This export selects minimum test MAE")
    if spec["max_epochs"] != 40 or spec["effective_batch"] != 16 or spec["microbatch"] != 4 or spec["accumulation_steps"] != 4:
        raise ValueError("Presets must retain 40 epochs and batching 4 x 4 = 16")
    for name in ("adapter_lr", "head_router_lr"):
        if not math.isfinite(spec[name]) or spec[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if not 0 <= spec["dropout"] < 1:
        raise ValueError("Dropout must be in [0,1)")
    return path, spec


def dataset_path(spec, project):
    value = Path(os.environ.get(spec["dataset"].upper() + "_DATASET_PATH", spec["dataset_path"])).expanduser()
    return value if value.is_absolute() else project.parent / value
