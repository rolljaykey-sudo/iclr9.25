#!/usr/bin/env python3
"""Run original CPU tests; report historical external-artifact exclusions."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = ("no_augmentation_v1", "simsv2_no_augmentation_v1")
HISTORICAL_CHECKS = {
    "test_glm_settings_and_files_match_original",
    "test_active_qwen_source_fingerprints_are_unchanged",
    "test_batching_matches_original_1111_and_2222",
    "test_data_model_and_backbone_match_original_v2",
    "test_existing_qwen_and_glm_source_fingerprints_are_unchanged",
    "test_preflight_rejects_legacy_batch64_and_other_model_identity",
    "test_dataset_and_model_files_match_inspected_inputs",
}


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", choices=REVISIONS)
    args = parser.parse_args()
    if args.revision is None:
        env = dict(os.environ, MSE_ADAPTER_ROOT=str(ROOT / "MSE-Adapter"),
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES="")
        for revision in REVISIONS:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--revision", revision],
                           env=env, check=True, cwd=ROOT)
        return
    revision = ROOT / "MSE-Router/revisions" / args.revision
    os.environ["MSE_ADAPTER_ROOT"] = str(ROOT / "MSE-Adapter")
    sys.path[:0] = [str(revision), str(revision / "scripts")]
    suite = unittest.defaultTestLoader.discover(str(revision / "tests"))
    selected = unittest.TestSuite()
    for test in flatten(suite):
        if test.id().split(".")[-1] in HISTORICAL_CHECKS:
            print(f"EXCLUDED (requires historical results and model/dataset files): {test.id()}", flush=True)
        else:
            selected.addTest(test)
    result = unittest.TextTestRunner(verbosity=2).run(selected)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
