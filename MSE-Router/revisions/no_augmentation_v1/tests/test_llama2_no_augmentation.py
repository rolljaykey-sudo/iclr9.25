from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REVISION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION / "scripts"))
import run_llama2_noaug as llama
import mse_router.model as model_module
from mse_router.backbone_model import BackboneMseRouter
from mse_router.data import augmentation_config
from mse_router.trainer import TrainingSettings


@contextlib.contextmanager
def configured_llama():
    names = (
        "UPSTREAM_DIR", "DEFAULT_MODEL", "DEFAULT_OUTPUT_ROOT", "BASELINE_ROOT",
        "ROUTER_SOURCE_FILES", "parse_args", "build_config", "validate_files",
        "verify_preflight",
    )
    previous = {name: getattr(llama.harness, name) for name in names}
    previous_model = model_module.QwenMseRouter
    previous_path = list(sys.path)
    # The production launcher is a fresh process; emulate that for upstream
    # config/data/utils imports when GLM tests have already run in this process.
    def upstream_module(name):
        return name.split(".")[0] in {"config", "data", "utils"}
    previous_modules = {name: module for name, module in sys.modules.items() if upstream_module(name)}
    for name in previous_modules:
        sys.modules.pop(name)
    try:
        llama.configure_harness()
        yield
    finally:
        for name, value in previous.items():
            setattr(llama.harness, name, value)
        model_module.QwenMseRouter = previous_model
        sys.path[:] = previous_path
        for name in list(sys.modules):
            if upstream_module(name):
                sys.modules.pop(name)
        sys.modules.update(previous_modules)


class Llama2NoAugTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(sys, "argv", ["llama", *arguments]):
            return llama.parse_args()

    def test_requested_seeds_and_clean_only_defaults(self):
        for seed in (4444, 5555):
            args = self.parse("train", "--seed", str(seed))
            self.assertEqual(args.seed, seed)
            self.assertEqual(args.training_augmentation, "none")
            self.assertEqual(args.router_variant, "full")
            self.assertTrue(args.skip_robustness)
            self.assertEqual(args.output_root, llama.OUTPUT_ROOT)
            self.assertEqual(args.preflight_result, llama.OUTPUT_ROOT / "preflight/result.json")

    def test_rejects_unrequested_seeds_modes_outputs_and_batch64(self):
        for arguments in (
            ("train",), ("train", "--seed", "3333"),
            ("preflight", "--training-augmentation", "standard"),
            ("preflight", "--router-variant", "uniform"),
            ("preflight", "--output-root", str(llama.BASELINE_ROOT)),
            ("preflight", "--output-root", str(llama.ROUTER_ROOT / "outputs/chatglm3-mosei-router-v2-no-augmentation")),
            ("aggregate",),
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(*arguments)
        with self.assertRaises(ValueError):
            llama.build_config(self.parse("preflight"), 4, 16)

    def test_batching_matches_original_1111_and_2222(self):
        for seed in (1111, 2222):
            manifest = json.loads((llama.BASELINE_ROOT / f"seed_{seed}/manifest.json").read_text())
            self.assertEqual(llama.BATCHING, manifest["batching"])

    def test_llama_config_and_nonaugmentation_settings_match_original(self):
        with configured_llama():
            config = llama.build_config(self.parse("train", "--seed", "4444"), 4, 4)
            self.assertIs(model_module.QwenMseRouter, BackboneMseRouter)
            self.assertEqual(config.router_backbone, "llama2")
            self.assertEqual(config.router_hidden_size, 4096)
            self.assertEqual(tuple(config.feature_dims), (4096, 74, 35))
            self.assertEqual(config.training_augmentation, "none")
            self.assertEqual((config.batch_size, config.update_epochs), (4, 4))
            self.assertEqual(config.router_upstream_dir, str(llama.UPSTREAM.resolve()))
        tree = ast.parse((llama.ROUTER_ROOT / "mse_router/trainer.py").read_text())
        settings_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainingSettings")
        original = {node.target.id: ast.literal_eval(node.value) for node in settings_class.body if isinstance(node, ast.AnnAssign)}
        actual = asdict(TrainingSettings(training_augmentation="none"))
        actual.pop("training_augmentation")
        self.assertEqual(actual, original)

    def test_data_model_and_backbone_match_original_v2(self):
        baseline = json.loads((llama.BASELINE_ROOT / "preflight/result.json").read_text())
        actual = llama.validate_files(self.parse("preflight"), hash_dataset=False)
        self.assertEqual(actual["model"], baseline["files"]["model"])
        for key in ("path", "size", "mtime_ns"):
            self.assertEqual(actual["dataset"][key], baseline["files"]["dataset"][key])
        for name in ("model.py", "math_utils.py", "backbone_model.py"):
            self.assertEqual((REVISION / "mse_router" / name).read_bytes(), (llama.ROUTER_ROOT / "mse_router" / name).read_bytes())

    def test_provenance_covers_llama_launcher_and_upstream(self):
        with configured_llama():
            files = llama.harness.source_provenance()["files"]
            for name in ("mse_router/backbone_model.py", "mse_router/trainer.py", "scripts/run_llama2_noaug.py", "scripts/llama2_noaug.slurm"):
                self.assertIn(name, files)
            self.assertTrue(any("MSE-Llama2-7B/config/config_regression.py" in name for name in files))
            self.assertFalse(any("MSE-Qwen-1.8B" in name or "MSE-ChatGLM3-6B" in name for name in files))

    def test_existing_qwen_and_glm_source_fingerprints_are_unchanged(self):
        for backbone in ("qwen", "chatglm3"):
            preflight = json.loads((llama.ROUTER_ROOT / f"outputs/{backbone}-mosei-router-v2-no-augmentation/preflight/result.json").read_text())
            for name, expected in preflight["provenance"]["files"].items():
                path = (llama.harness.ADAPTER_DIR / name.removeprefix("upstream_adapter/")) if name.startswith("upstream_adapter/") else REVISION / name
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_preflight_rejects_legacy_batch64_and_other_model_identity(self):
        with configured_llama(), tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preflight.json"
            args = self.parse("train", "--seed", "4444", "--preflight-result", str(path))
            files = llama.validate_files(args, hash_dataset=False)
            base = {
                "status": "ok", "training_augmentation": augmentation_config("none"),
                "training_settings": asdict(TrainingSettings(training_augmentation="none")),
                "router_variant": "full", "selected_batching": llama.BATCHING,
                "provenance": {}, "files": files,
            }
            with patch.object(llama.harness, "source_provenance", return_value={}):
                path.write_text(json.dumps(base))
                self.assertEqual(llama.verify_preflight(args)["status"], "ok")
                for key, value in (
                    ("selected_batching", {"microbatch": 4, "accumulation": 16, "effective_batch": 64}),
                    ("training_augmentation", augmentation_config("standard")),
                    ("provenance", {"wrong": "source"}),
                ):
                    path.write_text(json.dumps({**base, key: value}))
                    with self.assertRaises(RuntimeError):
                        llama.verify_preflight(args)
                wrong_model = copy.deepcopy(base)
                wrong_model["files"]["model"]["path"] = "/another/model"
                path.write_text(json.dumps(wrong_model))
                with self.assertRaisesRegex(RuntimeError, "model identity"):
                    llama.verify_preflight(args)


if __name__ == "__main__":
    unittest.main()
