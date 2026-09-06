from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from dataclasses import asdict
from unittest.mock import patch

REVISION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION / "scripts"))
import run_chatglm3_noaug as glm
import mse_router.model as model_module
from mse_router.backbone_model import BackboneMseRouter
from mse_router.trainer import TrainingSettings


@contextlib.contextmanager
def configured_glm():
    names = (
        "UPSTREAM_DIR", "DEFAULT_MODEL", "DEFAULT_OUTPUT_ROOT", "BASELINE_ROOT",
        "ROUTER_SOURCE_FILES", "parse_args", "build_config", "validate_files",
    )
    previous = {name: getattr(glm.harness, name) for name in names}
    previous_model = model_module.QwenMseRouter
    try:
        glm.configure_harness()
        yield
    finally:
        for name, value in previous.items():
            setattr(glm.harness, name, value)
        model_module.QwenMseRouter = previous_model


class ChatGLMNoAugTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(sys, "argv", ["glm", *arguments]):
            return glm.parse_args()

    def test_requested_seeds_and_clean_only_defaults(self):
        for seed in (3333, 4444, 5555):
            args = self.parse("train", "--seed", str(seed))
            self.assertEqual(args.seed, seed)
            self.assertEqual(args.training_augmentation, "none")
            self.assertEqual(args.router_variant, "full")
            self.assertTrue(args.skip_robustness)
            self.assertEqual(args.output_root, glm.OUTPUT_ROOT)
            self.assertEqual(args.preflight_result, glm.OUTPUT_ROOT / "preflight/result.json")

    def test_rejects_unrequested_seeds_augmented_modes_and_unrelated_outputs(self):
        for arguments in (
            ("train",), ("train", "--seed", "1111"),
            ("preflight", "--training-augmentation", "standard"),
            ("preflight", "--router-variant", "uniform"),
            ("preflight", "--output-root", str(glm.BASELINE_ROOT)),
            ("preflight", "--output-root", str(glm.ROUTER_ROOT / "outputs/qwen-mosei-router-v2-no-augmentation")),
            ("aggregate",),
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(*arguments)

    def test_backbone_copy_is_exact_original_v2(self):
        self.assertEqual(
            (REVISION / "mse_router/backbone_model.py").read_bytes(),
            (glm.ROUTER_ROOT / "mse_router/backbone_model.py").read_bytes(),
        )

    def test_glm_settings_and_files_match_original(self):
        with configured_glm():
            args = self.parse("train", "--seed", "3333")
            config = glm.build_config(args, 4, 4)
            self.assertIs(model_module.QwenMseRouter, BackboneMseRouter)
            self.assertEqual(config.router_backbone, "chatglm3")
            self.assertEqual(config.router_hidden_size, 4096)
            self.assertEqual(config.training_augmentation, "none")
            self.assertEqual((config.batch_size, config.update_epochs), (4, 4))
            baseline = json.loads((glm.BASELINE_ROOT / "seed_1111/result.json").read_text())
            settings = asdict(TrainingSettings(training_augmentation="none"))
            settings.pop("training_augmentation")
            self.assertEqual(settings, baseline["training_settings"])
            manifest = json.loads((glm.BASELINE_ROOT / "seed_1111/manifest.json").read_text())
            actual = glm.validate_files(args, hash_dataset=False)
            self.assertEqual(actual["model"], manifest["files"]["model"])
            for key in ("path", "size", "mtime_ns"):
                self.assertEqual(actual["dataset"][key], manifest["files"]["dataset"][key])

    def test_glm_provenance_covers_backbone_and_launcher_not_qwen_upstream(self):
        with configured_glm():
            files = glm.harness.source_provenance()["files"]
            for name in ("mse_router/backbone_model.py", "mse_router/trainer.py", "scripts/run_chatglm3_noaug.py", "scripts/chatglm3_noaug.slurm"):
                self.assertIn(name, files)
            self.assertTrue(any("MSE-ChatGLM3-6B/models/ChatGLM3/modeling_chatglm.py" in name for name in files))
            self.assertFalse(any("MSE-Qwen-1.8B" in name for name in files))

    def test_active_qwen_source_fingerprints_are_unchanged(self):
        preflight = json.loads((glm.ROUTER_ROOT / "outputs/qwen-mosei-router-v2-no-augmentation/preflight/result.json").read_text())
        for name, expected in preflight["provenance"]["files"].items():
            path = (glm.harness.ADAPTER_DIR / name.removeprefix("upstream_adapter/")) if name.startswith("upstream_adapter/") else REVISION / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
