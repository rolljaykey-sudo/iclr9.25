from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

REVISION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION))
from mse_router.data import augment_modalities, augmentation_config, prepare_training_batch
from mse_router.comparison import METRICS, build_comparison, render_comparison
from mse_router.trainer import RouterTrainer, TrainingSettings

spec = importlib.util.spec_from_file_location("noaug_runner", REVISION / "scripts/run_qwen_mosei_router.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
helper_spec = importlib.util.spec_from_file_location("noaug_aggregation", REVISION / "scripts/maybe_aggregate.py")
aggregation = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(aggregation)


def batch(size=4):
    return {
        "text": torch.arange(size * 3 * 5).reshape(size, 3, 5).float(),
        "audio": torch.ones(size, 6, 2), "vision": torch.ones(size, 6, 2),
        "text_lengths": torch.full((size,), 5),
        "audio_lengths": torch.full((size,), 6), "vision_lengths": torch.full((size,), 6),
        "labels": {"M": torch.zeros(size, 1)},
    }


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("llm", "text_pool", "text_projection", "text_adapter", "audio_encoder", "audio_adapter", "vision_encoder", "vision_adapter", "ordinal_head"):
            setattr(self, name, torch.nn.Dropout(0.1))
        self.router = torch.nn.Linear(1, 1)
        self.seen = []

    def forward(self, labels, text, audio, vision, presence, stage):
        self.seen.append((stage, presence.clone()))
        loss = self.router(torch.ones(len(labels), 1)).square().mean()
        return {"Loss": loss, "GenerationLoss": loss, "AuxiliaryLoss": loss * 0}


class AugmentationTests(unittest.TestCase):
    def test_none_preserves_inputs_and_rng(self):
        original = batch()
        before = copy.deepcopy(original)
        rng = torch.get_rng_state().clone()
        with patch("mse_router.data.augment_modalities", side_effect=AssertionError("must bypass")):
            changed, presence = prepare_training_batch(original, "none")
        self.assertIs(changed, original)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for key in original:
            if key != "labels":
                self.assertTrue(torch.equal(changed[key], before[key]))
        self.assertTrue(torch.equal(changed["labels"]["M"], before["labels"]["M"]))
        self.assertTrue(torch.equal(presence, torch.ones(4, 3)))

    def test_standard_is_exact_original_behavior(self):
        original = batch(32)
        torch.manual_seed(97)
        expected, expected_presence = augment_modalities(original)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(97)
        actual, actual_presence = prepare_training_batch(original, "standard")
        self.assertTrue(torch.equal(actual_presence, expected_presence))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for key in ("text", "audio", "vision"):
            self.assertTrue(torch.equal(actual[key], expected[key]))

    def test_invalid_modes_rejected(self):
        for value in ("clean", "off", ""):
            with self.assertRaises(ValueError):
                prepare_training_batch(batch(), value)
            with self.assertRaises(ValueError):
                TrainingSettings(training_augmentation=value)

    def test_both_training_stages_use_none_and_preserve_dropout_policy(self):
        trainer = RouterTrainer.__new__(RouterTrainer)
        trainer.args = SimpleNamespace(device=torch.device("cpu"))
        trainer.settings = TrainingSettings(training_augmentation="none")
        model = ToyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        with warnings.catch_warnings(), patch("mse_router.trainer.prepare_training_batch", wraps=prepare_training_batch) as prepare:
            warnings.simplefilter("ignore")
            for stage in ("stage1", "router"):
                trainer._train_epoch(model, [batch()], optimizer, Mock(), torch.cuda.amp.GradScaler(enabled=False), stage, 1, 1)
                self.assertTrue(model.router.training)
                self.assertEqual(model.ordinal_head.training, stage == "stage1")
            self.assertEqual([call.args[1] for call in prepare.call_args_list], ["none", "none"])
        self.assertEqual([stage for stage, _ in model.seen], ["stage1", "router"])
        self.assertTrue(all(torch.equal(presence, torch.ones(4, 3)) for _, presence in model.seen))

    def test_original_model_math_and_shared_sources_untouched(self):
        snapshot = json.loads((REVISION / "upstream_snapshot.json").read_text())
        workspace = REVISION.parents[2]
        for name, expected in snapshot["files"].items():
            self.assertEqual(hashlib.sha256((workspace / name).read_bytes()).hexdigest(), expected)
        for name in ("model.py", "math_utils.py"):
            self.assertEqual((REVISION / "mse_router" / name).read_bytes(), (runner.ROUTER_ROOT / "mse_router" / name).read_bytes())


class RunnerTests(unittest.TestCase):
    def parse(self, *args):
        with patch.object(sys, "argv", ["runner", *args]):
            return runner.parse_args()

    def test_cli_defaults_and_custom_three_seeds(self):
        default = self.parse("preflight")
        self.assertEqual(default.training_augmentation, "standard")
        self.assertIn("augmentation-control", str(default.output_root))
        args = self.parse("aggregate", "--training-augmentation", "none", "--seeds", "1111", "2222", "3333")
        self.assertEqual(args.seeds, [1111, 2222, 3333])
        self.assertEqual(args.preflight_result, args.output_root / "preflight/result.json")
        self.assertEqual(runner.ROUTER_ROOT.name, "MSE-Router")
        self.assertEqual(runner.ADAPTER_DIR.name, "MSE-Adapter")

    def test_duplicate_seeds_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("aggregate", "--seeds", "1111", "1111")

    def test_protects_historical_and_existing_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "outputs/baseline"
            args = SimpleNamespace(mode="train", seed=1111, output_root=root / "outputs/new")
            with patch.object(runner, "ROUTER_ROOT", root), patch.object(runner, "BASELINE_ROOT", baseline):
                directory = runner.claim_run_directory(args)
                marker = directory / "result.json"
                marker.write_text("keep")
                with self.assertRaises(FileExistsError):
                    runner.claim_run_directory(args)
                self.assertEqual(marker.read_text(), "keep")
                args.output_root = baseline
                with self.assertRaises(ValueError):
                    runner.claim_run_directory(args)
                args.output_root = root / "elsewhere"
                with self.assertRaises(ValueError):
                    runner.claim_run_directory(args)

    def test_preflight_mode_and_batch_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preflight.json"
            args = SimpleNamespace(preflight_result=path, training_augmentation="none", router_variant="full")
            base = {
                "status": "ok", "training_augmentation": augmentation_config("none"),
                "training_settings": asdict(TrainingSettings(training_augmentation="none")),
                "router_variant": "full", "selected_batching": {"microbatch": 4, "accumulation": 4, "effective_batch": 16},
                "provenance": {}, "files": {"dataset": {"path": "same", "size": 1, "mtime_ns": 2}, "model": {"files": {}}},
            }
            for change in ({"training_augmentation": augmentation_config("standard")}, {"training_augmentation": None}, {"selected_batching": {"microbatch": 2, "accumulation": 8, "effective_batch": 16}}):
                path.write_text(json.dumps({**base, **change}))
                with self.assertRaises(RuntimeError):
                    runner.verify_preflight(args)
            path.write_text(json.dumps(base))
            with patch.object(runner, "source_provenance", return_value={}), patch.object(runner, "validate_files", return_value=base["files"]):
                self.assertEqual(runner.verify_preflight(args)["status"], "ok")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.new, self.old = self.root / "new", self.root / "old"
        self.seeds = [1111, 2222, 3333]
        for index, seed in enumerate(self.seeds):
            for root, is_new in ((self.old, False), (self.new, True)):
                directory = root / f"seed_{seed}"
                directory.mkdir(parents=True)
                settings = asdict(TrainingSettings(training_augmentation="none"))
                if not is_new:
                    settings.pop("training_augmentation")
                metrics = {key: 0.75 for key in METRICS}
                metrics.update(MAE=0.5 + index * 0.01 - (0.01 if is_new else 0), Corr=0.77, condition="clean", samples=4659, invalid_generations=0, out_of_range_generations=0)
                result = {"status": "ok", "seed": seed, "router_variant": "full", "test": metrics, "training_settings": settings, "elapsed_seconds": 100, "peak_gpu_memory_bytes": 1000, "robustness": {}}
                manifest = {"status": "ok", "seed": seed, "router_variant": "full", "architecture": "v2", "batching": {"microbatch": 4, "accumulation": 4, "effective_batch": 16}, "split": {"fingerprint": "same"}, "files": {"dataset": "same", "model": "same"}}
                if is_new:
                    result.update(training_augmentation=augmentation_config("none"), checkpoint_selection="best_router_stage_valid_mae", run_robustness=False)
                    manifest["training_augmentation"] = augmentation_config("none")
                (directory / "result.json").write_text(json.dumps(result))
                (directory / "manifest.json").write_text(json.dumps(manifest))
                for stage in ("stage1", "router"):
                    (directory / f"{stage}_history.json").write_text(json.dumps([{"epoch": 1, "elapsed_seconds": 10, "train": {"loss": 1.0}, "valid": {"MAE": 0.5}}]))

    def compare(self):
        return build_comparison(self.new, self.old, self.seeds, "none")

    def change(self, filename, **changes):
        path = self.new / "seed_1111" / filename
        value = json.loads(path.read_text())
        value.update(changes)
        path.write_text(json.dumps(value))

    def test_three_seed_report_and_paired_statistics(self):
        result = self.compare()
        self.assertEqual(result["seeds"], self.seeds)
        self.assertEqual(result["conclusion"], "all_seeds_improved")
        self.assertAlmostEqual(result["metrics"]["MAE"]["paired_delta"]["mean"], -0.01)
        self.assertIn("1111", render_comparison(result))
        self.assertIn("总耗时不能直接", render_comparison(result))

    def test_last_finisher_runs_cpu_only_aggregation(self):
        with patch.object(sys, "argv", ["helper", "--output-root", str(self.new)]), patch.object(aggregation.subprocess, "run") as run:
            aggregation.main()
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertIn("1111", run.call_args.args[0])

    def test_unfinished_seeds_do_not_start_aggregation(self):
        self.change("manifest.json", status="running")
        with patch.object(sys, "argv", ["helper", "--output-root", str(self.new)]), patch.object(aggregation.subprocess, "run") as run:
            aggregation.main()
        run.assert_not_called()

    def test_second_finisher_does_not_overwrite_report(self):
        report = self.new / "three_seed_comparison.json"
        report.write_text(json.dumps({"status": "ok"}))
        (self.new / "three_seed_comparison.md").write_text("existing report")
        with patch.object(sys, "argv", ["helper", "--output-root", str(self.new)]), patch.object(aggregation.subprocess, "run") as run:
            aggregation.main()
        run.assert_not_called()
        self.assertEqual((self.new / "three_seed_comparison.md").read_text(), "existing report")

    def test_incomplete_seed_is_not_reported_as_success(self):
        self.change("result.json", status="error")
        with self.assertRaises(ValueError):
            self.compare()

    def test_mixed_mode_is_rejected(self):
        self.change("result.json", training_augmentation=augmentation_config("standard"))
        with self.assertRaises(ValueError):
            self.compare()

    def test_other_hyperparameter_changes_are_rejected(self):
        settings = asdict(TrainingSettings(training_augmentation="none", adapter_lr=0.0005))
        self.change("result.json", training_settings=settings)
        with self.assertRaises(ValueError):
            self.compare()

    def test_changed_split_is_rejected(self):
        self.change("manifest.json", split={"fingerprint": "different"})
        with self.assertRaises(ValueError):
            self.compare()

    def test_worse_results_are_preserved(self):
        for seed in self.seeds:
            path = self.new / f"seed_{seed}" / "result.json"
            value = json.loads(path.read_text())
            value["test"]["MAE"] += 0.10
            value["test"]["Corr"] -= 0.10
            path.write_text(json.dumps(value))
        result = self.compare()
        self.assertEqual(result["conclusion"], "mean_worsened")
        self.assertTrue(result["corr_mean_decreased"])


if __name__ == "__main__":
    unittest.main()
