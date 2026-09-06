import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_chatglm3_simsv2 as runner
from mse_router.data import make_calibration_split, prepare_training_batch, video_group
from mse_router.math_utils import ANCHORS, soft_ordinal_targets
from mse_router.model import QwenMseRouter


class Simsv2ProtocolTests(unittest.TestCase):
    def test_ordinal_targets_reconstruct_native_labels(self):
        labels = torch.tensor([-1., -.8, -.4, -.1, 0., .2, .6, 1.])
        targets = soft_ordinal_targets(labels)
        self.assertEqual(tuple(targets.shape), (8, 7))
        torch.testing.assert_close(targets.sum(-1), torch.ones(8))
        torch.testing.assert_close(targets @ ANCHORS, labels)
        self.assertTrue(bool((targets >= 0).all()))

    def test_parser_enforces_simsv2_range(self):
        values, info = QwenMseRouter.parse_responses(
            ["情感为-1.0", "+1.0", "0.6", "-1.1", "2.0", "没有数值"]
        )
        self.assertEqual(values, [-1., 1., .6, 0., 0., 0.])
        self.assertEqual(info["out_of_range_indices"], [3, 4])
        self.assertEqual(info["invalid_indices"], [5])

    def test_clean_training_preserves_inputs_and_rng(self):
        batch = {"text": torch.ones(3, 3, 50), "audio": torch.randn(3, 10, 25),
                 "vision": torch.randn(3, 8, 177)}
        before = {k: v.clone() for k, v in batch.items()}
        state = torch.get_rng_state().clone()
        clean, presence = prepare_training_batch(batch, "none")
        for key in before:
            torch.testing.assert_close(clean[key], before[key])
        torch.testing.assert_close(presence, torch.ones(3, 3))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_dataset_specific_configuration_and_metrics(self):
        args = argparse.Namespace(dataset_path=runner.DATASET, output_root=runner.OUTPUT_ROOT,
                                  model_path=runner.MODEL, num_workers=0)
        config = runner.build_config(args, 4, 4)
        self.assertEqual(config.datasetName, "simsv2")
        self.assertEqual(tuple(config.feature_dims), (4096, 25, 177))
        self.assertEqual(config.v_lstm_hidden_size, 64)
        self.assertIn("[-1.0, 1.0]", config.task_specific_prompt)
        self.assertEqual(config.batch_size * config.update_epochs, 16)
        from utils.metricsTop import MetricsTop
        labels = torch.linspace(-1, 1, 11)
        metrics = MetricsTop(config).getMetics("simsv2")(labels, labels)
        self.assertAlmostEqual(metrics["MAE"], 0.)
        self.assertAlmostEqual(metrics["Corr"], 1.)
        self.assertAlmostEqual(metrics["Mult_acc_5"], 1.)

    def test_calibration_is_repeatable_and_group_disjoint(self):
        class Dataset:
            ids = [f"video_{g:04d}$_${i}" for g in range(30) for i in range(7)]
            labels = {"M": np.tile(np.linspace(-1, 1, 7), 30)}

            def __len__(self):
                return len(self.ids)

        dataset = Dataset()
        first, second = make_calibration_split(dataset), make_calibration_split(dataset)
        self.assertEqual(first, second)
        a = {video_group(dataset.ids[i]) for i in first.train_indices}
        b = {video_group(dataset.ids[i]) for i in first.calibration_indices}
        self.assertFalse(a & b)
        self.assertEqual(set(first.train_indices) | set(first.calibration_indices), set(range(len(dataset))))

    def test_three_seed_report_uses_sample_std_and_correct_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed, value in zip(runner.SEEDS, [.2, .3, .4]):
                folder = root / f"seed_{seed}"
                folder.mkdir()
                test = {k: value for k in ("MAE", "Corr", "Mult_acc_2", "Mult_acc_2_weak",
                                           "Mult_acc_3", "Mult_acc_5", "F1_score", "R_squre")}
                test.update(condition="clean", samples=10)
                (folder / "result.json").write_text(json.dumps({
                    "status": "ok", "seed": seed, "dataset_protocol": runner.PROTOCOL, "test": test,
                }))
            with patch.object(runner, "dataset_identity", return_value={"splits": {"test": {"samples": 10}}}):
                summary = runner.aggregate_results(root)
                again = runner.aggregate_results(root)
            self.assertEqual(summary, again)
            self.assertEqual(summary["seeds"], [1111, 2222, 3333])
            self.assertAlmostEqual(summary["metrics"]["MAE"]["mean"], .3)
            self.assertAlmostEqual(summary["metrics"]["MAE"]["sample_std"], .1)

    def test_train_persists_protocol_before_aggregation(self):
        import mse_router.model as router_model

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed_dir = root / "seed_1111"
            seed_dir.mkdir()
            preflight = root / "preflight.json"
            preflight.write_text(json.dumps({"dataset_protocol": runner.PROTOCOL}))
            args = argparse.Namespace(output_root=root, seed=1111, preflight_result=preflight)

            def simulated_training(parsed):
                result = {"status": "ok", "seed": parsed.seed}
                (seed_dir / "result.json").write_text(json.dumps(result))
                (seed_dir / "manifest.json").write_text(json.dumps({"status": "ok"}))
                return result

            identity = {"splits": {name: {"samples": 1} for name in ("train", "valid", "test")}}
            with patch.dict(vars(runner.harness), vars(runner.harness).copy()), \
                 patch.object(runner.harness, "train", simulated_training), \
                 patch.object(runner, "dataset_identity", return_value=identity), \
                 patch.object(router_model, "QwenMseRouter", router_model.QwenMseRouter):
                runner.configure_harness(args)
                returned = runner.harness.train(args)
            saved = json.loads((seed_dir / "result.json").read_text())
            self.assertEqual(saved, returned)
            self.assertEqual(saved["dataset_protocol"], runner.PROTOCOL)

    def test_aggregation_waits_for_seed_protocol_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed in runner.SEEDS:
                directory = root / f"seed_{seed}"
                directory.mkdir()
                (directory / "result.json").write_text(json.dumps({"status": "ok", "seed": seed}))
            self.assertIsNone(runner.aggregate_results(root, require_all=False))
            self.assertFalse((root / "three_seed_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
