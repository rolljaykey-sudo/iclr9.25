import argparse
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_qwen_simsv2 as runner


class QwenSimsv2ProtocolTests(unittest.TestCase):
    def test_protocol_is_qwen_clean_simsv2(self):
        self.assertEqual(runner.SEEDS, (1111, 2222, 3333))
        self.assertEqual(runner.PROTOCOL["backbone"], "qwen-1.8b")
        self.assertEqual(runner.PROTOCOL["training_augmentation"], "none")
        torch.testing.assert_close(
            torch.tensor(runner.PROTOCOL["ordinal_anchors"]),
            torch.linspace(-1.0, 1.0, 7),
        )

    def test_dataset_and_model_files_match_inspected_inputs(self):
        args = argparse.Namespace(dataset_path=runner.DATASET, model_path=runner.MODEL)
        files = runner.validate_files(args, hash_dataset=False)
        self.assertEqual(files["dataset"]["size"], 3620390813)
        self.assertEqual(files["model"]["backbone"], "qwen-1.8b")
        self.assertEqual(set(files["model"]["files"]), set(runner.REQUIRED_MODEL_FILES))


if __name__ == "__main__":
    unittest.main()
