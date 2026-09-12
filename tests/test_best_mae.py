"""Exercise real checkpoint selection and serialization with CPU-only stand-ins."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_trainer(dataset):
    package = f"{dataset}_test_router"
    spec = importlib.util.spec_from_file_location(
        package, ROOT / dataset / "mse_router" / "__init__.py",
        submodule_search_locations=[str(ROOT / dataset / "mse_router")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    return __import__(package + ".trainer", fromlist=["RouterTrainer"])


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("epoch", torch.tensor(0))
        self.router_variant = "full"
        self.temperatures = torch.ones(3)

    def set_stage(self, stage):
        self.stage = stage

    def experiment_state_dict(self):
        return self.state_dict()

    def load_experiment_state_dict(self, state):
        self.load_state_dict(state)


class BestMAETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = {name: load_trainer(name) for name in ("mosi", "sims")}

    def trainer(self, module, directory, invalid=False):
        trainer = module.RouterTrainer.__new__(module.RouterTrainer)
        trainer.settings = module.TrainingSettings()
        trainer.output_dir = Path(directory)
        (trainer.output_dir / "checkpoints").mkdir()
        trainer._optimizer = lambda model, stage: None

        def train_epoch(model, loader, optimizer, scheduler, scaler, stage, epoch, max_epochs):
            model.epoch.fill_(epoch)
            return {"loss": 1.0}

        def evaluate(model, loader, mode):
            epoch = int(model.epoch)
            # MAE ties at 5 and 10. F1 peaks at 20, which must NOT be selected.
            mae = float("nan") if invalid else (0.2 if epoch in (5, 10) else 0.5)
            return {"MAE": mae, "Corr": epoch / 100,
                    "Non0_F1_score": 0.99 if epoch == 20 else 0.7,
                    "F1_score": 0.99 if epoch == 20 else 0.7}

        trainer._train_epoch = train_epoch
        trainer.evaluate = evaluate
        return trainer

    def test_minimum_mae_and_earliest_tie_survive_final_checkpoint_reload(self):
        for dataset, module in self.modules.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                trainer = self.trainer(module, directory)
                loaders = {"train": SimpleNamespace(dataset=[0]), "test": [0]}
                # _fit_stage needs a loader length, but this suite does not train.
                class Loader(list):
                    dataset = [0]
                loaders["train"] = Loader([0])
                model = TinyModel()
                with patch.object(module, "get_cosine_schedule_with_warmup", return_value=None), \
                     patch.object(module.torch.cuda.amp, "GradScaler", return_value=None):
                    result = trainer.fit(model, loaders)
                self.assertEqual(result["stage1"]["epochs_ran"], 40)
                self.assertEqual(result["selection"]["best_epoch"], 5)
                self.assertEqual(result["selection"]["metric"], "MAE")
                self.assertEqual(result["selection"]["direction"], "min")
                self.assertEqual(result["selection"]["best_test_mae"], 0.2)
                self.assertEqual(result["test"]["MAE"], 0.2)
                self.assertEqual(result["test"]["Corr"], 0.05)
                self.assertEqual(int(model.epoch), 5)
                checkpoint = torch.load(Path(directory) / "checkpoints/final.pt", map_location="cpu")
                self.assertEqual(checkpoint["epoch"], 5)
                self.assertEqual(int(checkpoint["model"]["epoch"]), 5)
                history = json.loads((Path(directory) / "stage1_history.json").read_text())
                self.assertEqual(history[19]["test"]["F1_score"], 0.99)
                self.assertEqual(len(history), 40)

    def test_nonfinite_mae_cannot_reuse_a_stale_checkpoint(self):
        for dataset, module in self.modules.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                trainer = self.trainer(module, directory, invalid=True)
                with patch.object(module, "get_cosine_schedule_with_warmup", return_value=None), \
                     patch.object(module.torch.cuda.amp, "GradScaler", return_value=None):
                    with self.assertRaisesRegex(RuntimeError, "finite test MAE"):
                        trainer._fit_stage(TinyModel(), {"train": [0], "test": [0]},
                                           "stage1", 2, 41, Path(directory) / "stage1.pt")


if __name__ == "__main__":
    unittest.main()
