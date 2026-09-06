from __future__ import annotations

import unittest

import numpy as np
import torch

from mse_router.data import augment_modalities, make_calibration_split
from mse_router.math_utils import (
    masked_softmax,
    normalized_entropy,
    normalized_js_divergence,
    soft_ordinal_targets,
)
from mse_router.model import (
    MultiScaleProjector,
    PackedLSTMEncoder,
    QwenMseRouter,
    Router,
)


class FakeDataset:
    def __init__(self) -> None:
        ids = []
        labels = []
        for group_index in range(70):
            label = (group_index % 7) - 3
            for clip in range(2):
                ids.append(f"video_{group_index}$_${clip}")
                labels.append([float(label)])
        self.ids = np.asarray(ids)
        self.labels = {"M": np.asarray(labels, dtype=np.float32)}

    def __len__(self) -> int:
        return len(self.ids)


class RouterMathTests(unittest.TestCase):
    def test_soft_ordinal_targets_interpolate_and_clip(self) -> None:
        labels = torch.tensor([-4.0, -2.5, 0.25, 3.0, 4.0])
        targets = soft_ordinal_targets(labels)
        self.assertTrue(torch.allclose(targets.sum(dim=-1), torch.ones(5)))
        self.assertEqual(targets[0].argmax().item(), 0)
        self.assertTrue(torch.allclose(targets[1, :2], torch.tensor([0.5, 0.5])))
        self.assertTrue(torch.allclose(targets[2, 3:5], torch.tensor([0.75, 0.25])))
        self.assertEqual(targets[-1].argmax().item(), 6)

    def test_entropy_and_js_are_normalized(self) -> None:
        uniform = torch.full((1, 7), 1.0 / 7.0)
        concentrated = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0]])
        opposite = torch.tensor([[0.0, 1, 0, 0, 0, 0, 0]])
        self.assertAlmostEqual(normalized_entropy(uniform).item(), 1.0, places=6)
        self.assertLess(normalized_entropy(concentrated).item(), 1e-5)
        self.assertLess(
            normalized_js_divergence(uniform, uniform).abs().item(), 1e-6
        )
        self.assertAlmostEqual(
            normalized_js_divergence(concentrated, opposite).item(), 1.0, places=5
        )

    def test_masked_softmax_excludes_missing_modalities(self) -> None:
        logits = torch.zeros(2, 3)
        mask = torch.tensor([[1, 1, 1], [1, 0, 1]])
        weights = masked_softmax(logits, mask)
        self.assertTrue(torch.allclose(weights[0], torch.full((3,), 1.0 / 3.0)))
        self.assertTrue(torch.allclose(weights[1], torch.tensor([0.5, 0.0, 0.5])))

    def test_zero_initialized_router_starts_uniform(self) -> None:
        router = Router()
        weights = router(torch.randn(3, 30), torch.ones(3, 3))
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0)))


class RouterComponentTests(unittest.TestCase):
    def test_projector_and_lstm_preserve_batch_one(self) -> None:
        encoder = PackedLSTMEncoder(5, 4)
        encoded = encoder(torch.randn(1, 6, 5), torch.tensor([4]))
        self.assertEqual(tuple(encoded.shape), (1, 256))
        projector = MultiScaleProjector(256, 32, 4)
        pseudo = projector(encoded)
        self.assertEqual(tuple(pseudo.shape), (1, 4, 32))

    def test_response_parser_rejects_invalid_and_out_of_range(self) -> None:
        values, diagnostics = QwenMseRouter.parse_responses(
            ["+1.2", "sentiment=-0.7", "unknown", "+4.0"]
        )
        self.assertEqual(values, [1.2, -0.7, 0.0, 0.0])
        self.assertEqual(diagnostics["invalid_indices"], [2])
        self.assertEqual(diagnostics["out_of_range_indices"], [3])

    def test_gate_preserves_uniform_scale(self) -> None:
        instance = object.__new__(QwenMseRouter)
        torch.nn.Module.__init__(instance)
        instance.pseudo_tokens = 4
        instance.hidden_size = 8
        pseudo = torch.randn(2, 3, 4, 8)
        presence = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32)
        pseudo[1, 1] = 0
        weights = torch.tensor(
            [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.0, 0.5]], dtype=torch.float32
        )
        gated = QwenMseRouter.gated_pseudo_tokens(
            instance, pseudo, weights, presence
        ).reshape_as(pseudo)
        self.assertTrue(torch.allclose(gated, pseudo))

    def test_v2_gate_keeps_uniform_audio_visual_scale_and_masks_missing(self) -> None:
        instance = object.__new__(QwenMseRouter)
        torch.nn.Module.__init__(instance)
        instance.pseudo_tokens = 4
        instance.hidden_size = 8
        pseudo = torch.randn(2, 3, 4, 8)
        presence = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32)
        pseudo[1, 1] = 0
        weights = torch.tensor(
            [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.0, 0.5]], dtype=torch.float32
        )
        gated, mask = QwenMseRouter.gated_non_text_pseudo_tokens(
            instance, pseudo, weights, presence
        )
        self.assertTrue(torch.allclose(gated.reshape(2, 2, 4, 8), pseudo[:, 1:]))
        self.assertTrue(torch.equal(mask[0], torch.ones(8, dtype=torch.long)))
        self.assertTrue(
            torch.equal(mask[1], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
        )

    def test_router_has_30_features_and_detaches_diagnostic_inputs(self) -> None:
        instance = object.__new__(QwenMseRouter)
        torch.nn.Module.__init__(instance)
        instance.log_temperatures = torch.nn.Parameter(
            torch.zeros(3), requires_grad=False
        )
        instance.router_variant = "full"
        instance.router = Router()
        torch.nn.init.normal_(instance.router.network[-1].weight)
        logits = torch.randn(2, 3, 7, requires_grad=True)
        presence = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32)
        weights, diagnostics = QwenMseRouter.route(instance, logits, presence)
        self.assertEqual(tuple(diagnostics["router_features"].shape), (2, 30))
        (weights * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
        self.assertIsNone(logits.grad)


class RouterDataTests(unittest.TestCase):
    def test_calibration_split_is_deterministic_and_group_safe(self) -> None:
        dataset = FakeDataset()
        first = make_calibration_split(dataset)
        second = make_calibration_split(dataset)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.calibration_indices, second.calibration_indices)
        train_groups = {
            dataset.ids[index].split("$_$")[0] for index in first.train_indices
        }
        calibration_groups = {
            dataset.ids[index].split("$_$")[0]
            for index in first.calibration_indices
        }
        self.assertFalse(train_groups & calibration_groups)

    def test_augmentation_drops_exactly_one_when_forced(self) -> None:
        batch_size = 32
        batch = {
            "text": torch.zeros(batch_size, 3, 5),
            "audio": torch.ones(batch_size, 6, 2),
            "vision": torch.ones(batch_size, 6, 2),
            "audio_lengths": torch.full((batch_size,), 6),
            "vision_lengths": torch.full((batch_size,), 6),
        }
        _, presence = augment_modalities(
            batch,
            modality_drop_probability=1.0,
            audio_noise_probability=0.0,
            vision_mask_probability=0.0,
        )
        self.assertTrue(torch.equal(presence.sum(dim=-1), torch.full((32,), 2.0)))


if __name__ == "__main__":
    unittest.main()
