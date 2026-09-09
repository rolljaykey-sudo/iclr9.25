from __future__ import annotations

import copy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from u_hera import ARCHITECTURE
from u_hera.analysis import select_positions
from u_hera.data import make_loaders
from u_hera.metrics import sentiment_metrics
from u_hera.model import MODALITIES, ModelSettings, UHERAModel
from u_hera.sequence import compact_left_padding, masked_softmax, per_sample_causal_loss
from u_hera.trainer import UHERATrainer
from u_hera.utils import accumulation_windows, capture_rng, restore_rng, utility_targets


class TinyTokenizer:
    eos_token_id, pad_token_id, bos_token_id = 0, 0, 1

    def encode(self, text, **kwargs):
        return [3 + ord(c) % 57 for c in text]

    def __call__(self, strings, **kwargs):
        rows = [self.encode(s) for s in strings]
        width = kwargs.get("max_length", max(map(len, rows)))
        rows = [r[:width] for r in rows]
        return dict(input_ids=torch.tensor([[0]*(width-len(r))+r for r in rows]),
                    attention_mask=torch.tensor([[0]*(width-len(r))+[1]*len(r) for r in rows]))

    def batch_decode(self, tokens, **kwargs):
        # Deterministic parsing fixture; the actual tiny Qwen still runs every forward.
        return ["+0.0" for _ in tokens]


def tiny_model():
    package_name = "u_hera_test_qwen"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        config_path = Path(__file__).resolve().parents[1] / "configs/qwen_mosi.json"
        package.__path__ = [json.loads(config_path.read_text())["model_path"]]
        sys.modules[package_name] = package
    qwen = importlib.import_module(package_name + ".modeling_qwen")
    config = qwen.QWenConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                            kv_channels=8, intermediate_size=64, seq_length=512, max_position_embeddings=512,
                            use_dynamic_ntk=False, use_logn_attn=False, use_flash_attn=False, fp32=True)
    settings = ModelSettings(dim=16, slots=8, heads=4, ffn_dim=32, dropout=0., audio_hidden=4, vision_hidden=4)
    return UHERAModel(qwen.QWenLMHeadModel(config), TinyTokenizer(), settings, "sentiment:")


def inputs():
    text = torch.tensor([[[0, 11, 12, 13, 14], [0, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
                         [[0, 0, 13, 12, 11], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]]])
    audio = (torch.randn(2, 6, 5), torch.tensor([5, 2]))
    vision = (torch.randn(2, 7, 20), torch.tensor([6, 3]))
    return (text, text[:, 1].sum(-1)), audio, vision


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = tiny_model().eval()
        self.text, self.audio, self.vision = inputs()

    def test_shapes_conservation_and_batch_one(self):
        for b in (1, 2):
            data = [tuple(t[:b] for t in pair) for pair in (self.text, self.audio, self.vision)]
            e = self.model.encode_evidence(*data)
            self.assertEqual(e["tokens"].shape, (b, 8, 32))
            self.assertEqual(e["gates"].shape, (b, 8, 4))
            torch.testing.assert_close(e["gates"].sum(-1), torch.ones(b, 8))
            for i, m in enumerate(MODALITIES):
                torch.testing.assert_close(e["alpha"][m].sum(-1), torch.ones(b, 8))
                torch.testing.assert_close(e["J"][m].sum(-1), e["G"][:, i])
                self.assertTrue(torch.equal(e["alpha"][m].masked_select(~e["masks"][m][:, None]), torch.zeros_like(e["alpha"][m].masked_select(~e["masks"][m][:, None]))))

    def test_padding_invariance_including_nonzero_tails(self):
        original = self.model.encode_evidence(self.text, self.audio, self.vision)
        modified = []
        for sequence, lengths in (self.audio, self.vision):
            x = sequence.clone()
            mask = torch.arange(x.shape[1])[None] >= lengths[:, None]
            x[mask] = 12345.
            modified.append((x, lengths))
        e = self.model.encode_evidence(self.text, *modified)
        torch.testing.assert_close(original["tokens"], e["tokens"])
        padded = F.pad(self.text[0], (3, 0))
        e2 = self.model.encode_evidence((padded, self.text[1]), self.audio, self.vision)
        torch.testing.assert_close(original["tokens"], e2["tokens"], atol=1e-6, rtol=1e-5)
        p1, m1 = self.model.final_prefix(original["tokens"], self.text[0], original["presence"])
        p2, m2 = self.model.final_prefix(e2["tokens"], padded, e2["presence"])
        torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.equal(m1, m2))

    def test_text_order_changes_evidence(self):
        original = self.model.encode_evidence(self.text, self.audio, self.vision)
        tensor = self.text[0].clone()
        tensor[0, 0, 1:] = tensor[0, 0, 1:].flip(0)
        shuffled = self.model.encode_evidence((tensor, self.text[1]), self.audio, self.vision)
        self.assertFalse(torch.allclose(original["tokens"][0], shuffled["tokens"][0], atol=1e-7, rtol=1e-6))

    def test_all_presence_patterns_and_null_identity(self):
        self.model.set_stage("joint")
        self.model.eval()
        for pattern in range(8):
            p = torch.tensor([[bool(pattern & (1 << i)) for i in range(3)]]).expand(2, -1).float()
            e = self.model.encode_evidence(self.text, self.audio, self.vision, p)
            self.assertTrue(torch.isfinite(e["tokens"]).all())
            torch.testing.assert_close(e["gates"].sum(-1), torch.ones(2, 8))
            for i, m in enumerate(MODALITIES):
                if not p[0, i]:
                    self.assertEqual(float(e["gates"][..., i].sum()), 0.)
                    self.assertEqual(float(e["alpha"][m].sum()), 0.)
            if pattern == 0:
                self.assertTrue(torch.equal(e["tokens"], self.model.evidence.composer.base_prefix[None].expand_as(e["tokens"])))

    def test_zero_length_sequences(self):
        e = self.model.encode_evidence((self.text[0][:, :, :0], torch.zeros(2).long()),
                                      (self.audio[0][:, :0], torch.zeros(2).long()),
                                      (self.vision[0][:, :0], torch.zeros(2).long()))
        self.assertTrue(torch.isfinite(e["tokens"]).all())
        torch.testing.assert_close(e["G"][:, 3], torch.ones(2))

    def test_counterfactual_composition_and_raw_text_fixed(self):
        e = self.model.encode_evidence(self.text, self.audio, self.vision)
        prefix, mask = self.model.final_prefix(e["tokens"], self.text[0], e["presence"])
        for modality in MODALITIES:
            deleted = self.model.evidence.delete_modality(e, modality)
            all_positions = self.model.evidence.delete_positions(e, modality, e["masks"][modality])
            torch.testing.assert_close(deleted, all_positions, atol=1e-7, rtol=1e-5)
            unchanged = self.model.evidence.delete_positions(e, modality, torch.zeros_like(e["masks"][modality]))
            torch.testing.assert_close(unchanged, e["tokens"])
            p, m = self.model.final_prefix(deleted, self.text[0], e["presence"])
            self.assertTrue(torch.equal(mask, m))
            self.assertTrue((((p-prefix).abs().sum(-1) > 1e-8).sum(-1) <= 8).all())
        empty = self.model.evidence.delete_modality(e, MODALITIES)
        torch.testing.assert_close(empty, self.model.evidence.composer.base_prefix[None].expand_as(empty), atol=1e-7, rtol=1e-5)

    def test_stage_optimizer_freezing_and_gradients(self):
        llm_before = {n: p.clone() for n, p in self.model.llm.named_parameters()}
        for stage in ("evidence_warmup", "router_warmup", "joint"):
            self.model.set_stage(stage)
            expected = {n: p.detach().clone() for n, p in self.model.named_parameters() if not n.startswith("llm.")}
            optimizer = torch.optim.AdamW(self.model.trainable_parameter_groups(.0005), eps=1e-4)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = self.model(torch.tensor([.2, -1.3]), self.text, self.audio, self.vision,
                                    utility_targets=torch.tensor([[.1, .6, .2, .1], [.2, .1, .6, .1]]))
                result["Loss"].backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.parameters() if p.requires_grad))
                optimizer.step()
            for name, p in self.model.named_parameters():
                if name.startswith("llm."):
                    self.assertFalse(p.requires_grad)
                    self.assertIsNone(p.grad)
                elif not p.requires_grad:
                    self.assertTrue(torch.equal(p, expected[name]), name)
            self.assertFalse(self.model.llm.training)
            if stage == "router_warmup":
                self.assertFalse(self.model.evidence.relations["text"].training)
                self.assertTrue(self.model.evidence.slot_router.training)
        for name, p in self.model.llm.named_parameters():
            self.assertTrue(torch.equal(p, llm_before[name]))

    def test_checkpoint_roundtrip_and_old_format_rejection(self):
        payload = dict(architecture=ARCHITECTURE, model_settings=asdict(self.model.settings),
                       routing_mode=self.model.routing_mode, model=self.model.experiment_state_dict())
        second = tiny_model().eval()
        # Only external weights are copied: frozen backbone identity is checked by the trainer.
        second.llm.load_state_dict(self.model.llm.state_dict())
        second.load_experiment_state_dict(payload)
        torch.testing.assert_close(self.model.encode_evidence(self.text, self.audio, self.vision)["tokens"],
                                   second.encode_evidence(self.text, self.audio, self.vision)["tokens"])
        self.assertFalse(any(n.startswith("llm.") for n in payload["model"]))
        with self.assertRaises(ValueError):
            second.load_experiment_state_dict({"architecture": "v4"})


class MathTests(unittest.TestCase):
    def test_loss_manual_sample_normalization_and_shift(self):
        torch.manual_seed(9)
        logits = torch.randn(2, 6, 7, requires_grad=True)
        targets = torch.tensor([[-100, -100, 2, 3, 4, 5], [-100, -100, -100, -100, 2, -100]])
        loss = per_sample_causal_loss(logits, targets)
        logp = logits.log_softmax(-1)
        expected = torch.stack((-(logp[0, 1, 2]+logp[0, 2, 3]+logp[0, 3, 4]+logp[0, 4, 5])/4, -logp[1, 3, 2]))
        torch.testing.assert_close(loss, expected)
        loss.mean().backward()
        self.assertEqual(float(logits.grad[:, -1].abs().sum()), 0.)
        with self.assertRaises(ValueError):
            per_sample_causal_loss(logits, torch.full_like(targets, -100))

    def test_compaction_targets_order_and_gradients(self):
        embeds = torch.arange(24.).reshape(2, 6, 2).requires_grad_()
        mask = torch.tensor([[1, 0, 1, 0, 1, 1], [1, 0, 0, 0, 1, 1]])
        labels = torch.arange(12).reshape(2, 6)
        packed, attention, targets = compact_left_padding(embeds, mask, labels)
        self.assertEqual(targets.tolist(), [[0, 2, 4, 5], [-100, 6, 10, 11]])
        packed.sum().backward()
        self.assertEqual(float(embeds.grad.masked_select(~mask.bool()[..., None]).sum()), 0.)

    def test_utility_sign_null_missing_and_finite_softmax(self):
        delta = torch.tensor([[.3, -.2, .1], [-.3, -.2, -.1], [0., 0., 0.]])
        presence = torch.tensor([[1., 1., 1.], [1., 1., 1.], [0., 0., 0.]])
        targets, tau = utility_targets(delta, presence)
        self.assertEqual(targets[0].argmax().item(), 0)
        self.assertEqual(targets[1].argmax().item(), 3)
        self.assertEqual(targets[2].tolist(), [0., 0., 0., 1.])
        self.assertAlmostEqual(tau, float(delta[:2].std(unbiased=False)))
        empty = masked_softmax(torch.tensor([[float("inf"), float("inf")]]), torch.zeros(1, 2).bool())
        self.assertEqual(empty.tolist(), [[0., 0.]])

    def test_tail_accumulation_equals_full_window_gradient(self):
        samples = torch.arange(7.).reshape(-1, 1)
        batches = [dict(id=list(range(i, min(i+3, 7))), x=samples[i:i+3]) for i in range(0, 7, 3)]
        for window in accumulation_windows(batches, 2):
            p = torch.tensor(2., requires_grad=True)
            for batch in window:
                (p.mul(batch["x"]).square().mean() * len(batch["id"]) / sum(len(b["id"]) for b in window)).backward()
            expected = torch.cat([b["x"] for b in window]).square().mean() * 4
            torch.testing.assert_close(p.grad, expected)

    def test_deletion_rank_ties_budget_and_id_reproducibility(self):
        scores = torch.ones(2, 9)
        mask = torch.tensor([[1]*8+[0], [1]*4+[0]*5]).bool()
        high = select_positions(scores, mask, ["a", "b"], "audio", "high")
        self.assertEqual(high[0].nonzero().flatten().tolist(), [0, 1])
        self.assertEqual(high[1].nonzero().flatten().tolist(), [0])
        random = select_positions(scores, mask, ["a", "b"], "audio", "random")
        reversed_rows = select_positions(scores.flip(0), mask.flip(0), ["b", "a"], "audio", "random")
        self.assertTrue(torch.equal(random, reversed_rows.flip(0)))
        self.assertTrue(torch.equal(random.sum(-1), high.sum(-1)))

    def test_mosi_neutral_conventions(self):
        metrics = sentiment_metrics([0., -.1, .1], [-.1, 0., .1])
        self.assertAlmostEqual(metrics["Has0_acc_2"], 1/3)
        self.assertAlmostEqual(metrics["Non0_acc_2"], 1.)


class ToyDataset(Dataset):
    def __init__(self, prefix, n):
        self.prefix, self.n = prefix, n
        self.examples = inputs()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        text, audio, vision = self.examples
        i = index % 2
        return dict(id=f"{self.prefix}_{index}", raw_text="toy", text=text[0][i], text_lengths=text[1][i],
                    audio=audio[0][i], audio_lengths=audio[1][i], vision=vision[0][i], vision_lengths=vision[1][i],
                    labels={"M": torch.tensor(.4 if i == 0 else -1.2)})


class TrainerTests(unittest.TestCase):
    def config(self):
        config = json.loads((Path(__file__).resolve().parents[1]/"configs/qwen_mosi.json").read_text())
        config["training"].update(microbatch=2, accumulation_steps=2, progress_interval=999)
        for stage in config["training"]["stages"]:
            config["training"]["stages"][stage] = dict(max_epochs=1, patience=1)
        return config

    def test_full_stages_cache_ids_resume_and_single_test_evaluation(self):
        torch.manual_seed(33)
        model = tiny_model()
        config = self.config()
        datasets = {k: ToyDataset(k, 5 if k == "train" else 2) for k in ("train", "valid", "test")}
        loaders = make_loaders(datasets, config["training"])
        with tempfile.TemporaryDirectory() as directory:
            trainer = UHERATrainer(model, config, directory, "identity")
            result = trainer.fit(loaders, run_analysis=False)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(trainer.cache["ids"], [f"train_{i}" for i in range(5)])
            self.assertEqual(trainer.cache["targets"].shape, (5, 4))
            self.assertIn(result["selection"]["stage"], ("router_warmup", "joint"))
            final = Path(directory)/"checkpoints/final.pt"
            trainer.load(final)
            model.set_stage("eval")
            with patch.object(model, "generate", side_effect=AssertionError("test was rerun")):
                trainer.evaluate(loaders["test"], "test", detailed=True, reuse=True)
            with patch.object(trainer, "train_epoch", side_effect=AssertionError("completed stage was retrained")):
                trainer.fit_stage(loaders, "joint", resume=True)
            payload = torch.load(final)
            payload["identity"] = "different"
            torch.save(payload, Path(directory)/"bad.pt")
            with self.assertRaises(ValueError):
                trainer.load(Path(directory)/"bad.pt")

    def test_reference_cache_preserves_rng_and_rejects_wrong_ids(self):
        config = self.config()
        model = tiny_model()
        loader = make_loaders({"train": ToyDataset("train", 3)}, config["training"])["train"]
        with tempfile.TemporaryDirectory() as directory:
            trainer = UHERATrainer(model, config, directory, "identity")
            reference = Path(directory)/"reference.pt"
            trainer.checkpoint(reference, "evidence_warmup", 0, {})
            before = capture_rng()
            trainer.cache_reference(loader, reference)
            torch.testing.assert_close(torch.get_rng_state(), before["torch"])
            wrong = make_loaders({"train": ToyDataset("other", 3)}, config["training"])["train"]
            with self.assertRaises(ValueError):
                trainer.cache_reference(wrong, reference)

    def test_interrupted_epoch_boundary_resume_matches_uninterrupted(self):
        config = self.config()
        config["training"]["stages"]["evidence_warmup"] = dict(max_epochs=2, patience=5)
        datasets = {k: ToyDataset(k, 5 if k == "train" else 2) for k in ("train", "valid")}
        first = tiny_model()
        initial = copy.deepcopy(first.state_dict())
        loaders = make_loaders(datasets, config["training"])
        with tempfile.TemporaryDirectory() as full_dir, tempfile.TemporaryDirectory() as resume_dir:
            torch.manual_seed(987)
            full = UHERATrainer(first, config, full_dir, "same")
            full.fit_stage(loaders, "evidence_warmup")
            final_full = torch.load(Path(full_dir)/"checkpoints/evidence_warmup/last.pt")
            second = tiny_model()
            second.load_state_dict(initial)
            interrupted = UHERATrainer(second, config, resume_dir, "same")
            original = interrupted.train_epoch
            def stop_at_second_epoch(*args):
                if args[-1] == 2:
                    raise RuntimeError("simulated interruption")
                return original(*args)
            torch.manual_seed(987)
            with patch.object(interrupted, "train_epoch", side_effect=stop_at_second_epoch):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    interrupted.fit_stage(loaders, "evidence_warmup")
            # Disturb all RNG/model values to ensure resume restores them.
            torch.manual_seed(123)
            third = tiny_model()
            third.llm.load_state_dict(first.llm.state_dict())
            resumed = UHERATrainer(third, config, resume_dir, "same")
            resumed.fit_stage(loaders, "evidence_warmup", resume=True)
            final_resumed = torch.load(Path(resume_dir)/"checkpoints/evidence_warmup/last.pt")
            for key, tensor in final_full["model"].items():
                torch.testing.assert_close(tensor, final_resumed["model"][key], atol=0, rtol=0)
            self.assertEqual(final_full["scheduler"], final_resumed["scheduler"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
