"""Post-selection contextual evidence deletion; this does not tune the model."""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import torch

from .data import move_batch, unpack
from .metrics import metrics_from_rows
from .model import MODALITIES
from .utils import preserve_rng, write_json, write_jsonl


def select_positions(joint, mask, ids, modality, kind, fraction=.25, seed=1111):
    removed = torch.zeros_like(mask, dtype=torch.bool)
    for b, sample_id in enumerate(ids):
        indices = mask[b].nonzero().flatten().cpu().numpy()
        count = min(len(indices), max(1, math.ceil(len(indices) * fraction)))
        if not count:
            continue
        scores = joint[b, indices].detach().float().cpu().numpy()
        if kind in ("high", "low"):
            order = np.argsort(-scores if kind == "high" else scores, kind="stable")
        elif kind == "random":
            digest = hashlib.sha256(f"{seed}:{sample_id}:{modality}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            order = rng.permutation(len(indices))
        else:
            raise ValueError(kind)
        removed[b, torch.as_tensor(indices[order[:count]], device=mask.device)] = True
    return removed


@torch.no_grad()
def analyze(trainer, loader):
    trainer.model.set_stage("eval")
    model = trainer.model
    conditions = ["full"] + [f"delete_{m}" for m in MODALITIES] + [f"{m}_{k}" for m in MODALITIES for k in ("high", "low", "random")]
    all_rows = {condition: [] for condition in conditions}
    started = time.monotonic()
    with preserve_rng():
        for step, cpu_batch in enumerate(loader, 1):
            batch = move_batch(cpu_batch, trainer.device)
            labels, text, audio, vision = unpack(batch)
            with trainer.amp():
                evidence = model.encode_evidence(text, audio, vision)
                original_prefix, original_mask = model.final_prefix(evidence["tokens"], text[0], evidence["presence"])
                for condition in conditions:
                    selected = None
                    if condition == "full":
                        tokens = evidence["tokens"]
                    elif condition.startswith("delete_"):
                        tokens = model.evidence.delete_modality(evidence, condition.removeprefix("delete_"))
                    else:
                        modality, kind = condition.split("_")
                        selected = select_positions(evidence["J"][modality], evidence["masks"][modality],
                                                    batch["id"], modality, kind,
                                                    trainer.config["analysis"]["deletion_fraction"],
                                                    trainer.config["analysis"]["random_seed"])
                        tokens = model.evidence.delete_positions(evidence, modality, selected)
                    prefix, mask = model.final_prefix(tokens, text[0], evidence["presence"])
                    if not torch.equal(mask, original_mask) or prefix.shape != original_prefix.shape:
                        raise AssertionError("deletion changed prefix layout")
                    losses = model._teacher_forcing_losses(prefix, mask, labels)
                    values, parsing = model.decode_prefix(prefix, mask)
                    rows = trainer.sample_rows(batch, values, parsing, evidence)
                    for i, row in enumerate(rows):
                        row["teacher_forcing_loss"] = float(losses[i])
                        if selected is not None:
                            row["removed_positions"] = selected[i].nonzero().flatten().cpu().tolist()
                    all_rows[condition].extend(rows)
            if step % 10 == 0:
                from .trainer import LOGGER
                LOGGER.info("deletion analysis %d/%d (%.1fs)", step, len(loader), time.monotonic()-started)
    full = all_rows["full"]
    full_mae = metrics_from_rows(full)["MAE"]
    summary = {}
    for condition, rows in all_rows.items():
        delta = []
        for row, reference in zip(rows, full):
            if row["id"] != reference["id"]:
                raise AssertionError("analysis sample order mismatch")
            row["loss_delta_from_full"] = row["teacher_forcing_loss"] - reference["teacher_forcing_loss"]
            delta.append(row["loss_delta_from_full"])
        metrics = metrics_from_rows(rows)
        metrics.update(mean_loss_delta=float(np.mean(delta)), delta_MAE=metrics["MAE"]-full_mae)
        summary[condition] = metrics
        write_jsonl(trainer.output / "analysis" / f"{condition}.jsonl", rows)
    result = dict(split="validation_only_after_checkpoint_selection", conditions=summary,
                  deletion_fraction=trainer.config["analysis"]["deletion_fraction"],
                  intervention="fixed_attention_gates_context_LN_statistics_read_contribution_deletion",
                  interpretation="auxiliary contextual evidence allocation, not complete modality or raw-frame causal contribution",
                  elapsed_seconds=time.monotonic()-started)
    write_json(trainer.output / "analysis" / "summary.json", result)
    return result
