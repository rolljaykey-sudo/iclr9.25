"""Three-stage U-HERA training with immutable offline reference utility."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup

from . import ARCHITECTURE
from .data import move_batch, sequential_loader, unpack
from .metrics import metrics_from_rows
from .model import MODALITIES, STAGES
from .utils import (accumulation_windows, atomic_save, capture_rng, object_hash, preserve_rng,
                    restore_rng, sha256_file, utility_targets, write_json, write_jsonl)

LOGGER = logging.getLogger("u_hera")


class UHERATrainer:
    def __init__(self, model, config, output_dir, identity):
        self.model = model
        self.config = config
        self.settings = config["training"]
        self.output = Path(output_dir)
        self.identity = identity
        self.device = next(model.parameters()).device
        self.cache = None
        self.targets = None
        self.output.mkdir(parents=True, exist_ok=True)

    def amp(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()

    def checkpoint(self, path, stage, epoch, valid, **extra):
        payload = dict(architecture=ARCHITECTURE, model_settings=asdict(self.model.settings),
                       identity=self.identity, model=self.model.experiment_state_dict(),
                       routing_mode=self.model.routing_mode, stage=stage, epoch=epoch,
                       valid_metrics=valid, rng=capture_rng(), **extra)
        atomic_save(path, payload)
        return payload

    def load(self, path):
        payload = torch.load(path, map_location="cpu")
        if payload.get("identity") != self.identity:
            raise ValueError(f"checkpoint identity mismatch: {path}")
        self.model.load_experiment_state_dict(payload)
        return payload

    def train_epoch(self, loader, optimizer, scheduler, scaler, stage, epoch):
        self.model.train()
        totals = dict(loss=0., generation=0., utility=0., samples=0, updates=0, skipped_updates=0)
        started = time.monotonic()
        for update, window in enumerate(accumulation_windows(loader, self.settings["accumulation_steps"]), 1):
            count = sum(len(b["id"]) for b in window)
            optimizer.zero_grad(set_to_none=True)
            for cpu_batch in window:
                batch = move_batch(cpu_batch, self.device)
                targets = None
                if stage != "evidence_warmup":
                    if self.targets is None:
                        raise ValueError("missing reference cache")
                    targets = torch.stack([self.targets[key] for key in batch["id"]]).to(self.device)
                with self.amp():
                    result = self.model(*unpack(batch), stage=stage, utility_targets=targets,
                                        utility_weight=self.settings["utility_weight"])
                    weighted_loss = result["Loss"] * (len(batch["id"]) / count)
                if not bool(torch.isfinite(result["Loss"])):
                    raise FloatingPointError(f"non-finite {stage} loss")
                scaler.scale(weighted_loss).backward()
                n = len(batch["id"])
                for key, output_key in (("loss", "Loss"), ("generation", "GenerationLoss"), ("utility", "UtilityLoss")):
                    totals[key] += float(result[output_key].detach()) * n
                totals["samples"] += n
                del result, weighted_loss
            scaler.unscale_(optimizer)
            norm = clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.settings["gradient_clip"])
            if self.device.type != "cuda" and not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite CPU gradient")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                totals["updates"] += 1
            else:
                totals["skipped_updates"] += 1
            if any(p.grad is not None for p in self.model.llm.parameters()):
                raise RuntimeError("frozen LLM received parameter gradients")
            if update % self.settings["progress_interval"] == 0:
                LOGGER.info("%s epoch %d update %d loss %.5f gen %.5f utility %.5f elapsed %.1fs",
                            stage, epoch, update, totals["loss"]/totals["samples"],
                            totals["generation"]/totals["samples"], totals["utility"]/totals["samples"],
                            time.monotonic()-started)
        if totals["updates"] == 0:
            raise FloatingPointError("no successful optimizer update in epoch")
        for key in ("loss", "generation", "utility"):
            totals[key] /= totals["samples"]
        totals["elapsed_seconds"] = time.monotonic() - started
        return totals

    @staticmethod
    def evidence_stats(evidence):
        tokens = F.normalize(evidence["tokens"].float(), dim=-1)
        cosine = tokens @ tokens.transpose(-1, -2)
        k = tokens.shape[1]
        off_diagonal = ~torch.eye(k, dtype=torch.bool, device=tokens.device)
        similarity = cosine[:, off_diagonal].mean(-1) if k > 1 else tokens.new_zeros(tokens.shape[0])
        overlaps = {}
        for m in MODALITIES:
            normalized = F.normalize(evidence["alpha"][m].float(), dim=-1)
            pair = normalized @ normalized.transpose(-1, -2)
            overlaps[m] = pair[:, off_diagonal].mean(-1) if k > 1 else similarity * 0
        return similarity, overlaps

    def sample_rows(self, batch, values, parsing, evidence, detailed=False):
        similarity, overlaps = self.evidence_stats(evidence)
        rows = []
        for i, sample_id in enumerate(batch["id"]):
            row = dict(id=sample_id, label=float(batch["labels"]["M"][i]), prediction=values[i],
                       raw_response=parsing["raw_responses"][i], invalid=i in parsing["invalid_indices"],
                       out_of_range=i in parsing["out_of_range_indices"], G=evidence["G"][i].detach().cpu().tolist(),
                       gates=evidence["gates"][i].detach().cpu().tolist(), slot_cosine=float(similarity[i]),
                       alpha_overlap={m: float(overlaps[m][i]) for m in MODALITIES})
            if detailed:
                row["raw_text"] = batch["raw_text"][i]
                row["text_token_ids"] = batch["text"][i, 0, batch["text"][i, 1].bool()].long().cpu().tolist()
                for m in MODALITIES:
                    valid = evidence["masks"][m][i]
                    row[m] = dict(positions=valid.nonzero().flatten().cpu().tolist(),
                                  alpha=evidence["alpha"][m][i, :, valid].cpu().tolist(),
                                  J=evidence["J"][m][i, valid].cpu().tolist())
            rows.append(row)
        return rows

    @torch.no_grad()
    def evaluate(self, loader, name, detailed=False, reuse=False):
        prediction_path = self.output / "predictions" / f"{name}.jsonl"
        metrics_path = self.output / "predictions" / f"{name}.metrics.json"
        model_hash = object_hash(dict(identity=self.identity, routing=self.model.routing_mode,
                                      checkpoint_state={k: sha_tensor(v) for k, v in self.model.experiment_state_dict().items()}))
        if reuse and prediction_path.exists():
            rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
            if any(r.get("evaluation_model_hash") != model_hash for r in rows):
                raise ValueError("existing predictions belong to a different checkpoint")
            if [r["id"] for r in rows] != [loader.dataset[i]["id"] for i in range(len(loader.dataset))]:
                raise ValueError("existing prediction sample IDs mismatch")
            metrics = metrics_from_rows(rows)
            write_json(metrics_path, dict(metrics=metrics, model_hash=model_hash, reused=True))
            return metrics
        self.model.eval()
        rows = []
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.monotonic()
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, self.device)
            _, text, audio, vision = unpack(batch)
            with self.amp():
                values, diagnostics = self.model.generate(text, audio, vision, return_diagnostics=True)
            rows.extend(self.sample_rows(batch, values, diagnostics, diagnostics["evidence"], detailed))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        for row in rows:
            row["evaluation_model_hash"] = model_hash
        write_jsonl(prediction_path, rows)
        metrics = metrics_from_rows(rows)
        metrics.update(elapsed_seconds=elapsed, samples_per_second=len(rows)/max(elapsed, 1e-9),
                       mean_slot_cosine=sum(r["slot_cosine"] for r in rows)/len(rows),
                       mean_alpha_overlap={m: sum(r["alpha_overlap"][m] for r in rows)/len(rows) for m in MODALITIES})
        write_json(metrics_path, dict(metrics=metrics, model_hash=model_hash))
        LOGGER.info("%s: MAE %.6f Corr %s invalid %.4f G %s (%.1fs)", name, metrics["MAE"],
                    metrics["Corr"], metrics["invalid_generation_rate"], metrics["mean_G_T_A_V_Null"], elapsed)
        return metrics

    def fit_stage(self, loaders, stage, resume=False):
        directory = self.output / "checkpoints" / stage
        best_path, last_path, done_path = (directory / x for x in ("best.pt", "last.pt", "done.pt"))
        if done_path.exists():
            if not resume:
                raise FileExistsError(done_path)
            done = self.load(done_path)
            restore_rng(done["rng"])
            return done["summary"]
        spec = self.settings["stages"][stage]
        self.model.set_stage(stage)
        optimizer = torch.optim.AdamW(self.model.trainable_parameter_groups(self.settings["learning_rate"]),
                                      eps=self.settings["adam_epsilon"], weight_decay=self.settings["weight_decay"])
        updates = math.ceil(len(loaders["train"]) / self.settings["accumulation_steps"])
        total = updates * spec["max_epochs"]
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(round(total*self.settings["warmup_fraction"])), total)
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda", init_scale=1024.)
        start_epoch, best_mae, best_epoch, history = 1, float("inf"), 0, []
        if last_path.exists() and resume:
            previous = self.load(last_path)
            if stage != "evidence_warmup" and previous.get("utility_cache_sha256") != self.cache_hash():
                raise ValueError("resumed training utility cache changed")
            optimizer.load_state_dict(previous["optimizer"])
            scheduler.load_state_dict(previous["scheduler"])
            scaler.load_state_dict(previous["scaler"])
            restore_rng(previous["rng"])
            start_epoch = previous["epoch"] + 1
            best_mae, best_epoch, history = previous["best_mae"], previous["best_epoch"], previous["history"]
        elif stage == "joint":
            # Joint starts from an already utility-trained Router. Retain it as epoch 0.
            baseline = self.evaluate(loaders["valid"], "joint_epoch_0000_valid")
            best_mae = baseline["MAE"]
            self.checkpoint(best_path, stage, 0, baseline)
        LOGGER.info("stage %s trainable parameters %d", stage, sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        for epoch in range(start_epoch, spec["max_epochs"] + 1):
            # An interrupted process may have written last.pt at an early-stop boundary.
            if history and history[-1]["epoch"] - best_epoch >= spec["patience"]:
                break
            train = self.train_epoch(loaders["train"], optimizer, scheduler, scaler, stage, epoch)
            valid = self.evaluate(loaders["valid"], f"{stage}_epoch_{epoch:04d}_valid")
            record = dict(epoch=epoch, train=train, valid=valid)
            history.append(record)
            if valid["MAE"] < best_mae - 1e-6:
                best_mae, best_epoch = valid["MAE"], epoch
                self.checkpoint(best_path, stage, epoch, valid)
            self.checkpoint(last_path, stage, epoch, valid, optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), history=history,
                            best_epoch=best_epoch, best_mae=best_mae,
                            utility_cache_sha256=self.cache_hash() if stage != "evidence_warmup" else None)
            write_json(self.output / f"{stage}_history.json", history)
            LOGGER.info("%s epoch %d/%d gen %.5f utility %.5f valid MAE %.6f best %.6f epoch %d",
                        stage, epoch, spec["max_epochs"], train["generation"], train["utility"], valid["MAE"], best_mae, best_epoch)
            if epoch - best_epoch >= spec["patience"]:
                break
        if not best_path.exists():
            raise RuntimeError("stage produced no selectable checkpoint")
        best = self.load(best_path)
        summary = dict(stage=stage, best_epoch=best_epoch, best_valid_mae=best_mae,
                       valid_metrics=best["valid_metrics"], epochs_ran=len(history), checkpoint=str(best_path))
        self.checkpoint(done_path, stage, best_epoch, best["valid_metrics"], summary=summary)
        write_json(self.output / f"{stage}_result.json", summary)
        return summary

    def cache_hash(self):
        return sha256_file(self.output / "utility_cache.pt") if self.cache is not None else None

    @torch.no_grad()
    def cache_reference(self, train_loader, reference_path):
        path = self.output / "utility_cache.pt"
        expected_ids = [train_loader.dataset[i]["id"] for i in range(len(train_loader.dataset))]
        manifest = dict(identity=self.identity, reference_sha256=sha256_file(reference_path),
                        training_ids_hash=object_hash(expected_ids), split="official_train_only",
                        intervention="subtract_direct_injection_fixed_candidates_gates_positions_raw_text",
                        target_loss="per_sample_mean_causal_label_token_CE")
        if path.exists():
            cache = torch.load(path, map_location="cpu")
            if cache["manifest"] != manifest or cache["ids"] != expected_ids:
                raise ValueError("reference utility cache identity mismatch")
        else:
            self.load(reference_path)
            self.model.set_stage("eval")
            self.model.routing_mode = "uniform"
            self.model.zero_grad(set_to_none=True)
            ids, full_losses, deleted_losses, presences = [], [], [], []
            started = time.monotonic()
            with preserve_rng():
                for step, cpu_batch in enumerate(sequential_loader(train_loader), 1):
                    batch = move_batch(cpu_batch, self.device)
                    labels, text, audio, vision = unpack(batch)
                    with self.amp():
                        evidence = self.model.encode_evidence(text, audio, vision)
                        prefix, mask = self.model.final_prefix(evidence["tokens"], text[0], evidence["presence"])
                        full = self.model._teacher_forcing_losses(prefix, mask, labels)
                        deleted = []
                        for m in MODALITIES:
                            tokens = self.model.evidence.delete_modality(evidence, m)
                            prefix_minus, mask_minus = self.model.final_prefix(tokens, text[0], evidence["presence"])
                            if not torch.equal(mask, mask_minus):
                                raise AssertionError("counterfactual changed positions/mask")
                            deleted.append(self.model._teacher_forcing_losses(prefix_minus, mask_minus, labels))
                    ids.extend(batch["id"])
                    full_losses.append(full.cpu())
                    deleted_losses.append(torch.stack(deleted, -1).cpu())
                    presences.append(evidence["presence"].cpu())
                    if step % self.settings["progress_interval"] == 0:
                        LOGGER.info("utility cache %d/%d (%.1fs)", step, len(train_loader), time.monotonic()-started)
            full, deleted, presence = torch.cat(full_losses), torch.cat(deleted_losses), torch.cat(presences)
            if ids != expected_ids or len(set(ids)) != len(ids):
                raise ValueError("reference traversal did not cover exactly the training samples")
            deltas = deleted - full[:, None]
            targets, temperature = utility_targets(deltas, presence, self.settings["utility_temperature_floor"])
            cache = dict(manifest=manifest, ids=ids, full_loss=full, deleted_loss=deleted, delta=deltas,
                         presence=presence, targets=targets, temperature=temperature,
                         elapsed_seconds=time.monotonic()-started)
            atomic_save(path, cache)
        # Recompute target semantics even for an existing cache, not just its filename.
        expected, scale = utility_targets(cache["delta"], cache["presence"], self.settings["utility_temperature_floor"])
        if not torch.equal(expected, cache["targets"]) or scale != cache["temperature"]:
            raise ValueError("corrupt utility target cache")
        self.cache = cache
        self.targets = dict(zip(cache["ids"], cache["targets"]))
        report = dict(manifest=manifest, samples=len(cache["ids"]), temperature=cache["temperature"],
                      mean_delta=cache["delta"].mean(0), positive_fraction=(cache["delta"] > 0).float().mean(0),
                      mean_target=cache["targets"].mean(0), elapsed_seconds=cache["elapsed_seconds"],
                      target_entropy=-(cache["targets"] * cache["targets"].clamp_min(1e-8).log()).sum(-1).mean(),
                      sha256=sha256_file(path))
        write_json(self.output / "utility_cache.json", report)
        LOGGER.info("utility cache ready: %d samples temperature %.6g", len(cache["ids"]), cache["temperature"])
        return report

    def fit(self, loaders, resume=False, run_analysis=True):
        started = time.monotonic()
        evidence_summary = self.fit_stage(loaders, "evidence_warmup", resume)
        reference_path = self.output / "checkpoints" / "reference.pt"
        if not reference_path.exists():
            self.checkpoint(reference_path, "evidence_warmup", evidence_summary["best_epoch"], evidence_summary["valid_metrics"])
        else:
            self.load(reference_path)
        cache_report = self.cache_reference(loaders["train"], reference_path)
        router_summary = self.fit_stage(loaders, "router_warmup", resume)
        joint_summary = self.fit_stage(loaders, "joint", resume)
        chosen = router_summary
        if joint_summary["best_valid_mae"] < router_summary["best_valid_mae"] - 1e-6:
            chosen = joint_summary
        self.load(chosen["checkpoint"])
        self.model.set_stage("eval")
        final_path = self.output / "checkpoints" / "final.pt"
        self.checkpoint(final_path, chosen["stage"], chosen["best_epoch"], chosen["valid_metrics"],
                        utility_cache_sha256=self.cache_hash(), selection="minimum_learned_stage_validation_MAE")
        write_json(self.output / "selection.json", chosen)
        valid = self.evaluate(loaders["valid"], "final_valid", detailed=True, reuse=True)
        test = self.evaluate(loaders["test"], "test", detailed=True, reuse=True)
        analysis = None
        if run_analysis:
            from .analysis import analyze
            analysis = analyze(self, loaders["valid"])
        result = dict(status="complete", architecture=ARCHITECTURE, seed=self.config["seed"],
                      identity=self.identity, reference=evidence_summary, utility_cache=cache_report,
                      router_warmup=router_summary, joint=joint_summary, selection=chosen,
                      valid=valid, test=test, analysis=analysis, final_checkpoint=str(final_path),
                      training_settings=self.settings, elapsed_seconds_this_process=time.monotonic()-started)
        write_json(self.output / "result.json", result)
        return result


def sha_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
