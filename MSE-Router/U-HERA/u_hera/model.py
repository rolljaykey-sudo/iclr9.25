"""Sequence evidence extraction, shared slots, and the frozen Qwen boundary."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import ARCHITECTURE
from .sequence import compact_left_padding, masked_softmax, per_sample_causal_loss

MODALITIES = ("text", "audio", "vision")
STAGES = ("evidence_warmup", "router_warmup", "joint")


@dataclass(frozen=True)
class ModelSettings:
    dim: int = 256
    slots: int = 8
    heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    audio_features: int = 5
    vision_features: int = 20
    audio_hidden: int = 64
    vision_hidden: int = 32
    gamma: float = 0.1
    gate_temperature: float = 1.0
    max_new_tokens: int = 4

    def __post_init__(self):
        if self.dim % self.heads or self.dim % 2 or min(self.dim, self.slots, self.heads) < 1:
            raise ValueError("positive slots and even dim divisible by heads are required")
        if self.gate_temperature <= 0 or self.gamma <= 0 or self.max_new_tokens < 1:
            raise ValueError("temperature, gamma and decoding budget must be positive")


class PackedLSTMSequenceEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=256):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(self, sequence, lengths):
        b, length, _ = sequence.shape
        lengths = lengths.reshape(-1).long()
        if lengths.shape != (b,) or bool(((lengths < 0) | (lengths > length)).any()):
            raise ValueError("invalid sequence lengths")
        if length == 0:
            return sequence.new_zeros(b, 0, self.projection.out_features), lengths.new_zeros(b, 0).bool()
        mask = torch.arange(length, device=sequence.device)[None] < lengths[:, None]
        sequence = sequence.masked_fill(~mask[..., None], 0)
        packed = pack_padded_sequence(sequence, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=length)
        return self.projection(out).masked_fill(~mask[..., None], 0), mask


class IntraModalRelationEncoder(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.dim = settings.dim
        self.block = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=settings.heads, dim_feedforward=settings.ffn_dim,
            dropout=settings.dropout, activation="gelu", batch_first=True, norm_first=True,
        )

    def forward(self, features, mask):
        if features.shape[1] == 0:
            return features
        # Logical positions are invariant to how much left padding text has.
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0).float()
        frequency = torch.exp(torch.arange(0, self.dim, 2, device=features.device).float() * (-math.log(10000.) / self.dim))
        angle = positions[..., None] * frequency
        positional = torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)
        x = (features + positional.to(features.dtype)).masked_fill(~mask[..., None], 0)
        # MHA cannot attend to an all-masked row. Expose a zero sentinel internally,
        # then mask the entire absent result, including all affine biases.
        safe = mask.clone()
        safe[:, 0] |= ~mask.any(-1)
        return self.block(x, src_key_padding_mask=~safe).masked_fill(~mask[..., None], 0)


class EvidenceReader(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, query, features, mask):
        values = self.value(features)
        queries, keys = self.query(query), self.key(features)
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = torch.matmul(queries.float(), keys.float().transpose(-1, -2)) * self.scale
            alpha = masked_softmax(scores, mask[:, None])
            candidate = torch.matmul(alpha, values.float())
        return candidate, alpha, values.float()


class SlotModalityRouter(nn.Module):
    def __init__(self, dim, temperature=1.):
        super().__init__()
        self.evidence = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.context = nn.Linear(dim, dim, bias=False)
        self.modality = nn.Parameter(torch.randn(3, dim) * .02)
        self.score = nn.Linear(dim, 1, bias=False)
        self.null_query = nn.Linear(dim, dim, bias=False)
        self.null_context = nn.Linear(dim, dim, bias=False)
        self.null_score = nn.Linear(dim, 1)
        self.temperature = temperature
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.null_score.weight)
        nn.init.constant_(self.null_score.bias, -2.)

    def forward(self, query, context, candidates, presence):
        x = (self.evidence(candidates) + self.query(query)[:, :, None]
             + self.context(context)[:, None, None] + self.modality[None, None])
        scores = self.score(torch.tanh(x)).squeeze(-1)
        null = self.null_score(torch.tanh(self.null_query(query) + self.null_context(context)[:, None]))
        available = torch.cat((presence.bool(), torch.ones_like(presence[:, :1], dtype=torch.bool)), -1)
        return masked_softmax(torch.cat((scores, null), -1) / self.temperature, available[:, None])


class EvidenceTokenComposer(nn.Module):
    def __init__(self, settings, hidden_size):
        super().__init__()
        self.modality_types = nn.Parameter(torch.randn(3, settings.dim) * .02)
        self.base_prefix = nn.Parameter(torch.randn(settings.slots, hidden_size) * .02)
        self.up_project = nn.Linear(settings.dim, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.up_project.weight)
        self.register_buffer("gamma", torch.tensor(settings.gamma))

    def compose_mixed(self, mixed):
        return self.base_prefix[None] + self.gamma * self.up_project(mixed)

    def forward(self, candidates, gates):
        mixed = (gates[..., :3, None] * (candidates + self.modality_types[None, None])).sum(2)
        return self.compose_mixed(mixed), mixed


class HierarchicalEvidenceRouter(nn.Module):
    def __init__(self, settings, hidden_size):
        super().__init__()
        d = settings.dim
        self.slots = settings.slots
        self.relations = nn.ModuleDict({m: IntraModalRelationEncoder(settings) for m in MODALITIES})
        self.context_mlp = nn.Sequential(nn.Linear(3 * d + 3, d), nn.GELU(), nn.Linear(d, d))
        self.context_to_query = nn.Linear(d, d, bias=False)
        self.slot_queries = nn.Parameter(torch.randn(settings.slots, d) * .02)
        self.readers = nn.ModuleDict({m: EvidenceReader(d) for m in MODALITIES})
        self.norms = nn.ModuleDict({m: nn.LayerNorm(d, elementwise_affine=False) for m in MODALITIES})
        self.slot_router = SlotModalityRouter(d, settings.gate_temperature)
        self.composer = EvidenceTokenComposer(settings, hidden_size)

    def forward(self, features, masks, presence, uniform=False):
        hidden = {m: self.relations[m](features[m], masks[m]) for m in MODALITIES}
        means = [hidden[m].float().sum(1) / masks[m].sum(1, keepdim=True).clamp_min(1) for m in MODALITIES]
        context = self.context_mlp(torch.cat((*means, presence.float()), -1))
        query = self.slot_queries[None] + self.context_to_query(context)[:, None]
        raw, alpha, values = {}, {}, {}
        for m in MODALITIES:
            raw[m], alpha[m], values[m] = self.readers[m](query, hidden[m], masks[m])
        candidates = torch.stack([self.norms[m](raw[m]) for m in MODALITIES], 2)
        if uniform:
            count = presence.sum(-1, keepdim=True)
            real = presence.float() / count.clamp_min(1)
            # A completely empty input has only a base prefix, even during warmup.
            gates = torch.cat((real, (count == 0).float()), -1)[:, None].expand(-1, self.slots, -1)
        else:
            gates = self.slot_router(query, context, candidates, presence)
        tokens, mixed = self.composer(candidates, gates)
        joint = {m: (gates[..., i, None] * alpha[m]).mean(1) for i, m in enumerate(MODALITIES)}
        return dict(tokens=tokens, mixed=mixed, candidates=candidates, raw_candidates=raw,
                    values=values, alpha=alpha, gates=gates, G=gates.mean(1), J=joint,
                    masks=masks, presence=presence, query=query, context=context)

    def delete_modality(self, evidence, modalities):
        if isinstance(modalities, str):
            modalities = [modalities]
        mixed = evidence["mixed"]
        for m in modalities:
            i = MODALITIES.index(m)
            part = evidence["candidates"][:, :, i] + self.composer.modality_types[i]
            mixed = mixed - evidence["gates"][:, :, i, None] * part
        return self.composer.compose_mixed(mixed)

    def delete_positions(self, evidence, modality, removed):
        """Delete contextual read contributions using frozen full-pass LN statistics.

        Sum_i alpha_i * ((V_i - mean(e))/std(e) + type) == LN(e) + type.
        Thus deleting all positions equals deleting the modality injection.
        """
        i = MODALITIES.index(modality)
        raw = evidence["raw_candidates"][modality].float()
        mean = raw.mean(-1, keepdim=True)
        std = (raw.var(-1, unbiased=False, keepdim=True) + self.norms[modality].eps).sqrt()
        selected = removed.bool() & evidence["masks"][modality]
        weights = evidence["alpha"][modality] * selected[:, None]
        mass = weights.sum(-1, keepdim=True)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            value = torch.matmul(weights.float(), evidence["values"][modality].float())
        contribution = (value - mass * mean) / std + mass * self.composer.modality_types[i]
        mixed = evidence["mixed"] - evidence["gates"][:, :, i, None] * contribution
        return self.composer.compose_mixed(mixed)


class UHERAModel(nn.Module):
    def __init__(self, llm, tokenizer, settings=None, task_prompt=None):
        super().__init__()
        self.settings = settings or ModelSettings()
        self.llm = llm.requires_grad_(False)
        self.llm.eval()
        self.llm.config.use_cache = False
        self.tokenizer = tokenizer
        self.hidden_size = int(llm.config.hidden_size)
        self.text_projection = nn.Sequential(nn.Linear(self.hidden_size, self.settings.dim), nn.GELU())
        self.audio_encoder = PackedLSTMSequenceEncoder(self.settings.audio_features, self.settings.audio_hidden, self.settings.dim)
        self.vision_encoder = PackedLSTMSequenceEncoder(self.settings.vision_features, self.settings.vision_hidden, self.settings.dim)
        self.evidence = HierarchicalEvidenceRouter(self.settings, self.hidden_size)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.stage = "evidence_warmup"
        self.routing_mode = "uniform"
        self.task_prompt = task_prompt or "Please predict the sentiment intensity of the above multimodal content in the range [-3.0, +3.0]. Assistant: The sentiment is"
        for name, ids in {
            "bos_ids": [int(tokenizer.bos_token_id)],
            "before_ids": tokenizer.encode("<Multimodal>", add_special_tokens=False),
            "after_ids": tokenizer.encode("</Multimodal>", add_special_tokens=False),
            "task_ids": tokenizer.encode(self.task_prompt, add_special_tokens=False),
        }.items():
            self.register_buffer(name, torch.tensor(ids, dtype=torch.long))
        self.set_stage("evidence_warmup")

    def embedding_layer(self):
        return self.llm.base_model.get_input_embeddings()

    def train(self, mode=True):
        super().train(mode)
        self.llm.eval()
        if mode and self.stage == "router_warmup":
            self.text_projection.eval()
            self.audio_encoder.eval()
            self.vision_encoder.eval()
            self.evidence.eval()
            self.evidence.slot_router.train()
        return self

    def set_stage(self, stage):
        if stage not in (*STAGES, "eval"):
            raise ValueError(f"unknown stage: {stage}")
        self.stage = stage
        if stage != "eval":
            self.routing_mode = "uniform" if stage == "evidence_warmup" else "learned"
        for name, parameter in self.named_parameters():
            router = name.startswith("evidence.slot_router.")
            parameter.requires_grad_(not name.startswith("llm.") and stage != "eval" and
                                     (stage == "joint" or (router if stage == "router_warmup" else not router)))
        self.train(stage != "eval")

    def trainable_parameter_groups(self, learning_rate):
        groups = []
        for kind in ("evidence", "router"):
            params = [p for n, p in self.named_parameters() if p.requires_grad and
                      ((n.startswith("evidence.slot_router.")) == (kind == "router"))]
            if params:
                groups.append(dict(params=params, lr=learning_rate, name=kind))
        ids = [id(p) for group in groups for p in group["params"]]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(ids) != len(set(ids)) or set(ids) != expected or not ids:
            raise RuntimeError("optimizer parameter coverage mismatch")
        return groups

    def encode_modalities(self, text, audio, vision, presence=None):
        tensor = text[0]
        b = tensor.shape[0]
        if presence is None:
            presence = torch.ones(b, 3, device=tensor.device)
        if presence.shape != (b, 3) or not bool(((presence == 0) | (presence == 1)).all()):
            raise ValueError("presence must be binary [B,3]")
        text_mask = tensor[:, 1].bool() & presence[:, 0, None].bool()
        raw = self.embedding_layer()(tensor[:, 0].long())
        features = {"text": self.text_projection(raw.float()).masked_fill(~text_mask[..., None], 0)}
        masks = {"text": text_mask}
        for i, (name, data, encoder) in enumerate((('audio', audio, self.audio_encoder), ('vision', vision, self.vision_encoder)), 1):
            sequence, lengths = data
            active_lengths = lengths.long() * presence[:, i].long()
            features[name], masks[name] = encoder(sequence, active_lengths)
        effective = torch.stack([masks[m].any(-1) for m in MODALITIES], -1).float()
        return features, masks, effective

    def encode_evidence(self, text, audio, vision, presence=None):
        features, masks, effective = self.encode_modalities(text, audio, vision, presence)
        return self.evidence(features, masks, effective, uniform=self.routing_mode == "uniform")

    def final_prefix(self, tokens, text_tensor, presence):
        b = tokens.shape[0]
        embed = self.embedding_layer()
        dtype = embed.weight.dtype
        prompt = lambda ids: embed(ids[None].expand(b, -1))
        raw = embed(text_tensor[:, 0].long())
        pieces = [prompt(self.bos_ids), prompt(self.before_ids), tokens.to(dtype),
                  prompt(self.after_ids), raw, prompt(self.task_ids)]
        masks = [torch.ones(x.shape[:2], dtype=torch.long, device=tokens.device) for x in pieces]
        masks[4] = text_tensor[:, 1].long() * presence[:, 0, None].long()
        prefix, attention, _ = compact_left_padding(torch.cat(pieces, 1), torch.cat(masks, 1))
        return prefix, attention

    def _causal_forward(self, embeddings, attention):
        positions = (attention.cumsum(-1) - 1).masked_fill(attention == 0, 0)
        return self.llm(inputs_embeds=embeddings, attention_mask=attention, position_ids=positions,
                        use_cache=False, return_dict=True)

    def _teacher_forcing_losses(self, prefix, mask, labels):
        strings = [f"{float(x):+.1f}" for x in labels.detach().float().clamp(-3, 3)]
        encoded = self.tokenizer(strings, padding=True, add_special_tokens=False, return_tensors="pt")
        ids, target_mask = (encoded[k].to(prefix.device) for k in ("input_ids", "attention_mask"))
        label_embed = self.embedding_layer()(ids).to(prefix.dtype)
        inputs = torch.cat((prefix, label_embed), 1)
        ignore = torch.full(mask.shape, -100, dtype=torch.long, device=prefix.device)
        targets = torch.cat((ignore, ids.masked_fill(~target_mask.bool(), -100)), 1)
        inputs, attention, targets = compact_left_padding(inputs, torch.cat((mask, target_mask), 1), targets)
        return per_sample_causal_loss(self._causal_forward(inputs, attention).logits, targets)

    def forward(self, labels, text, audio, vision, presence=None, stage=None, utility_targets=None, utility_weight=.1):
        if stage is not None and stage != self.stage:
            raise ValueError("call set_stage before forward to set optimizer/freeze policy")
        evidence = self.encode_evidence(text, audio, vision, presence)
        prefix, mask = self.final_prefix(evidence["tokens"], text[0], evidence["presence"])
        per_sample = self._teacher_forcing_losses(prefix, mask, labels.reshape(-1))
        utility = per_sample.new_zeros(())
        if self.stage in ("router_warmup", "joint"):
            if utility_targets is None or utility_targets.shape != evidence["G"].shape:
                raise ValueError("learned routing requires fixed reference targets [B,4]")
            targets = utility_targets.detach().float()
            if not bool(torch.isfinite(targets).all()) or bool((targets < 0).any()) or not torch.allclose(targets.sum(-1), torch.ones_like(targets[:, 0]), atol=1e-5):
                raise ValueError("invalid utility probability targets")
            if bool((targets[:, :3].masked_select(~evidence["presence"].bool()) > 0).any()):
                raise ValueError("utility targets allocate mass to absent modalities")
            utility = (targets * (targets.clamp_min(1e-8).log() - evidence["G"].float().clamp_min(1e-8).log())).sum(-1).mean()
        generation = per_sample.mean()
        return dict(Loss=generation + utility_weight * utility, GenerationLoss=generation,
                    UtilityLoss=utility, PerSampleLoss=per_sample, **evidence)

    @staticmethod
    def parse_responses(responses):
        values, invalid, out_of_range = [], [], []
        for index, response in enumerate(responses):
            match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", response.replace("−", "-").replace("–", "-"))
            if match is None:
                invalid.append(index)
                values.append(0.)
                continue
            value = float(match.group(0))
            if not math.isfinite(value) or not -3 <= value <= 3:
                out_of_range.append(index)
                value = 0.
            values.append(value)
        return values, dict(raw_responses=responses, invalid_indices=invalid, out_of_range_indices=out_of_range)

    @torch.no_grad()
    def decode_prefix(self, prefix, mask):
        generated = []
        finished = torch.zeros(prefix.shape[0], dtype=torch.bool, device=prefix.device)
        for _ in range(self.settings.max_new_tokens):
            token = self._causal_forward(prefix, mask).logits[:, -1].argmax(-1)
            token = torch.where(finished, torch.full_like(token, self.eos_token_id), token)
            generated.append(token)
            finished |= token == self.eos_token_id
            if bool(finished.all()):
                break
            prefix = torch.cat((prefix, self.embedding_layer()(token[:, None]).to(prefix.dtype)), 1)
            mask = torch.cat((mask, torch.ones_like(token[:, None])), 1)
        responses = self.tokenizer.batch_decode(torch.stack(generated, 1), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return self.parse_responses(responses)

    @torch.no_grad()
    def generate(self, text, audio, vision, presence=None, return_diagnostics=False):
        evidence = self.encode_evidence(text, audio, vision, presence)
        values, parsing = self.decode_prefix(*self.final_prefix(evidence["tokens"], text[0], evidence["presence"]))
        if return_diagnostics:
            return values, dict(**parsing, evidence=evidence)
        return values

    def experiment_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("llm.")}

    def load_experiment_state_dict(self, payload):
        if payload.get("architecture") != ARCHITECTURE or payload.get("model_settings") != asdict(self.settings):
            raise ValueError("incompatible U-HERA checkpoint architecture/settings")
        state = payload["model"]
        if any(k.startswith("llm.") for k in state):
            raise ValueError("adapter checkpoint must not contain frozen LLM weights")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith("llm.") for k in result.missing_keys):
            raise ValueError(f"incomplete checkpoint: {result}")
        self.routing_mode = payload["routing_mode"]
