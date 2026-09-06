"""Qwen-1.8B multimodal pseudo-token router.

The language model is shared by the three diagnostic passes and the final
generative pass.  Its parameters stay frozen, while gradients through it train
the modality adapters and (during stage 1) the shared ordinal head.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .math_utils import (
    masked_softmax,
    normalized_entropy,
    normalized_js_divergence,
    soft_cross_entropy,
    soft_ordinal_targets,
)


LOGGER = logging.getLogger("mse_router")
MODALITIES = ("text", "audio", "vision")
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.to(dtype=torch.bool)
        if bool((valid.sum(dim=1) == 0).any()):
            raise ValueError("every text sample must contain at least one valid token")
        scores = self.score(tokens).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


class PackedLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        safe_lengths = lengths.reshape(-1).long().clamp(min=1, max=sequence.shape[1])
        packed = pack_padded_sequence(
            sequence,
            safe_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.rnn(packed)
        # Indexing rather than squeeze preserves the batch dimension for B=1.
        return self.projection(self.dropout(hidden[-1]))


class MultiScaleProjector(nn.Module):
    """The original MSE-Adapter multi-scale projector, instantiated per modality."""

    def __init__(
        self, input_size: int = 256, hidden_size: int = 2048, pseudo_tokens: int = 4
    ) -> None:
        super().__init__()
        scale_hidden = 256
        self.scale1 = nn.Sequential(
            nn.Linear(input_size, hidden_size // 8),
            nn.GELU(),
            nn.Linear(hidden_size // 8, scale_hidden),
        )
        self.scale2 = nn.Sequential(
            nn.Linear(input_size, hidden_size // 32),
            nn.GELU(),
            nn.Linear(hidden_size // 32, scale_hidden),
        )
        self.scale3 = nn.Sequential(
            nn.Linear(input_size, hidden_size // 16),
            nn.GELU(),
            nn.Linear(hidden_size // 16, scale_hidden),
        )
        self.integrating = nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1)
        self.multi_scale_projector = nn.Linear(scale_hidden, hidden_size)
        self.token_projector = nn.Linear(1, pseudo_tokens)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scales = torch.stack(
            [self.scale1(features), self.scale2(features), self.scale3(features)], dim=2
        )
        integrated = self.integrating(scales.unsqueeze(1)).squeeze(3).squeeze(1)
        hidden = self.multi_scale_projector(integrated)
        return self.token_projector(hidden.unsqueeze(-1)).permute(0, 2, 1)


class Router(nn.Module):
    def __init__(self, input_size: int = 30, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        return masked_softmax(self.network(features), presence)


class QwenMseRouter(nn.Module):
    """Conflict-aware pseudo-token fusion while preserving generative supervision."""

    def __init__(self, args: Any) -> None:
        super().__init__()
        from modelscope import AutoModelForCausalLM, AutoTokenizer

        self.hidden_size = int(getattr(args, "router_hidden_size", 2048))
        self.pseudo_tokens = int(getattr(args, "pseudo_tokens", 4))
        self.max_new_tokens = int(getattr(args, "max_new_tokens", 4))
        self.aux_weight = float(getattr(args, "router_aux_weight", 0.3))
        self.router_variant = str(getattr(args, "router_variant", "full"))
        self.task_prompt = str(
            getattr(
                args,
                "task_specific_prompt",
                "Please predict the sentiment intensity of the above multimodal "
                "content in the range [-3.0, +3.0]. Assistant: The sentiment is",
            )
        )
        diagnostic_prompt = str(
            getattr(
                args,
                "diagnostic_prompt",
                "Predict this modality's sentiment intensity from -3 to +3.",
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrain_LM, padding_side="left", trust_remote_code=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            args.pretrain_LM, trust_remote_code=True, torch_dtype=torch.float16
        ).half()
        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        self.llm.config.use_cache = False
        if bool(getattr(args, "gradient_checkpointing", True)):
            self.llm.gradient_checkpointing_enable()

        actual_hidden = int(self.llm.config.hidden_size)
        if actual_hidden != self.hidden_size:
            raise ValueError(
                f"Qwen hidden size is {actual_hidden}, expected {self.hidden_size}"
            )
        eos = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        bos = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.tokenizer.pad_token_id = eos
        self.tokenizer.bos_token_id = bos
        self.eos_token_id = int(eos)
        self.bos_token_id = int(bos)

        text_dim, audio_dim, vision_dim = args.feature_dims
        if int(text_dim) != self.hidden_size:
            raise ValueError(
                f"token embedding dimension is {text_dim}, expected {self.hidden_size}"
            )
        self.text_pool = MaskedAttentionPool(self.hidden_size)
        self.text_projection = nn.Sequential(
            nn.Linear(self.hidden_size, 256), nn.GELU()
        )
        self.text_adapter = MultiScaleProjector(
            256, self.hidden_size, self.pseudo_tokens
        )
        self.audio_encoder = PackedLSTMEncoder(
            int(audio_dim),
            int(getattr(args, "a_lstm_hidden_size", 64)),
            num_layers=int(getattr(args, "a_lstm_layers", 1)),
            dropout=float(getattr(args, "a_lstm_dropout", 0.0)),
        )
        self.audio_adapter = MultiScaleProjector(
            256, self.hidden_size, self.pseudo_tokens
        )
        self.vision_encoder = PackedLSTMEncoder(
            int(vision_dim),
            int(getattr(args, "v_lstm_hidden_size", 32)),
            num_layers=int(getattr(args, "v_lstm_layers", 1)),
            dropout=float(getattr(args, "v_lstm_dropout", 0.0)),
        )
        self.vision_adapter = MultiScaleProjector(
            256, self.hidden_size, self.pseudo_tokens
        )
        self.modality_embeddings = nn.Parameter(torch.empty(3, self.hidden_size))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)

        self.ordinal_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 7),
        )
        self.router = Router(30, 64)
        self.log_temperatures = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.register_buffer("anchors", torch.arange(-3.0, 4.0), persistent=True)

        self._register_prompt("bos_ids", [self.bos_token_id])
        self._register_prompt("wrapper_before_ids", self._token_ids("<Multimodal>"))
        self._register_prompt("wrapper_after_ids", self._token_ids("</Multimodal>"))
        self._register_prompt("diagnostic_prompt_ids", self._token_ids(diagnostic_prompt))
        self._register_prompt("task_prompt_ids", self._token_ids(self.task_prompt))

    def _token_ids(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        return encoded["input_ids"].reshape(-1).tolist()

    def _register_prompt(self, name: str, ids: Iterable[int]) -> None:
        self.register_buffer(
            name, torch.tensor(list(ids), dtype=torch.long), persistent=False
        )

    @property
    def temperatures(self) -> torch.Tensor:
        return self.log_temperatures.exp().clamp(0.05, 10.0)

    def set_temperatures(self, temperatures: torch.Tensor | Iterable[float]) -> None:
        values = torch.as_tensor(
            temperatures,
            device=self.log_temperatures.device,
            dtype=self.log_temperatures.dtype,
        )
        if values.shape != (3,) or not bool(torch.isfinite(values).all()):
            raise ValueError("temperatures must contain three finite values")
        if bool(((values < 0.05) | (values > 10.0)).any()):
            raise ValueError("temperatures must lie in [0.05, 10.0]")
        with torch.no_grad():
            self.log_temperatures.copy_(values.log())

    def _embedding_layer(self) -> nn.Module:
        return self.llm.base_model.get_input_embeddings()

    def _expanded_prompt(self, ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        return self._embedding_layer()(ids.unsqueeze(0).expand(batch_size, -1))

    def encode_modalities(
        self,
        text: tuple[torch.Tensor, torch.Tensor],
        audio: tuple[torch.Tensor, torch.Tensor],
        vision: tuple[torch.Tensor, torch.Tensor],
        presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_tensor, _ = text
        audio_tensor, audio_lengths = audio
        vision_tensor, vision_lengths = vision
        token_ids = text_tensor[:, 0, :].long()
        text_mask = text_tensor[:, 1, :].long()
        text_embeddings = self._embedding_layer()(token_ids)
        text_features = self.text_projection(
            self.text_pool(text_embeddings, text_mask)
        )
        audio_features = self.audio_encoder(audio_tensor, audio_lengths)
        vision_features = self.vision_encoder(vision_tensor, vision_lengths)
        pseudo = torch.stack(
            [
                self.text_adapter(text_features),
                self.audio_adapter(audio_features),
                self.vision_adapter(vision_features),
            ],
            dim=1,
        )
        type_embeddings = self.modality_embeddings.to(dtype=pseudo.dtype)
        pseudo = pseudo + type_embeddings[None, :, None, :]
        if presence is not None:
            pseudo = pseudo * presence[:, :, None, None].to(dtype=pseudo.dtype)
        return pseudo

    def _wrapped_prefix(self, pseudo_tokens: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        batch_size = pseudo_tokens.shape[0]
        dtype = pseudo_tokens.dtype
        parts = [
            self._expanded_prompt(self.bos_ids, batch_size).to(dtype=dtype),
            self._expanded_prompt(self.wrapper_before_ids, batch_size).to(dtype=dtype),
            pseudo_tokens,
            self._expanded_prompt(self.wrapper_after_ids, batch_size).to(dtype=dtype),
            self._expanded_prompt(prompt, batch_size).to(dtype=dtype),
        ]
        return torch.cat(parts, dim=1)

    def diagnostic_logits(self, pseudo: torch.Tensor) -> torch.Tensor:
        batch_size = pseudo.shape[0]
        flattened = pseudo.reshape(
            batch_size * 3, self.pseudo_tokens, self.hidden_size
        )
        sequence = self._wrapped_prefix(flattened, self.diagnostic_prompt_ids)
        attention_mask = torch.ones(
            sequence.shape[:2], device=sequence.device, dtype=torch.long
        )
        outputs = self.llm.transformer(
            inputs_embeds=sequence,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        last_valid = outputs.last_hidden_state[:, -1, :]
        return self.ordinal_head(last_valid).reshape(batch_size, 3, 7)

    def _router_statistics(
        self, logits: torch.Tensor, presence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = F.softmax(
            logits.float() / self.temperatures[None, :, None], dim=-1
        )
        present = presence.to(dtype=torch.bool)
        uniform = torch.full_like(probabilities, 1.0 / probabilities.shape[-1])
        probabilities = torch.where(present[:, :, None], probabilities, uniform)
        entropy = normalized_entropy(probabilities)
        pairs = ((0, 1), (0, 2), (1, 2))
        conflicts = []
        for first, second in pairs:
            pair_present = present[:, first] & present[:, second]
            divergence = normalized_js_divergence(
                probabilities[:, first], probabilities[:, second]
            )
            conflicts.append(divergence * pair_present.to(divergence.dtype))
        conflict = torch.stack(conflicts, dim=-1)
        entropy = entropy * presence.to(entropy.dtype)
        features = torch.cat(
            [
                probabilities.reshape(probabilities.shape[0], -1),
                conflict,
                entropy,
                presence.float(),
            ],
            dim=-1,
        )
        return probabilities, conflict, entropy, features

    def _apply_router_variant(self, features: torch.Tensor) -> torch.Tensor:
        features = features.clone()
        if self.router_variant == "full":
            return features
        if self.router_variant == "no_conflict":
            features[:, 21:24] = 0
        elif self.router_variant == "no_uncertainty":
            features[:, 24:27] = 0
        elif self.router_variant == "predictions_only":
            features[:, 21:27] = 0
        elif self.router_variant == "uncertainty_only":
            features[:, :24] = 0
        elif self.router_variant != "uniform":
            raise ValueError(f"unknown router variant: {self.router_variant}")
        return features

    def route(
        self, logits: torch.Tensor, presence: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probabilities, conflicts, entropy, raw_features = self._router_statistics(
            logits, presence
        )
        router_features = self._apply_router_variant(raw_features.detach())
        if self.router_variant == "uniform":
            weights = presence.float() / presence.float().sum(dim=-1, keepdim=True)
        else:
            weights = self.router(router_features, presence)
        return weights, {
            "probabilities": probabilities,
            "conflicts": conflicts,
            "entropy": entropy,
            "router_features": router_features,
        }

    def gated_pseudo_tokens(
        self, pseudo: torch.Tensor, weights: torch.Tensor, presence: torch.Tensor
    ) -> torch.Tensor:
        present_count = presence.float().sum(dim=-1, keepdim=True)
        scales = present_count * weights
        gated = pseudo * scales[:, :, None, None].to(dtype=pseudo.dtype)
        return gated.reshape(pseudo.shape[0], 3 * self.pseudo_tokens, self.hidden_size)

    def gated_non_text_pseudo_tokens(
        self, pseudo: torch.Tensor, weights: torch.Tensor, presence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate audio/vision tokens while keeping natural text as a residual path.

        Multiplication by the number of present modalities preserves unit scale
        for a uniform router. The returned mask removes tokens belonging to a
        deliberately dropped modality from the final Qwen attention graph.
        """
        present_count = presence.float().sum(dim=-1, keepdim=True)
        scales = present_count * weights[:, 1:]
        gated = pseudo[:, 1:] * scales[:, :, None, None].to(dtype=pseudo.dtype)
        batch_size = pseudo.shape[0]
        gated = gated.reshape(
            batch_size, 2 * self.pseudo_tokens, self.hidden_size
        )
        token_mask = presence[:, 1:, None].expand(-1, -1, self.pseudo_tokens)
        return gated, token_mask.reshape(batch_size, -1).long()

    def final_prefix(
        self,
        pseudo: torch.Tensor,
        weights: torch.Tensor,
        presence: torch.Tensor,
        text_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build v2 input: gated A/V pseudo tokens plus original text tokens."""
        batch_size = pseudo.shape[0]
        gated_av, av_mask = self.gated_non_text_pseudo_tokens(
            pseudo, weights, presence
        )
        text_ids = text_tensor[:, 0, :].long()
        text_mask = text_tensor[:, 1, :].long()
        text_mask = text_mask * presence[:, 0, None].long()
        raw_text = self._embedding_layer()(text_ids).to(dtype=pseudo.dtype)
        ones = lambda length: torch.ones(
            batch_size, length, device=pseudo.device, dtype=torch.long
        )
        bos = self._expanded_prompt(self.bos_ids, batch_size).to(dtype=pseudo.dtype)
        before = self._expanded_prompt(self.wrapper_before_ids, batch_size).to(
            dtype=pseudo.dtype
        )
        after = self._expanded_prompt(self.wrapper_after_ids, batch_size).to(
            dtype=pseudo.dtype
        )
        task = self._expanded_prompt(self.task_prompt_ids, batch_size).to(
            dtype=pseudo.dtype
        )
        embeddings = torch.cat([bos, before, gated_av, after, raw_text, task], dim=1)
        attention_mask = torch.cat(
            [
                ones(bos.shape[1]),
                ones(before.shape[1]),
                av_mask,
                ones(after.shape[1]),
                text_mask,
                ones(task.shape[1]),
            ],
            dim=1,
        )
        return embeddings, attention_mask

    def _teacher_forcing_loss(
        self,
        prefix: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        label_strings = [
            f"{float(value):+.1f}" for value in labels.detach().float().clamp(-1, 1)
        ]
        encoded = self.tokenizer(
            label_strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(prefix.device)
        label_ids = encoded["input_ids"]
        label_mask = encoded["attention_mask"].long()
        label_embeddings = self._embedding_layer()(label_ids).to(dtype=prefix.dtype)
        inputs = torch.cat([prefix, label_embeddings], dim=1)
        ignore = torch.full(
            prefix.shape[:2], -100, device=prefix.device, dtype=torch.long
        )
        targets = label_ids.masked_fill(label_mask == 0, -100)
        targets = torch.cat([ignore, targets], dim=1)
        attention_mask = torch.cat([prefix_attention_mask, label_mask], dim=1)
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        output = self.llm(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=targets,
            use_cache=False,
            return_dict=True,
        )
        return output.loss

    def forward(
        self,
        labels: torch.Tensor,
        text: tuple[torch.Tensor, torch.Tensor],
        audio: tuple[torch.Tensor, torch.Tensor],
        vision: tuple[torch.Tensor, torch.Tensor],
        presence: torch.Tensor | None = None,
        stage: str = "stage1",
    ) -> dict[str, torch.Tensor]:
        batch_size = labels.reshape(-1).shape[0]
        if presence is None:
            presence = torch.ones(batch_size, 3, device=labels.device)
        presence = presence.to(device=labels.device, dtype=torch.float32)
        pseudo = self.encode_modalities(text, audio, vision, presence)
        logits = self.diagnostic_logits(pseudo)
        weights, diagnostics = self.route(logits, presence)
        prefix, prefix_attention_mask = self.final_prefix(
            pseudo, weights, presence, text[0]
        )
        generation_loss = self._teacher_forcing_loss(
            prefix, prefix_attention_mask, labels.reshape(-1)
        )

        ordinal_targets = soft_ordinal_targets(labels, self.anchors)
        per_modality = soft_cross_entropy(
            logits.float() / self.temperatures[None, :, None],
            ordinal_targets[:, None, :].expand(-1, 3, -1),
        )
        auxiliary_loss = (
            per_modality * presence
        ).sum() / presence.sum().clamp_min(1.0)
        loss = generation_loss
        if stage == "stage1":
            loss = loss + self.aux_weight * auxiliary_loss
        elif stage != "router":
            raise ValueError(f"unknown training stage: {stage}")
        return {
            "Loss": loss,
            "GenerationLoss": generation_loss,
            "AuxiliaryLoss": auxiliary_loss,
            "OrdinalLogits": logits,
            "Weights": weights,
            "Presence": presence,
            **diagnostics,
        }

    @staticmethod
    def parse_responses(responses: list[str]) -> tuple[list[float], dict[str, Any]]:
        values: list[float] = []
        invalid: list[int] = []
        out_of_range: list[int] = []
        for index, response in enumerate(responses):
            normalized = response.replace("–", "-").replace("−", "-")
            match = NUMBER_PATTERN.search(normalized)
            if match is None:
                values.append(0.0)
                invalid.append(index)
                continue
            try:
                value = float(match.group(0))
            except ValueError:
                values.append(0.0)
                invalid.append(index)
                continue
            if not math.isfinite(value) or value < -1.0 or value > 1.0:
                values.append(0.0)
                out_of_range.append(index)
            else:
                values.append(value)
        return values, {
            "raw_responses": responses,
            "invalid_indices": invalid,
            "out_of_range_indices": out_of_range,
            "invalid_count": len(invalid),
            "out_of_range_count": len(out_of_range),
        }

    @torch.no_grad()
    def diagnose(
        self,
        text: tuple[torch.Tensor, torch.Tensor],
        audio: tuple[torch.Tensor, torch.Tensor],
        vision: tuple[torch.Tensor, torch.Tensor],
        presence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = text[0].shape[0]
        if presence is None:
            presence = torch.ones(batch_size, 3, device=text[0].device)
        presence = presence.float()
        pseudo = self.encode_modalities(text, audio, vision, presence)
        logits = self.diagnostic_logits(pseudo)
        weights, diagnostics = self.route(logits, presence)
        return {"logits": logits, "weights": weights, **diagnostics}

    @torch.no_grad()
    def generate(
        self,
        text: tuple[torch.Tensor, torch.Tensor],
        audio: tuple[torch.Tensor, torch.Tensor],
        vision: tuple[torch.Tensor, torch.Tensor],
        presence: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> list[float] | tuple[list[float], dict[str, Any]]:
        batch_size = text[0].shape[0]
        if presence is None:
            presence = torch.ones(batch_size, 3, device=text[0].device)
        presence = presence.float()
        pseudo = self.encode_modalities(text, audio, vision, presence)
        logits = self.diagnostic_logits(pseudo)
        weights, router_diagnostics = self.route(logits, presence)
        prefix, prefix_attention_mask = self.final_prefix(
            pseudo, weights, presence, text[0]
        )
        # Qwen's GenerationMixin path assumes a cache when inputs_embeds are
        # supplied.  This explicit loop keeps use_cache=False while still
        # appending every generated token to the next full-prefix pass.
        running_embeddings = prefix
        running_attention_mask = prefix_attention_mask
        generated: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, device=prefix.device, dtype=torch.bool)
        for _ in range(self.max_new_tokens):
            position_ids = running_attention_mask.cumsum(dim=-1) - 1
            position_ids.masked_fill_(running_attention_mask == 0, 0)
            output = self.llm(
                inputs_embeds=running_embeddings,
                attention_mask=running_attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            next_token = output.logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, self.eos_token_id), next_token
            )
            generated.append(next_token)
            finished = finished | (next_token == self.eos_token_id)
            next_embedding = self._embedding_layer()(next_token[:, None]).to(
                dtype=running_embeddings.dtype
            )
            running_embeddings = torch.cat(
                [running_embeddings, next_embedding], dim=1
            )
            running_attention_mask = torch.cat(
                [
                    running_attention_mask,
                    torch.ones(
                        batch_size, 1, device=prefix.device, dtype=torch.long
                    ),
                ],
                dim=1,
            )
            if bool(finished.all()):
                break
        outputs = torch.stack(generated, dim=1)
        responses = self.tokenizer.batch_decode(
            outputs,
            add_special_tokens=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        values, parsing = self.parse_responses(responses)
        if not return_diagnostics:
            return values
        diagnostics: dict[str, Any] = {
            **parsing,
            "weights": weights.detach().float().cpu(),
            "probabilities": router_diagnostics["probabilities"].detach().cpu(),
            "conflicts": router_diagnostics["conflicts"].detach().cpu(),
            "entropy": router_diagnostics["entropy"].detach().cpu(),
            "temperatures": self.temperatures.detach().cpu(),
        }
        return values, diagnostics

    def set_stage(self, stage: str) -> None:
        if stage not in {"stage1", "calibration", "router", "eval"}:
            raise ValueError(f"unknown stage: {stage}")
        for name, parameter in self.named_parameters():
            if name.startswith("llm.") or name == "log_temperatures":
                parameter.requires_grad = False
            elif stage == "stage1":
                parameter.requires_grad = True
            elif stage == "router":
                parameter.requires_grad = name.startswith("router.") and (
                    self.router_variant != "uniform"
                )
            else:
                parameter.requires_grad = False

    def adapter_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.text_pool,
            self.text_projection,
            self.text_adapter,
            self.audio_encoder,
            self.audio_adapter,
            self.vision_encoder,
            self.vision_adapter,
        )
        parameters = [parameter for module in modules for parameter in module.parameters()]
        parameters.append(self.modality_embeddings)
        return parameters

    def experiment_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only experiment parameters/buffers, never frozen Qwen weights."""
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith("llm.")
        }

    def load_experiment_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_non_llm = [
            name for name in incompatible.missing_keys if not name.startswith("llm.")
        ]
        if unexpected or missing_non_llm:
            raise RuntimeError(
                f"incompatible router checkpoint: missing={missing_non_llm}, "
                f"unexpected={unexpected}"
            )
