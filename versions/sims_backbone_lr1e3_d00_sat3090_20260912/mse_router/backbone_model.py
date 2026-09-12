"""Llama2 and ChatGLM3 backbones for the ordered-text Wasserstein V4 router.

Only the frozen language-model boundary differs from :class:`QwenMseRouter`.
The modality encoders, router, losses, training stages, and generation parser
are shared with Qwen so all backbones use ordered natural-text diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .sequence import compact_left_padding
from .model import (
    ARCHITECTURE_VERSION,
    MultiScaleProjector,
    PackedLSTMEncoder,
    QwenMseRouter,
    Router,
)


SUPPORTED_BACKBONES = ("llama2", "chatglm3")


class BackboneMseRouter(QwenMseRouter):
    """V4 conflict router with a frozen Llama2 or ChatGLM3 language model."""

    def __init__(self, args: Any) -> None:
        # QwenMseRouter.__init__ is deliberately not called because it loads a
        # Qwen checkpoint. All experiment modules below mirror its V4 layout.
        nn.Module.__init__(self)
        self.backbone_name = str(getattr(args, "router_backbone", ""))
        if self.backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"router_backbone must be one of {SUPPORTED_BACKBONES}, "
                f"got {self.backbone_name!r}"
            )

        self.hidden_size = int(getattr(args, "router_hidden_size", 4096))
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

        self.tokenizer, self.llm = self._load_backbone(args)
        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        self.llm.config.use_cache = False
        if bool(getattr(args, "gradient_checkpointing", True)):
            self.llm.gradient_checkpointing_enable()

        actual_hidden = int(self.llm.config.hidden_size)
        if actual_hidden != self.hidden_size:
            raise ValueError(
                f"{self.backbone_name} hidden size is {actual_hidden}, "
                f"expected {self.hidden_size}"
            )

        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError(f"{self.backbone_name} tokenizer has no EOS token")
        bos = self.tokenizer.bos_token_id
        if bos is None and hasattr(self.tokenizer, "get_command"):
            bos = self.tokenizer.get_command("<bos>")
        if bos is None:
            raise ValueError(f"{self.backbone_name} tokenizer has no BOS token")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = int(eos)
        self.eos_token_id = int(eos)
        self.bos_token_id = int(bos)

        text_dim, audio_dim, vision_dim = args.feature_dims
        if int(text_dim) != self.hidden_size:
            raise ValueError(
                f"token embedding dimension is {text_dim}, "
                f"expected {self.hidden_size}"
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
        self.modality_embeddings = nn.Parameter(torch.empty(2, self.hidden_size))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)

        self.ordinal_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(256, 7),
        )
        self.router = Router(30, 64)
        self.log_temperatures = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.register_buffer("anchors", torch.linspace(-1.0, 1.0, 7), persistent=True)
        self.register_buffer("architecture_version", torch.tensor(ARCHITECTURE_VERSION), persistent=True)

        self._register_prompt("bos_ids", [self.bos_token_id])
        self._register_prompt("wrapper_before_ids", self._token_ids("<Multimodal>"))
        self._register_prompt("wrapper_after_ids", self._token_ids("</Multimodal>"))
        self._register_prompt(
            "diagnostic_prompt_ids", self._token_ids(diagnostic_prompt)
        )
        self._register_prompt("task_prompt_ids", self._token_ids(self.task_prompt))

    def _load_backbone(self, args: Any) -> tuple[Any, nn.Module]:
        model_path = Path(args.pretrain_LM).resolve()
        if self.backbone_name == "llama2":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left",
                local_files_only=True,
                trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.float16,
            ).half()
            return tokenizer, model

        upstream_dir = Path(args.router_upstream_dir).resolve()
        if str(upstream_dir) not in sys.path:
            sys.path.insert(0, str(upstream_dir))
        from models.ChatGLM3.modeling_chatglm import (
            ChatGLMForConditionalGeneration,
        )
        from models.ChatGLM3.tokenization_chatglm import ChatGLMTokenizer

        tokenizer = ChatGLMTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            local_files_only=True,
            trust_remote_code=False,
        )
        model = ChatGLMForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
        ).half()
        return tokenizer, model

    def _embedding_layer(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    def _dummy_input_ids(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.full(
            embeddings.shape[:2],
            int(self.tokenizer.pad_token_id),
            device=embeddings.device,
            dtype=torch.long,
        )

    def _causal_forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Any:
        common = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
            "use_cache": False,
            "return_dict": True,
        }
        if self.backbone_name == "chatglm3":
            return self.llm(
                input_ids=self._dummy_input_ids(embeddings),
                input_fusion=embeddings,
                **common,
            )
        return self.llm(inputs_embeds=embeddings, **common)

    def _backbone_last_hidden(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        common = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
            "return_dict": True,
        }
        if self.backbone_name == "chatglm3":
            output = self.llm.transformer(
                input_ids=self._dummy_input_ids(embeddings),
                input_fusion=embeddings,
                **common,
            )
            # The bundled ChatGLM encoder is sequence-major: [S, B, H].
            return output.last_hidden_state[-1, :, :]
        output = self.llm.model(inputs_embeds=embeddings, **common)
        return output.last_hidden_state[:, -1, :]

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
        inputs, attention_mask, targets = compact_left_padding(inputs, attention_mask, targets)
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        output = self._causal_forward(
            inputs, attention_mask, position_ids, labels=targets
        )
        return output.loss

    @torch.no_grad()
    def generate(
        self,
        text: tuple[torch.Tensor, torch.Tensor],
        audio: tuple[torch.Tensor, torch.Tensor],
        vision: tuple[torch.Tensor, torch.Tensor],
        presence: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        weights_override: torch.Tensor | None = None,
    ) -> list[float] | tuple[list[float], dict[str, Any]]:
        batch_size = text[0].shape[0]
        if presence is None:
            presence = torch.ones(batch_size, 3, device=text[0].device)
        presence = presence.float()
        pseudo = self.encode_modalities(text, audio, vision, presence)
        logits = self.diagnostic_logits(pseudo, text[0], presence)
        weights, router_diagnostics = self.route(logits, presence)
        if weights_override is not None:
            weights = self.validate_weight_override(weights_override, presence)
        prefix, prefix_attention_mask = self.final_prefix(
            pseudo, weights, presence, text[0]
        )
        running_embeddings = prefix
        running_attention_mask = prefix_attention_mask
        generated: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, device=prefix.device, dtype=torch.bool)
        for _ in range(self.max_new_tokens):
            position_ids = running_attention_mask.cumsum(dim=-1) - 1
            position_ids.masked_fill_(running_attention_mask == 0, 0)
            output = self._causal_forward(
                running_embeddings, running_attention_mask, position_ids
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
