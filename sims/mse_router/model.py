# 核心模型：冻结语言模型，联合训练音视频适配器、共享情感诊断头和模态路由器。
"""冻结 ChatGLM3 的多模态情感路由：音视频适配、单模态诊断、冲突门控和生成式监督。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
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
    normalized_wasserstein_distance,
    soft_cross_entropy,
    soft_ordinal_targets,
)


from .sequence import compact_left_padding


LOGGER = logging.getLogger("mse_router")

ARCHITECTURE = "pseudo_text_diagnostics_wasserstein_router_v5"
ARCHITECTURE_VERSION = 5
CONFLICT_METRIC = "wasserstein_1_normalized"
MODALITIES = ("text", "audio", "vision")
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


# 按真实序列长度打包音视频特征，取末层隐藏状态并投影到统一特征空间。
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
    # 初始化变长序列 LSTM、丢弃层和输出投影；单层 LSTM 不启用层间 dropout。
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

    # 按真实长度打包序列，使用末层隐藏状态得到每个样本的编码。
    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        safe_lengths = lengths.reshape(-1).long().clamp(min=1, max=sequence.shape[1])
        packed = pack_padded_sequence(
            sequence,
            safe_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.rnn(packed)
        # 直接索引末层状态，避免 squeeze 在 B=1 时误删批次维度。
        return self.projection(self.dropout(hidden[-1]))


# 用三个不同宽度的投影分支融合特征，再生成语言模型可接收的伪 token。
class MultiScaleProjector(nn.Module):
    """The original MSE-Adapter multi-scale projector, instantiated per modality."""

    # 初始化多尺度分支、跨尺度卷积和伪 token 投影。
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

    # 融合三路尺度特征，输出 [B, 伪 token 数, 骨干隐藏维度]。
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scales = torch.stack(
            [self.scale1(features), self.scale2(features), self.scale3(features)], dim=2
        )
        integrated = self.integrating(scales.unsqueeze(1)).squeeze(3).squeeze(1)
        hidden = self.multi_scale_projector(integrated)
        return self.token_projector(hidden.unsqueeze(-1)).permute(0, 2, 1)


# 将 30 维诊断特征映射为三模态权重；缺失模态的权重固定为零。
class Router(nn.Module):
    # 建立两层路由网络，并将末层初始化为零以获得均匀初始权重。
    def __init__(self, input_size: int = 30, hidden_size: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )
        # 输出层从零开始，使初始 logits 一致，现存模态获得均匀权重。
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    # 计算三模态 logits，并通过存在掩码归一化为路由权重。
    def forward(self, features: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        return masked_softmax(self.network(features), presence)


# 仅使用 ChatGLM3 骨干；各模态共享冻结语言模型和情感诊断头。
class ChatGLMMseRouter(nn.Module):
    """ChatGLM3 多模态情感路由模型。"""

    # 加载目标骨干，并建立与统一 V4 路由结构对应的实验模块。
    def __init__(self, args: Any) -> None:
        # 直接初始化冻结 ChatGLM3 及本实验的音视频、诊断头和路由模块。
        nn.Module.__init__(self)
        self.backbone_name = "chatglm3"

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
            nn.Dropout(float(getattr(args, "router_dropout", 0.1))),
            nn.Linear(256, 7),
        )
        self.router = Router(30, 64, dropout=float(getattr(args, "router_dropout", 0.1)))
        self.register_buffer("anchors", torch.linspace(-1.0, 1.0, 7), persistent=True)
        self.register_buffer("architecture_version", torch.tensor(ARCHITECTURE_VERSION), persistent=True)

        self._register_prompt("bos_ids", [self.bos_token_id])
        self._register_prompt("wrapper_before_ids", self._token_ids("<Multimodal>"))
        self._register_prompt("wrapper_after_ids", self._token_ids("</Multimodal>"))
        self._register_prompt(
            "diagnostic_prompt_ids", self._token_ids(diagnostic_prompt)
        )
        self._register_prompt("task_prompt_ids", self._token_ids(self.task_prompt))

        # Append the text branch without changing existing module initialization
        # or the global RNG stream used by the baseline training loader.
        with torch.random.fork_rng(devices=[]):
            self.text_pool = MaskedAttentionPool(self.hidden_size)
            self.text_projection = nn.Sequential(
                nn.Linear(self.hidden_size, 256), nn.GELU()
            )
            self.text_adapter = MultiScaleProjector(
                256, self.hidden_size, self.pseudo_tokens
            )
            self.text_modality_embedding = nn.Parameter(torch.empty(self.hidden_size))
            nn.init.normal_(self.text_modality_embedding, mean=0.0, std=0.02)

    # 将提示文本编码成不自动附加特殊标记的 token ID 列表。
    def _token_ids(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        return encoded["input_ids"].reshape(-1).tolist()

    # 将提示 token 注册为随设备移动的缓冲区，但不写入实验检查点。
    def _register_prompt(self, name: str, ids: Iterable[int]) -> None:
        self.register_buffer(
            name, torch.tensor(list(ids), dtype=torch.long), persistent=False
        )

    # 本实验没有温度校准，日志中的温度恒为 1，不参与任何参数更新。
    @property
    def temperatures(self) -> torch.Tensor:
        return torch.ones(3, device=self.anchors.device, dtype=torch.float32)

    # 取得当前骨干的输入词嵌入层，供提示、原始文本和生成 token 共用。
    def _embedding_layer(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    # 把同一提示的 token ID 扩展到整个批次，并查表得到词嵌入。
    def _expanded_prompt(self, ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        return self._embedding_layer()(ids.unsqueeze(0).expand(batch_size, -1))

    # 三模态编码成伪 Token；文本伪 Token 仅供诊断，最终预测保留原始文本。
    def encode_modalities(
        self,
        text: tuple[torch.Tensor, torch.Tensor],
        audio: tuple[torch.Tensor, torch.Tensor],
        vision: tuple[torch.Tensor, torch.Tensor],
        presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        audio_tensor, audio_lengths = audio
        vision_tensor, vision_lengths = vision
        audio_features = self.audio_encoder(audio_tensor, audio_lengths)
        vision_features = self.vision_encoder(vision_tensor, vision_lengths)
        av_pseudo = torch.stack(
            [
                self.audio_adapter(audio_features),
                self.vision_adapter(vision_features),
            ],
            dim=1,
        )
        type_embeddings = self.modality_embeddings.to(dtype=av_pseudo.dtype)
        av_pseudo = av_pseudo + type_embeddings[None, :, None, :]
        text_tensor, _ = text
        text_embeddings = self._embedding_layer()(text_tensor[:, 0, :].long())
        text_features = self.text_projection(
            self.text_pool(text_embeddings, text_tensor[:, 1, :])
        )
        text_pseudo = self.text_adapter(text_features)
        text_pseudo = text_pseudo + self.text_modality_embedding.to(text_pseudo.dtype)[None, None, :]
        pseudo = torch.cat([text_pseudo[:, None], av_pseudo], dim=1)
        if presence is not None:
            pseudo = pseudo * presence[:, :, None, None].to(dtype=pseudo.dtype)
        return pseudo

    # 在伪 token 两侧拼接多模态标记与诊断提示，形成语言模型输入前缀。
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

    # 从骨干前向结果中提取最后一个位置的隐藏状态，供共享情感诊断头使用。
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
        output = self.llm.transformer(
            input_ids=self._dummy_input_ids(embeddings),
            input_fusion=embeddings,
            **common,
        )
        # 上游 ChatGLM 编码器采用序列优先布局：[S, B, H]。
        return output.last_hidden_state[-1, :, :]

    # 通过 input_fusion 向 ChatGLM3 传入融合嵌入，并显式关闭缓存。
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
        return self.llm(
            input_ids=self._dummy_input_ids(embeddings),
            input_fusion=embeddings,
            **common,
        )

    # 三模态分别使用伪 Token 诊断，共享冻结 LLM 和情感诊断头。
    def diagnostic_logits(
        self, pseudo: torch.Tensor, text_tensor: torch.Tensor,
        presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Diagnose text/A/V pseudo tokens independently using the same LLM/head."""
        batch_size = pseudo.shape[0]
        if presence is None:
            presence = torch.ones(batch_size, 3, device=pseudo.device)
        text_mask = presence[:, :1].long().expand(-1, self.pseudo_tokens)
        # Keep autograd through the frozen backbone to train the text adapter.
        text_sequence = self._wrapped_prefix(pseudo[:, 0], self.diagnostic_prompt_ids)
        prefix_size = self.bos_ids.numel() + self.wrapper_before_ids.numel()
        suffix_size = self.wrapper_after_ids.numel() + self.diagnostic_prompt_ids.numel()
        text_attention = torch.cat([
            text_mask.new_ones(batch_size, prefix_size), text_mask,
            text_mask.new_ones(batch_size, suffix_size),
        ], dim=1)
        text_sequence, text_attention, _ = compact_left_padding(text_sequence, text_attention)
        text_positions = (text_attention.cumsum(-1) - 1).clamp_min(0)
        text_hidden = self._backbone_last_hidden(text_sequence, text_attention, text_positions)
        text_logits = self.ordinal_head(text_hidden)

        flattened = pseudo[:, 1:].reshape(
            batch_size * 2, self.pseudo_tokens, self.hidden_size
        )
        sequence = self._wrapped_prefix(flattened, self.diagnostic_prompt_ids)
        attention_mask = torch.ones(
            sequence.shape[:2], device=sequence.device, dtype=torch.long
        )
        positions = attention_mask.cumsum(-1) - 1
        last_valid = self._backbone_last_hidden(sequence, attention_mask, positions)
        av_logits = self.ordinal_head(last_valid).reshape(batch_size, 2, 7)
        return torch.cat([text_logits[:, None, :], av_logits], dim=1)

    # 依次拼接 21 维概率、3 维两两冲突、3 维熵和 3 维存在掩码，组成 30 维输入。
    def _router_statistics(
        self, logits: torch.Tensor, presence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = F.softmax(
            logits.float(), dim=-1
        )
        present = presence.to(dtype=torch.bool)
        uniform = torch.full_like(probabilities, 1.0 / probabilities.shape[-1])
        probabilities = torch.where(present[:, :, None], probabilities, uniform)
        entropy = normalized_entropy(probabilities)
        # 冲突顺序固定为文本-音频、文本-视觉、音频-视觉，对应特征下标 21:24。
        pairs = ((0, 1), (0, 2), (1, 2))
        conflicts = []
        for first, second in pairs:
            pair_present = present[:, first] & present[:, second]
            divergence = normalized_wasserstein_distance(
                probabilities[:, first], probabilities[:, second], self.anchors
            )
            # 配对中任一模态缺失时冲突置零，避免把缺少信息当作模态矛盾。
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

    # 按消融配置将对应特征置零，保持输入维度和三模态存在掩码不变。
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

    # 截断诊断特征到路由输入的梯度，再计算权重；路由器自身仍可由生成损失更新。
    def route(
        self, logits: torch.Tensor, presence: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probabilities, conflicts, entropy, raw_features = self._router_statistics(
            logits, presence
        )
        # 只截断诊断统计到路由输入的梯度；音视频适配器仍通过各自损失路径更新。
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

    # 用“存在模态数 × 路由权重”缩放三模态槽，均匀权重时保持单位尺度。
    def gated_pseudo_tokens(
        self, pseudo: torch.Tensor, weights: torch.Tensor, presence: torch.Tensor
    ) -> torch.Tensor:
        present_count = presence.float().sum(dim=-1, keepdim=True)
        scales = present_count * weights
        gated = pseudo * scales[:, :, None, None].to(dtype=pseudo.dtype)
        return gated.reshape(pseudo.shape[0], 3 * self.pseudo_tokens, self.hidden_size)

    # 只对音视频伪 token 做幅度门控，并返回缺失模态掩码；原始文本保留独立路径。
    def gated_non_text_pseudo_tokens(
        self, pseudo: torch.Tensor, weights: torch.Tensor, presence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate audio/vision tokens while keeping natural text as a residual path.

        Multiplication by the number of present modalities preserves unit scale
        for a uniform router. The returned mask removes tokens belonging to a
        deliberately dropped modality from the final ChatGLM3 attention graph.
        """
        present_count = presence.float().sum(dim=-1, keepdim=True)
        # 乘存在模态数校正幅度：均匀路由下每个存在模态的缩放系数为一。
        scales = present_count * weights[:, 1:]
        gated = pseudo[:, 1:] * scales[:, :, None, None].to(dtype=pseudo.dtype)
        batch_size = pseudo.shape[0]
        gated = gated.reshape(
            batch_size, 2 * self.pseudo_tokens, self.hidden_size
        )
        token_mask = presence[:, 1:, None].expand(-1, -1, self.pseudo_tokens)
        return gated, token_mask.reshape(batch_size, -1).long()

    # 拼接门控音视频、原始有序文本和任务提示，再压紧内部填充空隙。
    def final_prefix(
        self,
        pseudo: torch.Tensor,
        weights: torch.Tensor,
        presence: torch.Tensor,
        text_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a compact prefix from gated A/V tokens and ordered raw text."""
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
        # 音视频先作为多模态前缀，原始文本按 token 顺序保留在任务提示之前。
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
        embeddings, attention_mask, _ = compact_left_padding(embeddings, attention_mask)
        return embeddings, attention_mask

    # 将情感标签格式化成分数字符串，仅在标签 token 上计算因果语言模型损失。
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
        # 前缀仅作条件输入，监督目标只覆盖情感分数字符串。
        targets = torch.cat([ignore, targets], dim=1)
        attention_mask = torch.cat([prefix_attention_mask, label_mask], dim=1)
        inputs, attention_mask, targets = compact_left_padding(inputs, attention_mask, targets)
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        output = self._causal_forward(
            inputs, attention_mask, position_ids, labels=targets
        )
        return output.loss

    # 联合前向：编码、诊断、路由、生成监督，再叠加存在模态的软序数辅助损失。
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
        logits = self.diagnostic_logits(pseudo, text[0], presence)
        weights, diagnostics = self.route(logits, presence)
        prefix, prefix_attention_mask = self.final_prefix(
            pseudo, weights, presence, text[0]
        )
        generation_loss = self._teacher_forcing_loss(
            prefix, prefix_attention_mask, labels.reshape(-1)
        )

        # 三模态共享同一软序数目标；辅助损失只统计实际存在的模态。
        ordinal_targets = soft_ordinal_targets(labels, self.anchors)
        per_modality = soft_cross_entropy(
            logits.float(),
            ordinal_targets[:, None, :].expand(-1, 3, -1),
        )
        auxiliary_loss = (
            per_modality * presence
        ).sum() / presence.sum().clamp_min(1.0)
        loss = generation_loss
        if stage == "stage1":
            loss = loss + self.aux_weight * auxiliary_loss
        else:
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

    # 解析生成文本中的首个数值；无法解析或超出 [-1, 1] 时记为零并登记诊断索引。
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

    # 无梯度地返回三模态 logits、权重、概率、冲突和熵，供检查模型行为。
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
        logits = self.diagnostic_logits(pseudo, text[0], presence)
        weights, diagnostics = self.route(logits, presence)
        return {"logits": logits, "weights": weights, **diagnostics}

    # 逐 token 贪心生成情感分数；可在评估时覆盖路由权重并返回诊断信息。
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

    # 按阶段切换可训练参数：联合训练开放适配器、诊断头和路由器，骨干始终冻结。
    def set_stage(self, stage: str) -> None:
        if stage not in {"stage1", "eval"}:
            raise ValueError(f"unknown stage: {stage}")
        for name, parameter in self.named_parameters():
            if name.startswith("llm."):
                parameter.requires_grad = False
            elif stage == "stage1":
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    # 收集音视频编码器、适配器及模态嵌入，供优化器使用独立学习率。
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
        parameters.append(self.text_modality_embedding)
        return parameters

    # 仅导出实验模块的参数与缓冲区，排除冻结语言模型的大体积权重。
    def experiment_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only experiment parameters/buffers, never frozen ChatGLM3 weights."""
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith("llm.")
        }

    # 先检查 V4 架构版本，再加载实验参数，并拒绝多余键或非骨干参数缺失。
    def load_experiment_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        version = state.get("architecture_version")
        if version is None or int(version.item()) != ARCHITECTURE_VERSION:
            raise RuntimeError("Pseudo-text V5 requires a matching checkpoint; raw-text/JS Router weights are incompatible")
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

    # 校验评估用固定权重：每行和为一、非负有限，且缺失模态权重必须为零。
    @staticmethod
    def validate_weight_override(weights: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        """Fixed routing interventions for evaluation, never training."""
        weights = torch.as_tensor(weights, device=presence.device, dtype=torch.float32)
        if weights.shape != presence.shape or not bool(torch.isfinite(weights).all()):
            raise ValueError("override weights must be finite and shaped [B,3]")
        if bool((weights < 0).any()) or bool((weights[~presence.bool()] != 0).any()):
            raise ValueError("override weights must be nonnegative and exclude absent modalities")
        if not torch.allclose(weights.sum(-1), torch.ones_like(presence[:, 0]), atol=1e-6):
            raise ValueError("override weights must sum to one per sample")
        return weights

    # 按当前实验入口加载对应骨干和分词器，并适配其特殊 token 与输入接口。
    def _load_backbone(self, args: Any) -> tuple[Any, nn.Module]:
        model_path = Path(args.pretrain_LM).resolve()
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

    # 为需要 token ID 形状的 ChatGLM 接口构造占位输入，真实内容由融合嵌入提供。
    def _dummy_input_ids(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.full(
            embeddings.shape[:2],
            int(self.tokenizer.pad_token_id),
            device=embeddings.device,
            dtype=torch.long,
        )
