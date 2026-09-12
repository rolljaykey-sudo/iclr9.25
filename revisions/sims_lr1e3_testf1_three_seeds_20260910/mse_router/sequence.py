# 序列整理：保持有效 token 的原始顺序，将填充集中到左侧，并同步对齐监督标签。
"""Keep valid tokens in order, with padding only before the complete sequence."""

from __future__ import annotations

import torch


# 移除有效 token 之间的掩码空隙，保留顺序及梯度，并同步搬移因果监督标签。
def compact_left_padding(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Remove internal mask gaps and align optional causal targets with tokens.

    Qwen-1.8B's bundled RoPE ignores explicit position IDs. Contiguous valid
    sequences give it correct relative positions without editing the LLM.
    Gathering embeddings keeps gradients to adapters and Router weights.
    """
    if embeddings.ndim != 3 or attention_mask.shape != embeddings.shape[:2]:
        raise ValueError("expected embeddings [B,S,H] and mask [B,S]")
    if labels is not None and labels.shape != attention_mask.shape:
        raise ValueError("labels must have the same shape as the mask")
    valid = attention_mask.bool()
    lengths = valid.sum(-1)
    if bool((lengths == 0).any()):
        raise ValueError("each sequence must contain at least one valid token")
    width = embeddings.shape[1]
    positions = torch.arange(width, device=embeddings.device).expand_as(valid)
    # 给有效位置增加统一偏移：填充排到左边，各有效 token 之间仍保持原顺序。
    order = (positions + valid.long() * width).argsort(dim=-1)
    order = order[:, -int(lengths.max().item()):]
    mask = valid.gather(1, order)
    packed = embeddings.gather(1, order[..., None].expand(-1, -1, embeddings.shape[-1]))
    packed = packed.masked_fill(~mask[..., None], 0)
    # 标签按同一索引移动，填充位置设为 -100，使因果损失忽略这些位置。
    targets = None if labels is None else labels.gather(1, order).masked_fill(~mask, -100)
    return packed, mask.long(), targets
