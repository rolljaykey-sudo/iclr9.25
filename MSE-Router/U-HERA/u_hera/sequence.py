"""Ordered prefix/target compaction, adapted from the audited V4 Router."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compact_left_padding(embeddings, attention_mask, labels=None):
    if embeddings.ndim != 3 or attention_mask.shape != embeddings.shape[:2]:
        raise ValueError("expected embeddings [B,S,H] and mask [B,S]")
    if labels is not None and labels.shape != attention_mask.shape:
        raise ValueError("labels must match mask")
    valid = attention_mask.bool()
    lengths = valid.sum(-1)
    if bool((lengths == 0).any()):
        raise ValueError("empty complete language-model sequence")
    width = embeddings.shape[1]
    positions = torch.arange(width, device=embeddings.device).expand_as(valid)
    order = (positions + valid.long() * width).argsort(dim=-1)
    order = order[:, -int(lengths.max().item()):]
    mask = valid.gather(1, order)
    packed = embeddings.gather(1, order[..., None].expand(-1, -1, embeddings.shape[-1]))
    packed = packed.masked_fill(~mask[..., None], 0)
    targets = None if labels is None else labels.gather(1, order).masked_fill(~mask, -100)
    return packed, mask.long(), targets


def per_sample_causal_loss(logits, targets):
    """Mean CE of supervised next tokens, retaining equal sample weights."""
    shifted = targets[:, 1:]
    valid = shifted != -100
    count = valid.sum(-1)
    if bool((count == 0).any()):
        raise ValueError("each teacher-forcing sample requires a target token")
    # Select first: do not allocate an FP32 copy of all prefix/vocabulary logits.
    values = F.cross_entropy(logits[:, :-1][valid].float(), shifted[valid], reduction="none")
    rows = torch.arange(logits.shape[0], device=logits.device)[:, None].expand_as(valid)[valid]
    sums = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
    return sums.scatter_add(0, rows, values) / count


def masked_softmax(scores, mask, dim=-1):
    """FP32 probabilities; an entirely masked row returns exact zeros."""
    mask = mask.bool().expand_as(scores)
    active = mask.any(dim=dim, keepdim=True)
    scores = scores.float().masked_fill(~mask, -torch.inf)
    scores = torch.where(active, scores, torch.zeros_like(scores))
    return torch.softmax(scores, dim=dim).masked_fill(~mask, 0)
