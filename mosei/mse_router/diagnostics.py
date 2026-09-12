# 诊断工具：固定输入并干预路由权重，观察生成结果对门控变化的局部敏感性。
"""Optional clean-input interventions; these do not run during training/testing."""

from __future__ import annotations

from typing import Any

import torch


# 保持完整输入不变，仅改变权重，比较生成数值与首 token 分布相对均匀权重的变化。
@torch.no_grad()
def compare_gate_weights(model: Any, text: tuple, audio: tuple, vision: tuple) -> dict:
    """Hold all inputs fixed and change only routing weights on complete samples.

    Report generated values and first-token distribution changes. This measures
    local output sensitivity, not whether a gate learned calibrated reliability.
    Use a V4 conflict checkpoint in eval mode and a Slurm allocation for the real LLM.
    """
    if model.training:
        raise ValueError("call model.eval() before comparing weights")
    presence = torch.ones(text[0].shape[0], 3, device=text[0].device)
    pseudo = model.encode_modalities(text, audio, vision, presence)
    settings = {
        "uniform": [1 / 3, 1 / 3, 1 / 3],
        "audio_low": [2 / 3 - 0.01, 0.01, 1 / 3],
        "vision_low": [2 / 3 - 0.01, 1 / 3, 0.01],
    }
    baseline = None
    result = {}
    for name, row in settings.items():
        weights = presence.new_tensor(row).expand_as(presence)
        prefix, mask = model.final_prefix(pseudo, weights, presence, text[0])
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        output = model._causal_forward(prefix, mask, positions)
        probabilities = output.logits[:, -1].float().softmax(-1)
        if baseline is None:
            baseline = probabilities
        values = model.generate(text, audio, vision, weights_override=weights)
        result[name] = {
            "weights": row,
            "generated_values": values,
            "first_token_total_variation_from_uniform": (
                0.5 * (probabilities - baseline).abs().sum(-1)
            ).cpu().tolist(),
        }
    return result
