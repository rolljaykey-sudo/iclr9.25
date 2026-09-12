# 对外导出路由器使用的概率、熵、冲突距离和软标签工具。
"""Conflict- and uncertainty-aware multimodal routing for ChatGLM3."""

from .math_utils import (
    masked_softmax,
    normalized_entropy,
    normalized_js_divergence,
    normalized_wasserstein_distance,
    soft_ordinal_targets,
)

__all__ = [
    "masked_softmax",
    "normalized_entropy",
    "normalized_js_divergence",
    "normalized_wasserstein_distance",
    "soft_ordinal_targets",
]
