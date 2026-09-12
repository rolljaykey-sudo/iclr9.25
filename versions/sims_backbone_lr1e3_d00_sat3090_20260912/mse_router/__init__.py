"""Conflict- and uncertainty-aware multimodal routing for Qwen/MOSEI."""

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
