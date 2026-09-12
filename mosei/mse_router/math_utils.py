# 数学工具：连续标签插值、归一化熵、Wasserstein-1 冲突及缺失模态掩码。
"""Small, dependency-light mathematical helpers used by the router."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


ANCHORS = torch.arange(-3.0, 4.0)


# 将连续标签夹到锚点范围内，再把概率质量线性分配给相邻两个情感锚点。
def soft_ordinal_targets(
    labels: torch.Tensor, anchors: torch.Tensor | None = None
) -> torch.Tensor:
    """Linearly interpolate a continuous target between adjacent ordinal anchors."""
    if anchors is None:
        anchors = ANCHORS.to(device=labels.device, dtype=labels.dtype)
    if anchors.ndim != 1 or anchors.numel() < 2:
        raise ValueError("anchors must be a one-dimensional tensor with >= 2 entries")
    if not bool(torch.all(anchors[1:] > anchors[:-1])):
        raise ValueError("anchors must be strictly increasing")

    values = labels.reshape(-1).to(dtype=anchors.dtype)
    values = values.clamp(float(anchors[0]), float(anchors[-1]))
    # 查找相邻锚点；边界标签随后合并到同一端点，避免向范围外分配质量。
    upper = torch.searchsorted(anchors, values, right=False)
    upper = upper.clamp(1, anchors.numel() - 1)
    lower = upper - 1
    exact_low = values <= anchors[0]
    exact_high = values >= anchors[-1]
    lower = torch.where(exact_low, torch.zeros_like(lower), lower)
    upper = torch.where(exact_low, torch.zeros_like(upper), upper)
    top_index = torch.full_like(lower, anchors.numel() - 1)
    lower = torch.where(exact_high, top_index, lower)
    upper = torch.where(exact_high, top_index, upper)

    lower_anchor = anchors[lower]
    upper_anchor = anchors[upper]
    denominator = (upper_anchor - lower_anchor).clamp_min(
        torch.finfo(anchors.dtype).eps
    )
    upper_weight = torch.where(
        lower == upper,
        torch.zeros_like(values),
        (values - lower_anchor) / denominator,
    )
    result = torch.zeros(
        values.shape[0], anchors.numel(), device=values.device, dtype=values.dtype
    )
    # 两侧权重相加为一，且分布期望对应裁剪后的连续标签。
    result.scatter_add_(1, lower[:, None], (1.0 - upper_weight)[:, None])
    result.scatter_add_(1, upper[:, None], upper_weight[:, None])
    return result


# 计算软标签交叉熵并保留样本维度，交由调用方按模态是否存在进行归约。
def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"shape mismatch: logits={logits.shape}, targets={targets.shape}")
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)


# 用类别数的对数归一化熵，衡量单个模态预测分布的不确定性。
def normalized_entropy(probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy normalized to [0, 1] for a categorical distribution."""
    count = probabilities.shape[-1]
    if count < 2:
        raise ValueError("entropy requires at least two classes")
    p = probabilities.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1) / math.log(count)


# 保留归一化 JS 散度供参考和测试；V4 实际路由冲突使用 Wasserstein-1。
def normalized_js_divergence(
    first: torch.Tensor, second: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Jensen-Shannon divergence normalized by log(2), hence bounded by one."""
    if first.shape != second.shape:
        raise ValueError(f"shape mismatch: first={first.shape}, second={second.shape}")
    p = first.clamp_min(eps)
    q = second.clamp_min(eps)
    midpoint = 0.5 * (p + q)
    kl_pm = (p * (p.log() - midpoint.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - midpoint.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm) / math.log(2.0)


# 按相邻锚点间距积分两分布的 CDF 差，再除以锚点跨度，得到 [0, 1] 冲突值。
def normalized_wasserstein_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    anchors: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact 1D Wasserstein-1 on shared ordered anchors, divided by their span.

    For normalized masses p/q at a[0] < ... < a[K-1], integrate their CDF gap:
        sum_k abs(cumsum(p-q)[k]) * (a[k+1]-a[k]) / (a[-1]-a[0]).
    The sum excludes the final CDF entry. This is an exact finite-support
    formula, not Sinkhorn, a divergence on probabilities-as-samples, or merely
    the difference between predicted means. Arbitrary batch dimensions and
    nonuniform anchor spacing are supported. Half precision accumulates in
    float32; float64 is preserved for reference comparisons and gradchecks.

    Inputs are nonnegative finite masses with positive totals, normalized here
    as in scipy.stats.wasserstein_distance(..., u_weights=p, v_weights=q).
    Zero probabilities stay exactly zero. Result lies in [0,1].
    Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    """
    if first.shape != second.shape or first.ndim < 1 or first.shape[-1] < 2:
        raise ValueError("masses must have equal shape [..., K] with K >= 2")
    if first.device != second.device:
        raise ValueError("both distributions must be on the same device")
    if not first.is_floating_point() or not second.is_floating_point():
        raise ValueError("probability masses must be floating point")
    # 统一计算精度：FP16/BF16 至少提升到 FP32，FP64 对照与梯度检查保留双精度。
    dtype = torch.promote_types(first.dtype, second.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    p, q = first.to(dtype=dtype), second.to(dtype=dtype)
    a = torch.as_tensor(ANCHORS if anchors is None else anchors, device=p.device, dtype=dtype)
    if a.ndim != 1 or a.numel() != p.shape[-1]:
        raise ValueError("anchors must match the final probability dimension")
    if not bool(torch.isfinite(a).all()) or not bool((a[1:] > a[:-1]).all()):
        raise ValueError("anchors must be finite and strictly increasing")
    p_total, q_total = p.sum(-1, keepdim=True), q.sum(-1, keepdim=True)
    for masses, total in ((p, p_total), (q, q_total)):
        if not bool(torch.isfinite(masses).all()) or bool((masses < 0).any()):
            raise ValueError("probability masses must be finite and nonnegative")
        if not bool(torch.isfinite(total).all()) or bool((total <= 0).any()):
            raise ValueError("each distribution must have positive finite total mass")
    # 将非负质量归一化；零概率仍保持精确零，不人为加入平滑质量。
    p, q = p / p_total, q / q_total
    # 最后一个 CDF 点之后没有积分区间，因此只保留前 K-1 项。
    cdf_gap = (p - q).cumsum(dim=-1)[..., :-1].abs()
    widths = a[1:] - a[:-1]
    span = a[-1] - a[0]
    if not bool(torch.isfinite(span)) or not bool(torch.isfinite(widths).all()):
        raise ValueError("anchor span must be finite")
    # 先归一化区间宽度再求和，避免先乘上过大的锚点跨度。
    return (cdf_gap * (widths / span)).sum(dim=-1).clamp(0.0, 1.0)


# 仅在存在的模态之间分配权重；拒绝所有模态都缺失的样本。
def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over present modalities with exact zeros for absent modalities."""
    if logits.shape != mask.shape:
        raise ValueError(f"shape mismatch: logits={logits.shape}, mask={mask.shape}")
    present = mask.to(dtype=torch.bool)
    if bool((present.sum(dim=-1) == 0).any()):
        raise ValueError("each sample must contain at least one modality")
    masked_logits = logits.masked_fill(~present, torch.finfo(logits.dtype).min)
    weights = F.softmax(masked_logits, dim=-1)
    weights = weights * present.to(weights.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).eps
    )
