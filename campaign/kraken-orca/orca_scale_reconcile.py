# SPDX-License-Identifier: Apache-2.0
"""Bounded gate/up global-scale reconciliation for fused NVFP4 MoE kernels.

Pure torch on purpose: the math is unit-testable on CPU without vLLM or GPUs.

compressed-tensors NVFP4 dequantization is
``weight = fp4_code * block_scale_e4m3 / weight_global_scale``.
Fused w13 kernels (Marlin, B12X) accept one global scale per expert, while this
checkpoint stores independent gate and up divisors for 8,291 of 12,096 expert
pairs. Stock vLLM silently keeps the gate divisor; that mis-scales the up half
by up to 9.96x.

The only overflow-free unification is the smaller divisor: the larger-divisor
half's E4M3 block scales are multiplied by ``unified / old < 1`` and requantized.
FP4 payloads are never touched, so each weight element's relative error equals
its block scale's rounding error (measured worst 5.70%, mean 2.21% over rounded
blocks on this checkpoint). Exact preservation is impossible in E4M3 for
non-power-of-two ratios; callers must treat this as an approximation and measure
the fidelity delta against an exact per-projection reference.
"""

from __future__ import annotations

import torch

E4M3_MAX = 448.0


class ScaleReconciliationError(RuntimeError):
    """Raised when reconciliation cannot stay inside the E4M3 block-scale range."""


def reconcile_gate_up_scales(
    w13_weight_scale: torch.Tensor,
    w13_weight_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return reconciled block scales, a unified [E, 2] global scale and stats.

    ``w13_weight_scale`` is ``[E, 2 * N, K / 16]`` E4M3 block scales in fused
    gate/up row order; ``w13_weight_global_scale`` is ``[E, 2]`` positive FP32
    divisors. Matching pairs are returned untouched.
    """
    if w13_weight_global_scale.dtype != torch.float32:
        raise ScaleReconciliationError("global scales must be float32")
    if not torch.all(
        torch.isfinite(w13_weight_global_scale) & (w13_weight_global_scale > 0)
    ):
        raise ScaleReconciliationError("global scales must be finite and positive")
    gate = w13_weight_global_scale[:, 0]
    up = w13_weight_global_scale[:, 1]
    unified = torch.minimum(gate, up)
    stats = {
        "mismatched_experts": float((gate != up).sum()),
        "max_shrink_factor": 1.0,
        "max_block_scale_after": 0.0,
    }
    if float(stats["mismatched_experts"]) == 0.0:
        return w13_weight_scale, w13_weight_global_scale, stats

    rows = w13_weight_scale.shape[1]
    if rows % 2:
        raise ScaleReconciliationError("fused w13 block scales must have even rows")
    half = rows // 2
    reconciled = w13_weight_scale.clone().float()
    for column, old in ((0, gate), (1, up)):
        factor = torch.where(old > unified, unified / old, torch.ones_like(old))
        stats["max_shrink_factor"] = max(
            stats["max_shrink_factor"], 1.0 / float(factor.min())
        )
        slice_view = reconciled[:, column * half : (column + 1) * half]
        scaled = slice_view * factor[:, None, None]
        if float(scaled.abs().max()) > E4M3_MAX:
            raise ScaleReconciliationError(
                "reconciled block scales exceed the E4M3 range; refusing to saturate"
            )
        requantized = scaled.to(w13_weight_scale.dtype).float()
        if bool(((requantized == 0) & (slice_view != 0)).any()):
            raise ScaleReconciliationError(
                "reconciliation would round nonzero block scales to zero; refusing"
            )
        slice_view.copy_(requantized)
    stats["max_block_scale_after"] = float(reconciled.abs().max())
    if stats["max_block_scale_after"] > E4M3_MAX:
        raise ScaleReconciliationError("reconciled block scales exceed the E4M3 range")
    unified_pair = torch.stack((unified, unified), dim=1).contiguous()
    return reconciled.to(w13_weight_scale.dtype), unified_pair, stats
