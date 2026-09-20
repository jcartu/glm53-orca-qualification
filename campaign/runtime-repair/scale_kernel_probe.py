#!/usr/bin/env python3
"""Exercise real Marlin GEMMs, per-expert up scales, clipping and graph replay."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (
    _activation_with_up_scale_correction,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_nvfp4_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.scalar_type import scalar_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("Refusing to overwrite numerical evidence")
    torch.manual_seed(20260919)
    device = "cuda:0"
    dtype = torch.bfloat16
    experts, hidden, intermediate, tokens, topk = 3, 512, 128, 8, 2
    raw13 = torch.randint(
        0,
        256,
        (experts, intermediate * 2, hidden // 2),
        device=device,
        dtype=torch.uint8,
    )
    raw2 = torch.randint(
        0, 256, (experts, hidden, intermediate // 2), device=device, dtype=torch.uint8
    )
    sf13 = torch.full(
        (experts, intermediate * 2, hidden // 16),
        128.0,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    sf2 = torch.full(
        (experts, hidden, intermediate // 16),
        64.0,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    gate_global = torch.tensor([21504.0, 21504.0, 16384.0], device=device)
    up_global = torch.tensor([21504.0, 26496.0, 32768.0], device=device)
    down_global = torch.tensor([16384.0, 16384.0, 16384.0], device=device)
    ratio = gate_global / up_global
    gate = dequantize_to_dtype(
        raw13[:, :intermediate].contiguous(),
        sf13[:, :intermediate].contiguous(),
        1 / gate_global,
        dtype,
        swizzle=False,
    )
    up = dequantize_to_dtype(
        raw13[:, intermediate:].contiguous(),
        sf13[:, intermediate:].contiguous(),
        1 / up_global,
        dtype,
        swizzle=False,
    )
    down = dequantize_to_dtype(raw2, sf2, 1 / down_global, dtype, swizzle=False)
    layer = SimpleNamespace(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        params_dtype=dtype,
    )
    w13, s13, g13, w2, s2, g2 = prepare_nvfp4_moe_layer_for_marlin(
        layer,
        raw13.clone(),
        sf13.clone(),
        1 / gate_global,
        raw2.clone(),
        sf2.clone(),
        1 / down_global,
        True,
    )
    ids_list = [[i % experts, (i + 1) % experts] for i in range(tokens)]
    ids = torch.tensor(ids_list, device=device, dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]] * tokens, device=device, dtype=torch.float32)
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    x[::2].mul_(40)
    config = ApplyMoEActivationConfig(clamp_limit=10.0)

    def activation(kind, out, inputs, **kwargs):
        apply_moe_activation(kind, out, inputs, activation_config=config, **kwargs)

    corrected = _activation_with_up_scale_correction(activation, ratio)

    def execute(values, correct):
        return fused_marlin_moe(
            values,
            w13,
            w2,
            None,
            None,
            s13,
            s2,
            weights,
            ids,
            scalar_types.float4_e2m1f.id,
            global_num_experts=experts,
            activation=MoEActivation.SILU,
            activation_func=corrected if correct else activation,
            global_scale1=g13,
            global_scale2=g2,
            workspace=layer.workspace,
        )

    def reference(values):
        rows = []
        for token, choices in enumerate(ids_list):
            total = torch.zeros(hidden, device=device, dtype=torch.float32)
            for position, expert in enumerate(choices):
                a = (
                    F.linear(values[token : token + 1], gate[expert])
                    .float()
                    .clamp(max=10)
                )
                b = (
                    F.linear(values[token : token + 1], up[expert])
                    .float()
                    .clamp(-10, 10)
                )
                activated = (F.silu(a) * b).to(dtype)
                contribution = F.linear(activated, down[expert]).float().squeeze(0)
                total += contribution * weights[token, position]
            rows.append(total.to(dtype))
        return torch.stack(rows)

    def metrics(actual, expected):
        delta = actual.float() - expected.float()
        return {
            "relative_rmse": float(
                torch.linalg.vector_norm(delta)
                / torch.linalg.vector_norm(expected.float()).clamp_min(1e-8)
            ),
            "max_abs_error": float(delta.abs().max()),
            "finite": bool(torch.isfinite(actual).all()),
        }

    with torch.inference_mode():
        expected = reference(x)
        broken = execute(x, False)
        fixed = execute(x, True)
        torch.cuda.synchronize()
        broken_metrics = metrics(broken, expected)
        fixed_metrics = metrics(fixed, expected)
        if not fixed_metrics["finite"] or fixed_metrics["relative_rmse"] > 0.015:
            raise AssertionError({"corrected_kernel_failed": fixed_metrics})
        if (
            broken_metrics["relative_rmse"] < 0.05
            or fixed_metrics["relative_rmse"] >= broken_metrics["relative_rmse"] * 0.2
        ):
            raise AssertionError(
                {
                    "negative_control_not_discriminating": broken_metrics,
                    "fixed": fixed_metrics,
                }
            )
        # Actual CUDA graph replay must use new input values, not a stale output.
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            execute(x, True)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_output = execute(x, True)
        x.mul_(0.7)
        graph.replay()
        torch.cuda.synchronize()
        graph_metrics = metrics(graph_output, reference(x))
        if not graph_metrics["finite"] or graph_metrics["relative_rmse"] > 0.015:
            raise AssertionError({"graph_replay_failed": graph_metrics})
    result = {
        "schema": "orca-marlin-scale-correction-cuda.v1",
        "passed": True,
        "device": torch.cuda.get_device_name(0),
        "shape": {
            "experts": experts,
            "hidden": hidden,
            "intermediate": intermediate,
            "tokens": tokens,
            "topk": topk,
        },
        "unmodified_collapsed_scale_negative_control": broken_metrics,
        "per_expert_up_correction": fixed_metrics,
        "cuda_graph_changed_input_replay": graph_metrics,
        "reference": "Separate NVFP4 gate/up/down dequantization with their own stored global divisors, BF16 linear operations and FP32 gated/clipped activation.",
        "limit": 10.0,
        "source_weights_requantized": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
