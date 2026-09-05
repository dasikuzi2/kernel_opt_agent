#!/usr/bin/env python3
"""Screen Marlin FP32 versus native-output cross-CTA reduction by shape."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
from vllm.scalar_type import scalar_types


SHAPES = (
    {"name": "linear_attention_qkvz", "n": 8192, "k": 1024, "calls": 18},
    {"name": "attention_out", "n": 1024, "k": 2048, "calls": 24},
    {"name": "full_attention_qkv", "n": 5120, "k": 1024, "calls": 6},
    {"name": "mlp_gate_up", "n": 7168, "k": 1024, "calls": 24},
    {"name": "mlp_down", "n": 1024, "k": 3584, "calls": 24},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure_us(fn, iterations: int, repeats: int, warmups: int) -> tuple[float, list[float]]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples), samples


def benchmark(shape: dict, args: argparse.Namespace) -> dict:
    n, k = shape["n"], shape["k"]
    torch.manual_seed(args.seed + n + k)
    x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16) * 0.02
    weight = torch.randn((k, n), device="cuda", dtype=torch.bfloat16) * 0.02
    _, packed, scales, g_idx, sort_indices, _ = marlin_quantize(
        weight, scalar_types.uint4b8, args.group_size, False
    )
    workspace = marlin_make_workspace_new(torch.device("cuda", 0))
    empty = torch.empty(0, device="cuda", dtype=torch.int32)

    def run(use_fp32_reduce: bool):
        return apply_gptq_marlin_linear(
            x,
            packed,
            scales,
            empty,
            g_idx,
            sort_indices,
            workspace,
            scalar_types.uint4b8,
            output_size_per_partition=n,
            input_size_per_partition=k,
            is_k_full=True,
            input_dtype=torch.bfloat16,
            use_fp32_reduce=use_fp32_reduce,
        )

    true_us, true_samples = measure_us(
        lambda: run(True), args.iterations, args.repeats, args.warmups
    )
    false_us, false_samples = measure_us(
        lambda: run(False), args.iterations, args.repeats, args.warmups
    )
    true_out, false_out = run(True), run(False)
    torch.cuda.synchronize()
    return {
        **shape,
        "fp32_reduce_median_us": true_us,
        "native_reduce_median_us": false_us,
        "native_reduce_speedup": true_us / false_us,
        "weighted_fp32_reduce_us": true_us * shape["calls"],
        "weighted_native_reduce_us": false_us * shape["calls"],
        "fp32_reduce_samples_us": true_samples,
        "native_reduce_samples_us": false_samples,
        "numerical_delta": {
            "exact_elements": int((true_out == false_out).sum().item()),
            "elements": true_out.numel(),
            "max_abs": float((true_out.float() - false_out.float()).abs().max().item()),
            "mean_abs": float((true_out.float() - false_out.float()).abs().mean().item()),
            "argmax_equal": bool(true_out.argmax().item() == false_out.argmax().item()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    collapsed: dict[tuple[int, int], int] = {}
    for row in inventory["projection_groups"]:
        _, n, k = map(int, row["decode_shape_m_n_k"])
        collapsed[(n, k)] = collapsed.get((n, k), 0) + int(row["calls_per_decode"])
    if {(r["n"], r["k"]): r["calls"] for r in SHAPES} != collapsed:
        raise RuntimeError("shape inventory changed")

    results = [benchmark(shape, args) for shape in SHAPES]
    control = sum(row["weighted_fp32_reduce_us"] for row in results)
    candidate = sum(row["weighted_native_reduce_us"] for row in results)
    payload = {
        "schema_version": "marlin-reduction-shape-screen-v1",
        "status": "PASS",
        "claim_scope": "ATOMIC_SCREEN_ONLY_REQUIRES_END_TO_END_CAUSAL_VALIDATION",
        "device": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        },
        "controls": {
            "m": 1,
            "group_size": args.group_size,
            "activation_dtype": "bfloat16",
            "iterations": args.iterations,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "inventory": {"path": str(args.inventory), "sha256": digest(args.inventory)},
        },
        "results": results,
        "projection_weighted_summary": {
            "fp32_reduce_us_per_decode": control,
            "native_reduce_us_per_decode": candidate,
            "native_reduce_speedup": control / candidate,
            "winning_shapes": [row["name"] for row in results if row["native_reduce_speedup"] > 1.0],
        },
        "warnings": [
            "Hot-cache random-tensor timing is a screening signal, not a whole-model claim.",
            "Disabling FP32 reduction changes accumulation precision and must pass exact greedy token identity before promotion.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
