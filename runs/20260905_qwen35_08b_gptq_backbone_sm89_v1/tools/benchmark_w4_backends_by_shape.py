#!/usr/bin/env python3
"""Screen Marlin and Humming on the exact Qwen3.5 GPTQ projection shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.forward import humming_forward
from humming.testing.runner import KernelTestCase, KernelTestRunner
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types


SHAPES = (
    {"name": "linear_attention_qkvz", "m": 1, "n": 8192, "k": 1024, "calls": 18},
    {"name": "attention_out", "m": 1, "n": 1024, "k": 2048, "calls": 24},
    {"name": "full_attention_qkv", "m": 1, "n": 5120, "k": 1024, "calls": 6},
    {"name": "mlp_gate_up", "m": 1, "n": 7168, "k": 1024, "calls": 24},
    {"name": "mlp_down", "m": 1, "n": 1024, "k": 3584, "calls": 24},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure_us(fn, *, iterations: int, repeats: int, warmups: int) -> tuple[float, list[float]]:
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


def benchmark_shape(shape: dict, args: argparse.Namespace) -> dict:
    m, n, k = shape["m"], shape["n"], shape["k"]
    torch.manual_seed(args.seed + n + k)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * 0.02
    dense_weight = torch.randn((k, n), dtype=torch.bfloat16, device="cuda") * 0.02

    _, marlin_weight, marlin_scales, g_idx, sort_indices, _ = marlin_quantize(
        dense_weight,
        scalar_types.uint4b8,
        args.group_size,
        False,
    )
    workspace = marlin_make_workspace_new(torch.device("cuda", 0))
    zero_points = torch.empty(0, dtype=torch.int32, device="cuda")

    def run_marlin():
        return apply_gptq_marlin_linear(
            x,
            marlin_weight,
            marlin_scales,
            zero_points,
            g_idx,
            sort_indices,
            workspace,
            scalar_types.uint4b8,
            output_size_per_partition=n,
            input_size_per_partition=k,
            is_k_full=True,
            input_dtype=torch.bfloat16,
        )

    marlin_us, marlin_samples = measure_us(
        run_marlin,
        iterations=args.iterations,
        repeats=args.repeats,
        warmups=args.warmups,
    )
    marlin_output = run_marlin()

    layer_config = LayerConfig(
        shape_n=n,
        shape_k=k,
        b_dtype=dtypes.uint4,
        a_dtype=dtypes.bfloat16,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=args.group_size,
    )
    compute_config = ComputeConfig(gemm_type=GemmType.DENSE)
    runner = KernelTestRunner(
        KernelTestCase(
            name=shape["name"],
            layer_config=layer_config,
            compute_config=compute_config,
            seed=args.seed + n + k,
        )
    )
    # Compile the same M=1 heuristic used by Humming's production forward path
    # and verify its output before timing it.
    runner.run([m])
    tensors = runner.kernel_tensors
    locks = torch.zeros(1024, dtype=torch.int32, device="cuda")

    def run_humming():
        return humming_forward(
            layer_config,
            inputs=x,
            weight=tensors["weight"],
            weight_scale=tensors.get("weight_scale"),
            zero_point=tensors.get("zero_point"),
            weight_scale_2=tensors.get("weight_scale_2"),
            bias=tensors.get("bias"),
            locks=locks,
            compute_config=compute_config.to_str(),
        )

    humming_us, humming_samples = measure_us(
        run_humming,
        iterations=args.iterations,
        repeats=args.repeats,
        warmups=args.warmups,
    )
    humming_output = run_humming()
    torch.cuda.synchronize()
    record = {
        **shape,
        "marlin_median_us": marlin_us,
        "humming_median_us": humming_us,
        "humming_speedup_vs_marlin": marlin_us / humming_us,
        "weighted_decode_us_marlin": marlin_us * shape["calls"],
        "weighted_decode_us_humming": humming_us * shape["calls"],
        "marlin_samples_us": marlin_samples,
        "humming_samples_us": humming_samples,
        "numerical_sanity": {
            "both_finite": bool(torch.isfinite(marlin_output).all() and torch.isfinite(humming_output).all()),
            "marlin_shape": list(marlin_output.shape),
            "humming_shape": list(humming_output.shape),
        },
    }
    del (
        dense_weight,
        marlin_weight,
        marlin_scales,
        g_idx,
        sort_indices,
        workspace,
        runner,
        tensors,
    )
    torch.cuda.empty_cache()
    return record


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
    if not args.inventory.is_file():
        raise FileNotFoundError(args.inventory)

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    observed = {(row["n"], row["k"], row["calls"]) for row in SHAPES}
    # Two logical out-projection groups share [N=1024,K=2048], so collapse
    # their 18+6 calls before comparing with the five unique benchmark shapes.
    collapsed_expected: dict[tuple[int, int], int] = {}
    for row in inventory["projection_groups"]:
        _, n, k = map(int, row["decode_shape_m_n_k"])
        calls = int(row["calls_per_decode"])
        collapsed_expected[(n, k)] = collapsed_expected.get((n, k), 0) + calls
    if observed != {(n, k, calls) for (n, k), calls in collapsed_expected.items()}:
        raise RuntimeError("hard-coded shape screen no longer matches the checkpoint inventory")

    results = [benchmark_shape(shape, args) for shape in SHAPES]
    best_per_shape_us = sum(
        min(row["weighted_decode_us_marlin"], row["weighted_decode_us_humming"])
        for row in results
    )
    marlin_total_us = sum(row["weighted_decode_us_marlin"] for row in results)
    humming_total_us = sum(row["weighted_decode_us_humming"] for row in results)
    payload = {
        "schema_version": "w4a16-backend-shape-screen-v1",
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
            "weight_type": "uint4b8/symmetric-groupwise",
            "iterations": args.iterations,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "inventory": {"path": str(args.inventory), "sha256": digest(args.inventory)},
        },
        "results": results,
        "projection_weighted_summary": {
            "marlin_us_per_decode": marlin_total_us,
            "humming_us_per_decode": humming_total_us,
            "oracle_per_shape_us_per_decode": best_per_shape_us,
            "oracle_speedup_vs_marlin": marlin_total_us / best_per_shape_us,
            "humming_global_speedup_vs_marlin": marlin_total_us / humming_total_us,
            "humming_winning_shapes": [
                row["name"] for row in results if row["humming_speedup_vs_marlin"] > 1.0
            ],
        },
        "warnings": [
            "Random tensors and isolated hot-cache launches rank shapes but cannot qualify whole-model speedup.",
            "Marlin and Humming independently quantize the same dense random weight; only latency and basic finite/shape sanity are claimed.",
            "A mixed backend is worth implementing only if the weighted oracle margin can plausibly clear the 3% whole-model gate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
