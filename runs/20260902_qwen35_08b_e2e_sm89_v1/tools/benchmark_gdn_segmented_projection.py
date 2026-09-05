#!/usr/bin/env python3
"""Screen one-launch SM89 BF16 QKVZ+BA projection against two cuBLAS calls."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _segmented_gemv_kernel(
    x_ptr,
    main_weight_ptr,
    tail_weight_ptr,
    main_output_ptr,
    tail_output_ptr,
    MAIN_N: tl.constexpr,
    TAIL_N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAIN_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_k = tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + offsets_k, mask=offsets_k < K, other=0.0)
    if pid < MAIN_BLOCKS:
        offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        weight = tl.load(
            main_weight_ptr + offsets_n[:, None] * K + offsets_k[None, :],
            mask=(offsets_n[:, None] < MAIN_N) & (offsets_k[None, :] < K),
            other=0.0,
        )
        accum = tl.sum(
            weight.to(tl.float32) * x[None, :].to(tl.float32), axis=1
        )
        tl.store(main_output_ptr + offsets_n, accum, mask=offsets_n < MAIN_N)
    else:
        tail_pid = pid - MAIN_BLOCKS
        offsets_n = tail_pid * BLOCK_N + tl.arange(0, BLOCK_N)
        weight = tl.load(
            tail_weight_ptr + offsets_n[:, None] * K + offsets_k[None, :],
            mask=(offsets_n[:, None] < TAIL_N) & (offsets_k[None, :] < K),
            other=0.0,
        )
        accum = tl.sum(
            weight.to(tl.float32) * x[None, :].to(tl.float32), axis=1
        )
        tl.store(tail_output_ptr + offsets_n, accum, mask=offsets_n < TAIL_N)


def segmented_gemv(
    x: torch.Tensor,
    main_weight: torch.Tensor,
    tail_weight: torch.Tensor,
    *,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape != (1, 1024):
        raise ValueError(f"expected x=(1, 1024), got {tuple(x.shape)}")
    if main_weight.shape != (8192, 1024):
        raise ValueError(f"unexpected main weight: {tuple(main_weight.shape)}")
    if tail_weight.shape != (32, 1024):
        raise ValueError(f"unexpected tail weight: {tuple(tail_weight.shape)}")
    main_output = torch.empty((1, 8192), dtype=x.dtype, device=x.device)
    tail_output = torch.empty((1, 32), dtype=x.dtype, device=x.device)
    main_blocks = triton.cdiv(main_weight.shape[0], block_n)
    tail_blocks = triton.cdiv(tail_weight.shape[0], block_n)
    _segmented_gemv_kernel[(main_blocks + tail_blocks,)](
        x,
        main_weight,
        tail_weight,
        main_output,
        tail_output,
        MAIN_N=main_weight.shape[0],
        TAIL_N=tail_weight.shape[0],
        K=x.shape[-1],
        BLOCK_N=block_n,
        BLOCK_K=triton.next_power_of_2(x.shape[-1]),
        MAIN_BLOCKS=main_blocks,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return main_output, tail_output


def _event_us(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def paired(
    baseline_fn,
    candidate_fn,
    *,
    iterations: int,
    repeats: int,
    eviction: torch.Tensor | None = None,
) -> dict:
    for _ in range(10):
        baseline_fn()
        candidate_fn()
    torch.cuda.synchronize()
    result = {"baseline_us": [], "candidate_us": [], "order": []}
    for repeat in range(repeats):
        order = (
            (("baseline", baseline_fn), ("candidate", candidate_fn))
            if repeat % 2 == 0
            else (("candidate", candidate_fn), ("baseline", baseline_fn))
        )
        result["order"].append([name for name, _ in order])
        for name, fn in order:
            if eviction is not None:
                eviction.add_(1)
            result[f"{name}_us"].append(_event_us(fn, iterations))
    baseline_median = statistics.median(result["baseline_us"])
    candidate_median = statistics.median(result["candidate_us"])
    result.update(
        {
            "baseline_median_us": baseline_median,
            "candidate_median_us": candidate_median,
            "speedup": baseline_median / candidate_median,
            "iterations_per_sample": iterations,
            "cache_state": "cold" if eviction is not None else "warm",
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warm-iterations", type=int, default=50)
    parser.add_argument("--warm-repeats", type=int, default=9)
    parser.add_argument("--cold-repeats", type=int, default=21)
    args = parser.parse_args()

    torch.manual_seed(20260905)
    x = torch.randn((1, 1024), device="cuda", dtype=torch.bfloat16) * 0.1
    main_weight = (
        torch.randn((8192, 1024), device="cuda", dtype=torch.bfloat16) * 0.1
    )
    tail_weight = (
        torch.randn((32, 1024), device="cuda", dtype=torch.bfloat16) * 0.1
    )
    baseline_fn = lambda: (
        F.linear(x, main_weight),
        F.linear(x, tail_weight),
    )
    reference_main, reference_tail = baseline_fn()
    schedules = []
    for block_n in (1, 2, 4, 8):
        for num_warps in (4, 8):
            candidate_fn = lambda block_n=block_n, num_warps=num_warps: segmented_gemv(
                x,
                main_weight,
                tail_weight,
                block_n=block_n,
                num_warps=num_warps,
                num_stages=1,
            )
            candidate_main, candidate_tail = candidate_fn()
            torch.cuda.synchronize()
            correct = torch.allclose(
                candidate_main.float(), reference_main.float(), rtol=0.05, atol=0.05
            ) and torch.allclose(
                candidate_tail.float(), reference_tail.float(), rtol=0.05, atol=0.05
            )
            schedules.append(
                {
                    "block_n": block_n,
                    "num_warps": num_warps,
                    "num_stages": 1,
                    "correct": bool(correct),
                    "main_max_abs": float(
                        (candidate_main.float() - reference_main.float()).abs().max()
                    ),
                    "tail_max_abs": float(
                        (candidate_tail.float() - reference_tail.float()).abs().max()
                    ),
                    "warm": paired(
                        baseline_fn,
                        candidate_fn,
                        iterations=args.warm_iterations,
                        repeats=args.warm_repeats,
                    ),
                }
            )
    valid = [row for row in schedules if row["correct"]]
    best = max(valid, key=lambda row: row["warm"]["speedup"])
    candidate_fn = lambda: segmented_gemv(
        x,
        main_weight,
        tail_weight,
        block_n=best["block_n"],
        num_warps=best["num_warps"],
        num_stages=best["num_stages"],
    )
    eviction = torch.zeros(
        64 * 1024 * 1024 // 2, device="cuda", dtype=torch.bfloat16
    )
    cold = paired(
        baseline_fn,
        candidate_fn,
        iterations=1,
        repeats=args.cold_repeats,
        eviction=eviction,
    )
    payload = {
        "schema_version": "sm89-gdn-segmented-projection-screen-v1",
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "shapes": {"x": [1, 1024], "main": [8192, 1024], "tail": [32, 1024]},
        "l2_bytes": int(torch.cuda.get_device_properties(0).L2_cache_size),
        "eviction_bytes": eviction.numel() * eviction.element_size(),
        "best_schedule": {
            "block_n": best["block_n"],
            "num_warps": best["num_warps"],
            "num_stages": best["num_stages"],
        },
        "best_warm": best["warm"],
        "best_cold": cold,
        "schedules": schedules,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "best_schedule", "best_warm", "best_cold")}, indent=2))


if __name__ == "__main__":
    main()
