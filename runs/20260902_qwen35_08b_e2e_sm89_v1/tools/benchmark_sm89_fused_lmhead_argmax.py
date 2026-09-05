#!/usr/bin/env python3
"""Bounded SM89 BF16 lm-head + greedy-argmax fusion screen.

The production comparison is the accepted Triton lm-head kernel followed by
the same BF16->FP32 argmax used by vLLM's greedy sampler.  The candidate keeps
the accepted dot-product reduction and rounds every score to BF16 before a
hierarchical argmax, but never materializes the full logits tensor.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


N = 248320
K = 1024


@triton.jit
def _lmhead_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + offsets_k, mask=offsets_k < k, other=0.0)
    weight = tl.load(
        weight_ptr + offsets_n[:, None] * k + offsets_k[None, :],
        mask=(offsets_n[:, None] < n) & (offsets_k[None, :] < k),
        other=0.0,
    )
    accum = tl.sum(weight.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
    tl.store(output_ptr + offsets_n, accum, mask=offsets_n < n)


@triton.jit
def _lmhead_local_argmax_kernel(
    x_ptr,
    weight_ptr,
    local_values_ptr,
    local_indices_ptr,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    mask_n = offsets_n < n
    x = tl.load(x_ptr + offsets_k, mask=offsets_k < k, other=0.0)
    weight = tl.load(
        weight_ptr + offsets_n[:, None] * k + offsets_k[None, :],
        mask=mask_n[:, None] & (offsets_k[None, :] < k),
        other=0.0,
    )
    accum = tl.sum(weight.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
    # Match the accepted path exactly at its externally visible boundary:
    # logits are stored as BF16, then widened to FP32 by the sampler.
    rounded = accum.to(tl.bfloat16).to(tl.float32)
    rounded = tl.where(mask_n, rounded, float("-inf"))
    value, relative_index = tl.max(rounded, axis=0, return_indices=True)
    tl.store(local_values_ptr + pid, value)
    tl.store(local_indices_ptr + pid, pid * BLOCK_N + relative_index)


@triton.jit
def _reduce_argmax_kernel(
    values_ptr,
    indices_ptr,
    output_values_ptr,
    output_indices_ptr,
    length: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < length
    values = tl.load(values_ptr + offsets, mask=mask, other=float("-inf"))
    value, relative_index = tl.max(values, axis=0, return_indices=True)
    index = tl.load(indices_ptr + pid * BLOCK + relative_index)
    tl.store(output_values_ptr + pid, value)
    tl.store(output_indices_ptr + pid, index)


def accepted_lmhead_then_argmax(
    x: torch.Tensor,
    weight: torch.Tensor,
    logits: torch.Tensor,
    token: torch.Tensor,
) -> None:
    _lmhead_kernel[(triton.cdiv(N, 4),)](
        x,
        weight,
        logits,
        n=N,
        k=K,
        BLOCK_N=4,
        BLOCK_K=K,
        num_warps=8,
        num_stages=1,
    )
    token.copy_(logits.float().argmax(dim=-1).to(torch.int32))


def fused_lmhead_argmax(
    x: torch.Tensor,
    weight: torch.Tensor,
    local_values: torch.Tensor,
    local_indices: torch.Tensor,
    group_values: torch.Tensor,
    group_indices: torch.Tensor,
    token: torch.Tensor,
    *,
    block_n: int,
    num_warps: int,
) -> None:
    local_count = triton.cdiv(N, block_n)
    group_count = triton.cdiv(local_count, 1024)
    _lmhead_local_argmax_kernel[(local_count,)](
        x,
        weight,
        local_values,
        local_indices,
        n=N,
        k=K,
        BLOCK_N=block_n,
        BLOCK_K=K,
        num_warps=num_warps,
        num_stages=1,
    )
    _reduce_argmax_kernel[(group_count,)](
        local_values,
        local_indices,
        group_values,
        group_indices,
        length=local_count,
        BLOCK=1024,
        num_warps=8,
    )
    final_block = triton.next_power_of_2(group_count)
    _reduce_argmax_kernel[(1,)](
        group_values,
        group_indices,
        group_values,
        token,
        length=group_count,
        BLOCK=final_block,
        num_warps=4,
    )


def capture(fn) -> torch.cuda.CUDAGraph:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def paired_timing(baseline_fn, candidate_fn, iterations: int, repeats: int) -> dict:
    for _ in range(10):
        baseline_fn()
        candidate_fn()
    torch.cuda.synchronize()
    samples = {"baseline_us": [], "candidate_us": [], "orders": []}
    for repeat in range(repeats):
        order = (
            (("baseline", baseline_fn), ("candidate", candidate_fn))
            if repeat % 2 == 0
            else (("candidate", candidate_fn), ("baseline", baseline_fn))
        )
        samples["orders"].append([name for name, _ in order])
        for name, fn in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            end.synchronize()
            samples[f"{name}_us"].append(start.elapsed_time(end) * 1000 / iterations)
    baseline = statistics.median(samples["baseline_us"])
    candidate = statistics.median(samples["candidate_us"])
    samples.update(
        baseline_median_us=baseline,
        candidate_median_us=candidate,
        speedup=baseline / candidate,
        saving_us=baseline - candidate,
    )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    torch.manual_seed(20260905)
    device = torch.device("cuda")
    weight = (torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.02).contiguous()
    x = (torch.randn(1, K, device=device, dtype=torch.bfloat16) * 0.02).contiguous()
    logits = torch.empty((1, N), device=device, dtype=torch.bfloat16)
    baseline_token = torch.empty((1,), device=device, dtype=torch.int32)

    candidates = []
    for block_n, num_warps in ((4, 4), (4, 8), (8, 4), (8, 8), (16, 4), (16, 8)):
        local_count = triton.cdiv(N, block_n)
        group_count = triton.cdiv(local_count, 1024)
        local_values = torch.empty(local_count, device=device, dtype=torch.float32)
        local_indices = torch.empty(local_count, device=device, dtype=torch.int32)
        group_values = torch.empty(group_count, device=device, dtype=torch.float32)
        group_indices = torch.empty(group_count, device=device, dtype=torch.int32)
        candidate_token = torch.empty((1,), device=device, dtype=torch.int32)

        baseline_fn = lambda: accepted_lmhead_then_argmax(
            x, weight, logits, baseline_token
        )
        candidate_fn = lambda: fused_lmhead_argmax(
            x,
            weight,
            local_values,
            local_indices,
            group_values,
            group_indices,
            candidate_token,
            block_n=block_n,
            num_warps=num_warps,
        )
        baseline_fn()
        candidate_fn()
        torch.cuda.synchronize()
        random_exact = int(baseline_token.item()) == int(candidate_token.item())

        x.zero_()
        baseline_fn()
        candidate_fn()
        torch.cuda.synchronize()
        tie_exact = int(baseline_token.item()) == int(candidate_token.item()) == 0
        x.normal_(0.0, 0.02)

        direct = paired_timing(
            baseline_fn, candidate_fn, args.iterations, args.repeats
        )
        baseline_graph = capture(baseline_fn)
        candidate_graph = capture(candidate_fn)
        graph = paired_timing(
            baseline_graph.replay,
            candidate_graph.replay,
            args.iterations,
            args.repeats,
        )
        candidates.append(
            {
                "block_n": block_n,
                "num_warps": num_warps,
                "random_argmax_exact": random_exact,
                "all_zero_tie_returns_first_token": tie_exact,
                "direct": direct,
                "cuda_graph": graph,
                "predicted_whole_step_speedup": 8.079 / (8.079 - graph["saving_us"] / 1000),
                "passes_3pct_materiality_floor": graph["saving_us"] >= 235.3,
            }
        )

    best = max(candidates, key=lambda row: row["cuda_graph"]["saving_us"])
    result = {
        "schema_version": "sm89-fused-lmhead-argmax-screen-v1",
        "scope": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "dtype": "bfloat16",
            "shape": [1, N, K],
            "baseline": "accepted BLOCK_N=4/8-warp lm-head, BF16 logits materialization, FP32 torch.argmax",
            "candidate": "same BF16-rounded row scores with two-level hierarchical argmax",
            "strict_frontier_tpot_ms": 8.079,
            "minimum_whole_step_saving_for_1.03x_us": 235.3,
        },
        "candidates": candidates,
        "best": best,
        "decision": (
            "INTEGRATE"
            if best["passes_3pct_materiality_floor"]
            and best["random_argmax_exact"]
            and best["all_zero_tie_returns_first_token"]
            else "STOP_BELOW_MATERIALITY_OR_CORRECTNESS_GATE"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
