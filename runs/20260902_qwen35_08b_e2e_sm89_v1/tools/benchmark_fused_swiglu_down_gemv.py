#!/usr/bin/env python3
"""Bounded SM89 screen for fusing SwiGLU materialization into MLP down GEMV."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton


@triton.jit
def fused_swiglu_down_gemv_kernel(
    gate_up,
    weight,
    out,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)
    gate = tl.load(gate_up + cols, mask=cols < K, other=0.0).to(tl.float32)
    up = tl.load(gate_up + K + cols, mask=cols < K, other=0.0).to(tl.float32)
    # Match the stock compiled activation boundary: its result is materialized
    # as BF16 before the down projection consumes it.
    activated = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16)
    w = tl.load(
        weight + rows[:, None] * K + cols[None, :],
        mask=(rows[:, None] < N) & (cols[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    accum = tl.sum(w * activated[None, :].to(tl.float32), axis=1)
    tl.store(out + rows, accum, mask=rows < N)


def launch_candidate(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    warps: int,
) -> None:
    n, k = weight.shape
    fused_swiglu_down_gemv_kernel[(triton.cdiv(n, block_n),)](
        gate_up,
        weight,
        out,
        N=n,
        K=k,
        BLOCK_N=block_n,
        BLOCK_K=triton.next_power_of_2(k),
        num_warps=warps,
        num_stages=1,
    )


def capture_graph(fn) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def elapsed_us(fn, iterations: int, repeats: int) -> list[float]:
    for _ in range(25):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(31)
    device = "cuda"
    n, k = 1024, 3584
    gate_up = torch.randn(2 * k, dtype=torch.bfloat16, device=device)
    weight = torch.randn(n, k, dtype=torch.bfloat16, device=device)
    stock_holder = [torch.empty(0, device=device)]

    def stock_call() -> None:
        stock_holder[0] = F.linear(F.silu(gate_up[:k]) * gate_up[k:], weight)

    stock_call()
    torch.cuda.synchronize()
    stock_reference = stock_holder[0].clone()
    stock_graph = capture_graph(stock_call)
    stock_samples = elapsed_us(stock_graph.replay, args.iterations, args.repeats)
    stock_median = statistics.median(stock_samples)

    results = []
    for block_n, warps in ((1, 4), (1, 8), (2, 4), (2, 8), (4, 4), (4, 8)):
        out = torch.empty(n, dtype=torch.bfloat16, device=device)
        try:
            call = lambda bn=block_n, w=warps: launch_candidate(
                gate_up, weight, out, bn, w
            )
            call()
            torch.cuda.synchronize()
            correctness = {
                "output_equal": torch.equal(stock_reference, out),
                "max_abs": float((stock_reference.float() - out.float()).abs().max()),
                "mismatch_count": int(torch.count_nonzero(stock_reference != out)),
            }
            graph = capture_graph(call)
            samples = elapsed_us(graph.replay, args.iterations, args.repeats)
            median = statistics.median(samples)
            results.append(
                {
                    "block_n": block_n,
                    "warps": warps,
                    "median_us": median,
                    "speedup_vs_stock_activation_plus_cublas": stock_median / median,
                    "samples_us": samples,
                    "correctness": correctness,
                }
            )
        except Exception as error:
            results.append(
                {
                    "block_n": block_n,
                    "warps": warps,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    valid = [row for row in results if "median_us" in row]
    best = min(valid, key=lambda row: row["median_us"])
    result = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "shape": {"N": n, "K": k, "input": "BF16[7168]", "weight": "BF16[1024,3584]"},
        "stock": {
            "path": "torch.compile-compatible SwiGLU materialization followed by cuBLAS GEMV",
            "median_us": stock_median,
            "samples_us": stock_samples,
        },
        "candidate_family": "fused_swiglu_down_gemv",
        "best": best,
        "results": results,
        "promotion_gate": {
            "minimum_local_speedup": 1.03,
            "requires_output_equal_for_strict_frontier": True,
            "passed": best["speedup_vs_stock_activation_plus_cublas"] >= 1.03
            and best["correctness"]["output_equal"],
        },
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
