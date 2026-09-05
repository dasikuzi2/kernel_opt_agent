#!/usr/bin/env python3
"""Compare vLLM's CuTeDSL skinny GEMM with torch/cuBLAS on Qwen3.5 shapes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    shape_dynamic_skinny_gemm,
)


SHAPES = (
    # name, M, N, K, multiplicity per decoded token
    ("gdn_qkvz", 1, 8192, 1024, 18),
    ("gdn_ba", 1, 32, 1024, 18),
    ("gdn_out", 1, 1024, 2048, 18),
    # vLLM stacks q=4096, k=512, and v=512 into one projection.
    ("attention_qkv", 1, 5120, 1024, 6),
    ("attention_out", 1, 1024, 2048, 6),
    ("mlp_gate_up", 1, 7168, 1024, 24),
    ("mlp_down", 1, 1024, 3584, 24),
    ("lm_head", 1, 248320, 1024, 1),
    # MTP-1 target verification evaluates two positions. These shapes test
    # whether one kernel can reuse each BF16 weight row for both positions;
    # the single-token decode results do not predict this amortization.
    ("gdn_qkvz_m2", 2, 8192, 1024, 18),
    ("gdn_ba_m2", 2, 32, 1024, 18),
    ("gdn_out_m2", 2, 1024, 2048, 18),
    ("attention_qkv_m2", 2, 5120, 1024, 6),
    ("attention_out_m2", 2, 1024, 2048, 6),
    ("mlp_gate_up_m2", 2, 7168, 1024, 24),
    ("mlp_down_m2", 2, 1024, 3584, 24),
    # MTP-1 verifies the target token and one draft token together.  The
    # production M=1 specialization deliberately falls back for this shape,
    # so screen a kernel that reads each vocabulary row once and computes both
    # logits before considering a full-model speculative-decode experiment.
    ("lm_head_m2", 2, 248320, 1024, 1),
)


@triton.jit
def _gemv_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n: tl.constexpr,
    k: tl.constexpr,
    m: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    x0 = tl.load(x_ptr + offsets_k, mask=offsets_k < k, other=0.0)
    weight = tl.load(
        weight_ptr + offsets_n[:, None] * k + offsets_k[None, :],
        mask=(offsets_n[:, None] < n) & (offsets_k[None, :] < k),
        other=0.0,
    )
    accum0 = tl.sum(
        weight.to(tl.float32) * x0[None, :].to(tl.float32), axis=1
    )
    tl.store(output_ptr + offsets_n, accum0, mask=offsets_n < n)
    if m == 2:
        x1 = tl.load(x_ptr + k + offsets_k, mask=offsets_k < k, other=0.0)
        accum1 = tl.sum(
            weight.to(tl.float32) * x1[None, :].to(tl.float32), axis=1
        )
        tl.store(output_ptr + n + offsets_n, accum1, mask=offsets_n < n)


def triton_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    if m not in (1, 2):
        raise ValueError("screening kernel only supports M=1 or M=2")
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_k = triton.next_power_of_2(k)
    _gemv_kernel[(triton.cdiv(n, block_n),)](
        x,
        weight,
        output,
        n=n,
        k=k,
        m=m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def elapsed_us(fn, iterations: int, repeats: int) -> list[float]:
    for _ in range(20):
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
    return samples


def paired_elapsed_us(
    baseline_fn,
    candidate_fn,
    iterations: int,
    repeats: int,
) -> dict:
    """Interleave both sides so laptop boost drift cannot favor one backend."""
    for _ in range(20):
        baseline_fn()
        candidate_fn()
    torch.cuda.synchronize()
    samples = {"baseline_us": [], "candidate_us": [], "order": []}
    for repeat in range(repeats):
        order = (
            (("baseline", baseline_fn), ("candidate", candidate_fn))
            if repeat % 2 == 0
            else (("candidate", candidate_fn), ("baseline", baseline_fn))
        )
        samples["order"].append([name for name, _ in order])
        for name, fn in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            end.synchronize()
            samples[f"{name}_us"].append(
                start.elapsed_time(end) * 1000.0 / iterations
            )
    baseline_median = statistics.median(samples["baseline_us"])
    candidate_median = statistics.median(samples["candidate_us"])
    samples.update(
        {
            "baseline_median_us": baseline_median,
            "candidate_median_us": candidate_median,
            "speedup": baseline_median / candidate_median,
        }
    )
    return samples


def cold_paired_elapsed_us(
    baseline_fn,
    candidate_fn,
    eviction: torch.Tensor,
    repeats: int,
) -> dict:
    """Time one invocation after evicting weights from the last-level cache."""

    def evict() -> None:
        # The mutation forces a read and write of a buffer larger than SM89 L2.
        # It is deliberately queued before the start event and excluded from
        # the measured interval.
        eviction.add_(1)

    for _ in range(5):
        evict()
        baseline_fn()
        evict()
        candidate_fn()
    torch.cuda.synchronize()
    samples = {"baseline_us": [], "candidate_us": [], "order": []}
    for repeat in range(repeats):
        order = (
            (("baseline", baseline_fn), ("candidate", candidate_fn))
            if repeat % 2 == 0
            else (("candidate", candidate_fn), ("baseline", baseline_fn))
        )
        samples["order"].append([name for name, _ in order])
        for name, fn in order:
            evict()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples[f"{name}_us"].append(start.elapsed_time(end) * 1000.0)
    baseline_median = statistics.median(samples["baseline_us"])
    candidate_median = statistics.median(samples["candidate_us"])
    samples.update(
        {
            "baseline_median_us": baseline_median,
            "candidate_median_us": candidate_median,
            "speedup": baseline_median / candidate_median,
            "method": "single invocation after an untimed L2-eviction mutation",
            "eviction_bytes": eviction.numel() * eviction.element_size(),
        }
    )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--paired-iterations", type=int, default=50)
    parser.add_argument("--paired-repeats", type=int, default=9)
    parser.add_argument("--cold-cache-repeats", type=int, default=11)
    parser.add_argument("--cold-cache-eviction-mib", type=int, default=64)
    parser.add_argument(
        "--production-stream-bytes",
        type=int,
        help="Total unique bytes streamed by one production iteration.",
    )
    parser.add_argument(
        "--extended-schedule-search",
        action="store_true",
        help="Search a larger block/warp/stage space for the selected shapes.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        choices=tuple(shape[0] for shape in SHAPES),
        help="Benchmark only the named shape; repeat to select more than one.",
    )
    args = parser.parse_args()

    torch.manual_seed(20260905)
    device_l2_bytes = int(torch.cuda.get_device_properties(0).L2_cache_size)
    rows = []
    selected_shapes = (
        tuple(shape for shape in SHAPES if shape[0] in args.shape)
        if args.shape
        else SHAPES
    )
    for name, m, n, k, multiplicity in selected_shapes:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        reference = F.linear(x, weight)
        torch_times = elapsed_us(
            lambda: F.linear(x, weight), args.iterations, args.repeats
        )
        torch_median = statistics.median(torch_times)
        candidates = []
        schedules = (
            (
                (block_n, num_warps, num_stages)
                for block_n in (1, 2, 4, 8, 16, 32)
                for num_warps in (1, 2, 4, 8)
                for num_stages in (1, 2, 3)
            )
            if args.extended_schedule_search
            else (
                (block_n, num_warps, 1)
                for block_n in (1, 2, 4, 8)
                for num_warps in (4, 8)
            )
        )
        for block_n, num_warps, num_stages in schedules:
            try:
                    candidate = triton_gemv(
                        x,
                        weight,
                        block_n=block_n,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    torch.cuda.synchronize()
                    max_abs = float(
                        (candidate.float() - reference.float()).abs().max()
                    )
                    mean_abs = float(
                        (candidate.float() - reference.float()).abs().mean()
                    )
                    close = bool(
                        torch.allclose(
                            candidate.float(),
                            reference.float(),
                            rtol=0.05,
                            atol=0.05,
                        )
                    )
                    times = elapsed_us(
                        lambda block_n=block_n, num_warps=num_warps, num_stages=num_stages: triton_gemv(
                            x,
                            weight,
                            block_n=block_n,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        ),
                        args.iterations,
                        args.repeats,
                    )
                    triton_median = statistics.median(times)
                    candidates.append(
                        {
                            "block_n": block_n,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            "correct": close,
                            "max_abs": max_abs,
                            "mean_abs": mean_abs,
                            "median_us": triton_median,
                            "speedup": torch_median / triton_median,
                            "samples_us": times,
                        }
                    )
            except Exception as exc:
                candidates.append(
                    {
                        "block_n": block_n,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                        "correct": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        valid = [
            candidate
            for candidate in candidates
            if candidate.get("correct") and "median_us" in candidate
        ]
        best = min(valid, key=lambda candidate: candidate["median_us"]) if valid else None
        paired = None
        cold_paired = None
        paired_vs_incumbent = None
        if best is not None:
            block_n = int(best["block_n"])
            num_warps = int(best["num_warps"])
            num_stages = int(best["num_stages"])
            paired = paired_elapsed_us(
                lambda: F.linear(x, weight),
                lambda: triton_gemv(
                    x,
                    weight,
                    block_n=block_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                ),
                args.paired_iterations,
                args.paired_repeats,
            )
            eviction = torch.zeros(
                args.cold_cache_eviction_mib * 1024 * 1024 // 2,
                dtype=torch.bfloat16,
                device="cuda",
            )
            cold_paired = cold_paired_elapsed_us(
                lambda: F.linear(x, weight),
                lambda: triton_gemv(
                    x,
                    weight,
                    block_n=block_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                ),
                eviction,
                args.cold_cache_repeats,
            )
            if args.extended_schedule_search and name == "lm_head":
                paired_vs_incumbent = paired_elapsed_us(
                    lambda: triton_gemv(
                        x,
                        weight,
                        block_n=4,
                        num_warps=8,
                        num_stages=1,
                    ),
                    lambda: triton_gemv(
                        x,
                        weight,
                        block_n=block_n,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    ),
                    args.paired_iterations,
                    args.paired_repeats,
                )
        rows.append(
            {
                "name": name,
                "shape": [m, n, k],
                "multiplicity": multiplicity,
                "weight_bytes": n * k * 2,
                "weight_fits_l2": n * k * 2 <= device_l2_bytes,
                "production_cache_mismatch": bool(
                    args.production_stream_bytes
                    and n * k * 2 <= device_l2_bytes
                    and args.production_stream_bytes > device_l2_bytes
                ),
                "torch_median_us": torch_median,
                "torch_samples_us": torch_times,
                "best_triton": best,
                "paired_best_vs_torch": paired,
                "cold_paired_best_vs_torch": cold_paired,
                "paired_best_vs_incumbent": paired_vs_incumbent,
                "candidates": candidates,
            }
        )
        del x, weight, reference
        torch.cuda.empty_cache()

    comparable = [row for row in rows if row["best_triton"] is not None]
    torch_total = sum(
        row["torch_median_us"] * row["multiplicity"] for row in comparable
    )
    skinny_total = sum(
        row["best_triton"]["median_us"] * row["multiplicity"]
        for row in comparable
    )
    paired_torch_total = sum(
        row["paired_best_vs_torch"]["baseline_median_us"] * row["multiplicity"]
        for row in comparable
    )
    paired_skinny_total = sum(
        row["paired_best_vs_torch"]["candidate_median_us"] * row["multiplicity"]
        for row in comparable
    )
    result = {
        "schema_version": "qwen35-bf16-triton-gemv-screen-v3",
        "device": torch.cuda.get_device_name(),
        "device_l2_bytes": device_l2_bytes,
        "torch_version": torch.__version__,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "paired_iterations": args.paired_iterations,
        "paired_repeats": args.paired_repeats,
        "cold_cache_repeats": args.cold_cache_repeats,
        "cold_cache_eviction_mib": args.cold_cache_eviction_mib,
        "production_stream_bytes": args.production_stream_bytes,
        "cache_interpretation": (
            "Use cold-cache timings for production prediction when an isolated "
            "weight fits L2 but the production iteration streams more than L2."
        ),
        "extended_schedule_search": args.extended_schedule_search,
        "cutedsl_probe": {
            "python_package_available": shape_dynamic_skinny_gemm.is_available(),
            "sm89_compatible": False,
            "reason": "vLLM CuTeDSL skinny GEMM requires SM90 or newer",
        },
        "weighted_shape_sum_us": {
            "torch": torch_total,
            "skinny": skinny_total,
            "speedup": torch_total / skinny_total if skinny_total else None,
            "warning": "sum of isolated medians is a screening estimate, not end-to-end latency",
        },
        "paired_weighted_shape_sum_us": {
            "torch": paired_torch_total,
            "skinny": paired_skinny_total,
            "speedup": paired_torch_total / paired_skinny_total,
            "method": "same-process interleaved order, reversed every repeat",
            "warning": "still an isolated shape sum; use only to decide full-model promotion",
        },
        "results": rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
