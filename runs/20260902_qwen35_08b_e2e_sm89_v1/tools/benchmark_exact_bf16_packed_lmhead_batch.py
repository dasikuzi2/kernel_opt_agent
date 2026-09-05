#!/usr/bin/env python3
"""Screen weight-reusing exact-packed BF16 lm-head kernels for small batches."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from transformers import Qwen3_5ForConditionalGeneration


PACK_BLOCK = 256


@triton.jit
def _packed_lmhead_batch_kernel(
    x_ptr,
    sign_mantissa_ptr,
    exponent_nibbles_ptr,
    base_exponent_ptr,
    fallback_slot_ptr,
    fallback_bits_ptr,
    output_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PACK_VALUES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    valid_weight = (offsets_n[:, None] < n) & (offsets_k[None, :] < k)
    linear = offsets_n[:, None] * k + offsets_k[None, :]
    block_id = linear // PACK_VALUES
    in_block = linear % PACK_VALUES

    sm = tl.load(sign_mantissa_ptr + linear, mask=valid_weight, other=0).to(
        tl.int32
    )
    pair = tl.load(
        exponent_nibbles_ptr + linear // 2, mask=valid_weight, other=0
    ).to(tl.int32)
    delta = (pair >> ((linear & 1) * 4)) & 0xF
    base = tl.load(
        base_exponent_ptr + block_id, mask=valid_weight, other=0
    ).to(tl.int32)
    slot = tl.load(fallback_slot_ptr + block_id, mask=valid_weight, other=-1)
    packed_bits = (
        ((sm & 0x80) << 8) | ((base + delta) << 7) | (sm & 0x7F)
    )
    fallback_bits = tl.load(
        fallback_bits_ptr + slot * PACK_VALUES + in_block,
        mask=valid_weight & (slot >= 0),
        other=0,
    ).to(tl.int32) & 0xFFFF
    raw_bf16 = tl.where(slot >= 0, fallback_bits, packed_bits)
    weight = tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=f,r",
        [raw_bf16 << 16],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

    for local_m in tl.static_range(0, BLOCK_M):
        row = pid_m * BLOCK_M + local_m
        x = tl.load(
            x_ptr + row * k + offsets_k,
            mask=(row < m) & (offsets_k < k),
            other=0.0,
        ).to(tl.float32)
        accum = tl.sum(weight * x[None, :], axis=1)
        tl.store(
            output_ptr + row * n + offsets_n,
            accum,
            mask=(row < m) & (offsets_n < n),
        )


def pack_exact_bf16(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    bits = weight.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).flatten()
    blocks = bits.reshape(-1, PACK_BLOCK)
    exponent_blocks = bits.bitwise_right_shift(7).bitwise_and(0xFF).reshape(
        -1, PACK_BLOCK
    )
    base = exponent_blocks.amin(dim=1)
    delta_blocks = exponent_blocks - base[:, None]
    packed_blocks = delta_blocks.amax(dim=1) <= 15
    sign_mantissa = bits.bitwise_and(0x7F).bitwise_or(
        bits.bitwise_right_shift(8).bitwise_and(0x80)
    ).to(torch.uint8)
    delta = torch.where(
        packed_blocks[:, None], delta_blocks, torch.zeros_like(delta_blocks)
    ).flatten()
    delta_pairs = delta.reshape(-1, 2)
    exponent_nibbles = (delta_pairs[:, 0] | (delta_pairs[:, 1] << 4)).to(
        torch.uint8
    )
    failed_ids = torch.nonzero(~packed_blocks, as_tuple=False).flatten()
    fallback_slot = torch.full(
        (blocks.shape[0],), -1, dtype=torch.int32, device=weight.device
    )
    fallback_slot[failed_ids] = torch.arange(
        failed_ids.numel(), dtype=torch.int32, device=weight.device
    )
    return {
        "sign_mantissa": sign_mantissa.contiguous(),
        "exponent_nibbles": exponent_nibbles.contiguous(),
        "base_exponent": base.to(torch.uint8).contiguous(),
        "fallback_slot": fallback_slot,
        "fallback_bits": blocks[failed_ids].to(torch.int16).contiguous(),
    }


def launch_packed(
    x: torch.Tensor,
    packed: dict[str, torch.Tensor],
    output: torch.Tensor,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    num_warps: int,
) -> None:
    m = x.shape[0]
    _packed_lmhead_batch_kernel[
        (triton.cdiv(n, block_n), triton.cdiv(m, block_m))
    ](
        x,
        packed["sign_mantissa"],
        packed["exponent_nibbles"],
        packed["base_exponent"],
        packed["fallback_slot"],
        packed["fallback_bits"],
        output,
        m=m,
        n=n,
        k=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=triton.next_power_of_2(k),
        PACK_VALUES=PACK_BLOCK,
        num_warps=num_warps,
        num_stages=1,
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


def paired_samples(
    baseline: torch.cuda.CUDAGraph,
    candidate: torch.cuda.CUDAGraph,
    iterations: int,
    repeats: int,
    warmup_pairs: int,
) -> dict[str, list]:
    for _ in range(warmup_pairs):
        baseline.replay()
        candidate.replay()
    torch.cuda.synchronize()
    result: dict[str, list] = {"baseline_us": [], "candidate_us": [], "order": []}
    for repeat in range(repeats):
        order = (
            (("baseline", baseline), ("candidate", candidate))
            if repeat % 2 == 0
            else (("candidate", candidate), ("baseline", baseline))
        )
        result["order"].append("-".join(name for name, _ in order))
        for name, graph in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            end.synchronize()
            result[f"{name}_us"].append(start.elapsed_time(end) * 1000 / iterations)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--block-n", default="2,4,8,16")
    parser.add_argument("--warps", default="4,8")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup-pairs", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(20260905)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model.resolve(), dtype=torch.bfloat16, local_files_only=True
    ).eval().to("cuda")
    weight = model.lm_head.weight.detach().contiguous()
    n, k = weight.shape
    del model
    torch.cuda.empty_cache()
    packed = pack_exact_bf16(weight)

    results = []
    for m in [int(value) for value in args.batch_sizes.split(",")]:
        x = (torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
        baseline_output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        candidate_output = torch.empty_like(baseline_output)
        baseline_fn = lambda: torch.mm(x, weight.t(), out=baseline_output)
        baseline_fn()
        torch.cuda.synchronize()
        baseline_graph = capture(baseline_fn)
        for block_n in [int(value) for value in args.block_n.split(",")]:
            for num_warps in [int(value) for value in args.warps.split(",")]:
                candidate_fn = lambda bn=block_n, nw=num_warps: launch_packed(
                    x, packed, candidate_output, n, k, m, bn, nw
                )
                candidate_fn()
                torch.cuda.synchronize()
                argmax_exact = bool(
                    torch.equal(
                        baseline_output.float().argmax(dim=-1),
                        candidate_output.float().argmax(dim=-1),
                    )
                )
                maximum_absolute_difference = float(
                    (baseline_output.float() - candidate_output.float())
                    .abs()
                    .max()
                    .item()
                )
                candidate_graph = capture(candidate_fn)
                samples = paired_samples(
                    baseline_graph,
                    candidate_graph,
                    args.iterations,
                    args.repeats,
                    args.warmup_pairs,
                )
                baseline_us = statistics.median(samples["baseline_us"])
                candidate_us = statistics.median(samples["candidate_us"])
                results.append(
                    {
                        "batch_size": m,
                        "block_m": m,
                        "block_n": block_n,
                        "num_warps": num_warps,
                        "argmax_exact": argmax_exact,
                        "maximum_absolute_logit_difference": maximum_absolute_difference,
                        "baseline_us": baseline_us,
                        "candidate_us": candidate_us,
                        "speedup": baseline_us / candidate_us,
                        "samples": samples,
                    }
                )

    best_by_batch = {}
    for m in sorted({item["batch_size"] for item in results}):
        best_by_batch[str(m)] = max(
            (item for item in results if item["batch_size"] == m),
            key=lambda item: item["speedup"],
        )
    payload = {
        "schema_version": "exact-packed-bf16-lmhead-small-batch-screen-v1",
        "status": "PASS",
        "scope": {
            "model": str(args.model.resolve()),
            "gpu": torch.cuda.get_device_name(0),
            "weight_shape": [n, k],
            "baseline": "torch.mm BF16 production library path",
            "candidate": "one CTA reuses each reconstructed weight tile across BLOCK_M rows",
        },
        "controls": {
            "iterations": args.iterations,
            "repeats": args.repeats,
            "warmup_pairs": args.warmup_pairs,
            "cuda_graph": True,
            "paired_order": "alternating",
        },
        "best_by_batch": best_by_batch,
        "results": results,
        "decision": (
            "PROMOTE_BATCH_VARIANT_FOR_END_TO_END_SCREEN"
            if any(
                item["argmax_exact"] and item["speedup"] >= 1.03
                for key, item in best_by_batch.items()
                if int(key) > 1
            )
            else "MEASURED_REJECT_SMALL_BATCH_EXTENSION"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "decision": payload["decision"], "best_by_batch": best_by_batch}, indent=2))


if __name__ == "__main__":
    main()
