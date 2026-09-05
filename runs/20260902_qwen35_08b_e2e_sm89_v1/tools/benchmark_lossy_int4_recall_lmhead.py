#!/usr/bin/env python3
"""Screen packed INT4 lm-head scans intended only for shortlist recall."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from transformers import Qwen3_5ForConditionalGeneration

from benchmark_exact_bf16_packed_lmhead import (
    capture,
    launch_packed,
    pack_exact_bf16,
    paired_graph_samples,
)


@triton.jit
def _int4_recall_lmhead_kernel(
    x_ptr,
    packed_qweight_ptr,
    scale_ptr,
    output_ptr,
    n: tl.constexpr,
    k: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_p = tl.arange(0, BLOCK_K_PACKED)
    k_packed: tl.constexpr = k // 2
    valid = (offsets_n[:, None] < n) & (offsets_p[None, :] < k_packed)
    linear = offsets_n[:, None] * k_packed + offsets_p[None, :]
    byte = tl.load(packed_qweight_ptr + linear, mask=valid, other=0).to(tl.uint32)
    low = (byte & 15).to(tl.int32)
    high = ((byte >> 4) & 15).to(tl.int32)
    qlow = tl.where(low < 8, low, low - 16).to(tl.float32)
    qhigh = tl.where(high < 8, high, high - 16).to(tl.float32)

    offsets_k0 = offsets_p * 2
    offsets_k1 = offsets_k0 + 1
    x0 = tl.load(x_ptr + offsets_k0, mask=offsets_k0 < k, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + offsets_k1, mask=offsets_k1 < k, other=0.0).to(tl.float32)
    groups_per_row: tl.constexpr = k // GROUP_K
    scale_offsets = offsets_n[:, None] * groups_per_row + offsets_k0[None, :] // GROUP_K
    scale = tl.load(scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.float32)
    accum = tl.sum((qlow * x0[None, :] + qhigh * x1[None, :]) * scale, axis=1)
    tl.store(output_ptr + offsets_n, accum, mask=offsets_n < n)


def quantize_int4(weight: torch.Tensor, group_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = weight.shape
    if k % group_k or group_k % 2:
        raise ValueError("K must be divisible by an even group_k")
    blocks = weight.float().reshape(n, k // group_k, group_k)
    scales = blocks.abs().amax(dim=2).div(7.0).clamp_min(torch.finfo(torch.float32).tiny)
    qweight = torch.round(blocks / scales[:, :, None]).clamp(-7, 7).to(torch.int8).reshape(n, k)
    low = qweight[:, 0::2].to(torch.int16) & 15
    high = (qweight[:, 1::2].to(torch.int16) & 15) << 4
    packed = (low | high).to(torch.uint8).contiguous()
    return packed, scales.to(torch.float16).contiguous()


def launch_int4(
    x: torch.Tensor,
    packed_qweight: torch.Tensor,
    scales: torch.Tensor,
    output: torch.Tensor,
    k: int,
    group_k: int,
    block_n: int,
    num_warps: int,
) -> None:
    n = packed_qweight.shape[0]
    _int4_recall_lmhead_kernel[(triton.cdiv(n, block_n),)](
        x,
        packed_qweight,
        scales,
        output,
        n=n,
        k=k,
        GROUP_K=group_k,
        BLOCK_N=block_n,
        BLOCK_K_PACKED=triton.next_power_of_2(k // 2),
        num_warps=num_warps,
        num_stages=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups", default="32,128,1024")
    parser.add_argument("--schedules", default="4x4,8x4,16x4,32x4")
    parser.add_argument("--vectors", type=int, default=8)
    parser.add_argument("--shortlist", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(20260905)
    model_path = args.model.resolve()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).eval().to("cuda")
    weight = model.lm_head.weight.detach().contiguous()
    n, k = weight.shape
    del model
    torch.cuda.empty_cache()

    packed_exact, exact = pack_exact_bf16(weight)
    del packed_exact["packed_blocks"]
    vectors = (
        torch.randn((args.vectors, k), device="cuda", dtype=torch.bfloat16) * 0.02
    ).contiguous()
    exact_outputs = []
    for vector in vectors:
        output = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
        launch_packed(vector[None, :], packed_exact, output, n, k, 16, 8)
        exact_outputs.append(output)
    torch.cuda.synchronize()

    schedules = [tuple(int(value) for value in item.split("x")) for item in args.schedules.split(",")]
    results = []
    for group_k in (int(value) for value in args.groups.split(",")):
        packed_qweight, scales = quantize_int4(weight, group_k)
        candidate_bytes = packed_qweight.numel() + scales.numel() * scales.element_size()
        candidate_output = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
        for block_n, num_warps in schedules:
            launch_int4(vectors[0:1], packed_qweight, scales, candidate_output, k, group_k, block_n, num_warps)
            torch.cuda.synchronize()
            exact_graph = capture(
                lambda: launch_packed(vectors[0:1], packed_exact, exact_outputs[0], n, k, 16, 8)
            )
            candidate_graph = capture(
                lambda: launch_int4(vectors[0:1], packed_qweight, scales, candidate_output, k, group_k, block_n, num_warps)
            )
            samples = paired_graph_samples(exact_graph, candidate_graph, args.iterations, args.repeats)
            control_us = statistics.median(samples["baseline_us"])
            candidate_us = statistics.median(samples["candidate_us"])

            ranks = []
            top1_equal = 0
            for vector, exact_output in zip(vectors, exact_outputs, strict=True):
                launch_int4(vector[None, :], packed_qweight, scales, candidate_output, k, group_k, block_n, num_warps)
                torch.cuda.synchronize()
                exact_top1 = exact_output.argmax().item()
                approximate_winner_score = candidate_output[0, exact_top1]
                rank = int((candidate_output[0] > approximate_winner_score).sum().item()) + 1
                ranks.append(rank)
                top1_equal += int(candidate_output.argmax().item() == exact_top1)
            results.append(
                {
                    "group_k": group_k,
                    "block_n": block_n,
                    "num_warps": num_warps,
                    "candidate_bytes": candidate_bytes,
                    "byte_fraction_vs_dense_bf16": candidate_bytes / (weight.numel() * 2),
                    "median_exact_packed_us": control_us,
                    "median_int4_scan_us": candidate_us,
                    "speedup_vs_exact_packed": control_us / candidate_us,
                    "quality": {
                        "random_vectors": args.vectors,
                        "top1_equal": top1_equal,
                        "exact_winner_recalled_in_topk": sum(rank <= args.shortlist for rank in ranks),
                        "shortlist": args.shortlist,
                        "exact_winner_approximate_ranks": ranks,
                        "maximum_rank": max(ranks),
                    },
                    "samples": samples,
                }
            )
        del packed_qweight, scales
        torch.cuda.empty_cache()

    best_by_group = [
        max((item for item in results if item["group_k"] == group_k), key=lambda item: item["speedup_vs_exact_packed"])
        for group_k in sorted({item["group_k"] for item in results})
    ]
    payload = {
        "schema_version": "sm89-lossy-int4-recall-lmhead-screen-v1",
        "status": "PASS",
        "scope": {
            "model": str(model_path),
            "gpu": torch.cuda.get_device_name(0),
            "weight_shape": [n, k],
            "control": "exact-packed BF16 lm-head",
            "candidate": "packed symmetric groupwise INT4 scan for shortlist recall only",
        },
        "control_weight_bit_exact": exact,
        "results": results,
        "best_by_group": best_by_group,
        "best_speed": max(results, key=lambda item: item["speedup_vs_exact_packed"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
