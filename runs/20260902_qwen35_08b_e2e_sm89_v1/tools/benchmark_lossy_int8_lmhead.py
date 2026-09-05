#!/usr/bin/env python3
"""Screen bandwidth-oriented INT8 lm-head candidates on SM89.

The accepted exact-packed BF16 kernel is the control.  Candidates use
symmetric weight-only INT8 with one scale per contiguous K group.  This is a
lossy frontier: speed and output stability are reported separately so a fast
kernel cannot silently inherit the exact candidate's correctness claim.
"""

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
def _int8_lmhead_kernel(
    x_ptr,
    qweight_ptr,
    scale_ptr,
    output_ptr,
    n: tl.constexpr,
    k: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    valid = (offsets_n[:, None] < n) & (offsets_k[None, :] < k)
    linear = offsets_n[:, None] * k + offsets_k[None, :]
    qweight = tl.load(qweight_ptr + linear, mask=valid, other=0).to(tl.float32)
    groups_per_row: tl.constexpr = k // GROUP_K
    scale_offsets = offsets_n[:, None] * groups_per_row + offsets_k[None, :] // GROUP_K
    scale = tl.load(scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets_k, mask=offsets_k < k, other=0.0).to(tl.float32)
    accum = tl.sum(qweight * scale * x[None, :], axis=1)
    tl.store(output_ptr + offsets_n, accum, mask=offsets_n < n)


def quantize_int8(weight: torch.Tensor, group_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = weight.shape
    if k % group_k:
        raise ValueError(f"K={k} is not divisible by group_k={group_k}")
    blocks = weight.float().reshape(n, k // group_k, group_k)
    scales = blocks.abs().amax(dim=2).div(127.0).clamp_min(torch.finfo(torch.float32).tiny)
    qweight = torch.round(blocks / scales[:, :, None]).clamp(-127, 127).to(torch.int8)
    return qweight.reshape(n, k).contiguous(), scales.to(torch.float16).contiguous()


def launch_int8(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    output: torch.Tensor,
    group_k: int,
    block_n: int,
    num_warps: int,
) -> None:
    n, k = qweight.shape
    _int8_lmhead_kernel[(triton.cdiv(n, block_n),)](
        x,
        qweight,
        scales,
        output,
        n=n,
        k=k,
        GROUP_K=group_k,
        BLOCK_N=block_n,
        BLOCK_K=triton.next_power_of_2(k),
        num_warps=num_warps,
        num_stages=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups", default="32,64,128,256,1024")
    parser.add_argument("--schedules", default="4x4,4x8,8x4,8x8,16x4,16x8,32x4,32x8")
    parser.add_argument("--vectors", type=int, default=8)
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

    packed, exact = pack_exact_bf16(weight)
    del packed["packed_blocks"]
    vectors = (
        torch.randn((args.vectors, k), device="cuda", dtype=torch.bfloat16) * 0.02
    ).contiguous()
    exact_outputs = []
    for vector in vectors:
        output = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
        launch_packed(vector[None, :], packed, output, n, k, 16, 8)
        exact_outputs.append(output)
    torch.cuda.synchronize()

    schedules = [
        tuple(int(value) for value in item.split("x"))
        for item in args.schedules.split(",")
    ]
    results = []
    for group_k in (int(value) for value in args.groups.split(",")):
        qweight, scales = quantize_int8(weight, group_k)
        int8_bytes = qweight.numel() + scales.numel() * scales.element_size()
        candidate_output = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
        for block_n, num_warps in schedules:
            launch_int8(
                vectors[0:1], qweight, scales, candidate_output,
                group_k, block_n, num_warps,
            )
            torch.cuda.synchronize()
            exact_graph = capture(
                lambda: launch_packed(
                    vectors[0:1], packed, exact_outputs[0], n, k, 16, 8
                )
            )
            candidate_graph = capture(
                lambda: launch_int8(
                    vectors[0:1], qweight, scales, candidate_output,
                    group_k, block_n, num_warps,
                )
            )
            samples = paired_graph_samples(
                exact_graph, candidate_graph, args.iterations, args.repeats
            )
            control_us = statistics.median(samples["baseline_us"])
            candidate_us = statistics.median(samples["candidate_us"])

            quality = []
            for vector, exact_output in zip(vectors, exact_outputs, strict=True):
                launch_int8(
                    vector[None, :], qweight, scales, candidate_output,
                    group_k, block_n, num_warps,
                )
                torch.cuda.synchronize()
                delta = candidate_output.float() - exact_output.float()
                quality.append(
                    {
                        "top1_equal": bool(
                            candidate_output.argmax().item() == exact_output.argmax().item()
                        ),
                        "maximum_absolute_logit_difference": float(delta.abs().max().item()),
                        "mean_absolute_logit_difference": float(delta.abs().mean().item()),
                    }
                )
            results.append(
                {
                    "group_k": group_k,
                    "block_n": block_n,
                    "num_warps": num_warps,
                    "int8_bytes": int8_bytes,
                    "byte_fraction_vs_dense_bf16": int8_bytes / (weight.numel() * 2),
                    "median_exact_packed_us": control_us,
                    "median_int8_us": candidate_us,
                    "speedup_vs_exact_packed": control_us / candidate_us,
                    "quality": {
                        "random_vectors": args.vectors,
                        "top1_equal": sum(item["top1_equal"] for item in quality),
                        "maximum_absolute_logit_difference": max(
                            item["maximum_absolute_logit_difference"] for item in quality
                        ),
                        "mean_absolute_logit_difference": statistics.mean(
                            item["mean_absolute_logit_difference"] for item in quality
                        ),
                    },
                    "samples": samples,
                }
            )
        del qweight, scales
        torch.cuda.empty_cache()

    best_by_group = []
    for group_k in sorted({item["group_k"] for item in results}):
        best_by_group.append(
            max(
                (item for item in results if item["group_k"] == group_k),
                key=lambda item: item["speedup_vs_exact_packed"],
            )
        )
    payload = {
        "schema_version": "sm89-lossy-int8-lmhead-screen-v1",
        "status": "PASS",
        "scope": {
            "model": str(model_path),
            "gpu": torch.cuda.get_device_name(0),
            "weight_shape": [n, k],
            "control": "exact-packed BF16 lm-head, FP32 reduction, BF16 logits",
            "candidate": "symmetric groupwise INT8 weight-only, FP16 scales, FP32 reduction, BF16 logits",
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
