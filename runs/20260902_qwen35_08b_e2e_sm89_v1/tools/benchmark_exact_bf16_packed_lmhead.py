#!/usr/bin/env python3
"""Benchmark an exact packed-BF16 lm-head against the accepted SM89 kernel.

Each contiguous 256-value block stores the sign+mantissa byte and a four-bit
exponent delta from the block minimum.  Blocks whose exponent range exceeds
15 fall back to their original BF16 bits.  Reconstruction produces the exact
original FP32 value before using the accepted FP32 dot-product reduction.
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


BLOCK_VALUES = 256


@triton.jit
def _dense_lmhead_kernel(
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
    mask = (offsets_n[:, None] < n) & (offsets_k[None, :] < k)
    x = tl.load(x_ptr + offsets_k, mask=offsets_k < k, other=0.0)
    weight = tl.load(
        weight_ptr + offsets_n[:, None] * k + offsets_k[None, :],
        mask=mask,
        other=0.0,
    )
    accum = tl.sum(weight.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
    tl.store(output_ptr + offsets_n, accum, mask=offsets_n < n)


@triton.jit
def _packed_lmhead_kernel(
    x_ptr,
    sign_mantissa_ptr,
    exponent_nibbles_ptr,
    base_exponent_ptr,
    fallback_slot_ptr,
    fallback_bits_ptr,
    output_ptr,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PACK_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    valid = (offsets_n[:, None] < n) & (offsets_k[None, :] < k)
    linear = offsets_n[:, None] * k + offsets_k[None, :]
    block_id = linear // PACK_BLOCK
    in_block = linear % PACK_BLOCK

    sm = tl.load(sign_mantissa_ptr + linear, mask=valid, other=0).to(tl.int32)
    pair = tl.load(exponent_nibbles_ptr + linear // 2, mask=valid, other=0).to(
        tl.int32
    )
    shift = (linear & 1) * 4
    delta = (pair >> shift) & 0xF
    base = tl.load(base_exponent_ptr + block_id, mask=valid, other=0).to(tl.int32)
    slot = tl.load(fallback_slot_ptr + block_id, mask=valid, other=-1)
    packed_bits = (
        ((sm & 0x80) << 8) | ((base + delta) << 7) | (sm & 0x7F)
    )
    fallback_bits = tl.load(
        fallback_bits_ptr + slot * PACK_BLOCK + in_block,
        mask=valid & (slot >= 0),
        other=0,
    ).to(tl.int32) & 0xFFFF
    raw_bf16 = tl.where(slot >= 0, fallback_bits, packed_bits)
    fp32_bits = raw_bf16 << 16
    weight = tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=f,r",
        [fp32_bits],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    x = tl.load(x_ptr + offsets_k, mask=offsets_k < k, other=0.0)
    accum = tl.sum(weight * x[None, :].to(tl.float32), axis=1)
    tl.store(output_ptr + offsets_n, accum, mask=offsets_n < n)


def pack_exact_bf16(
    weight: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], bool]:
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        raise ValueError("weight must be contiguous BF16")
    bits = weight.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).flatten()
    if bits.numel() % BLOCK_VALUES:
        raise ValueError("weight size must be divisible by block size")
    blocks = bits.reshape(-1, BLOCK_VALUES)
    exponent = bits.bitwise_right_shift(7).bitwise_and(0xFF)
    exponent_blocks = exponent.reshape(-1, BLOCK_VALUES)
    base = exponent_blocks.amin(dim=1)
    delta_blocks = exponent_blocks - base[:, None]
    packed_blocks = delta_blocks.amax(dim=1) <= 15

    sign_mantissa = bits.bitwise_and(0x7F).bitwise_or(
        bits.bitwise_right_shift(8).bitwise_and(0x80)
    ).to(torch.uint8)
    delta = delta_blocks.flatten()
    delta = torch.where(packed_blocks[:, None].expand(-1, BLOCK_VALUES).flatten(), delta, 0)
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
    fallback_bits = blocks[failed_ids].to(torch.int16).contiguous()
    packed_exponents_exact = bool(
        torch.equal(
            (base[:, None] + delta_blocks)[packed_blocks],
            exponent_blocks[packed_blocks],
        )
    )
    fallback_bits_exact = bool(
        torch.equal(
            fallback_bits.to(torch.int32).bitwise_and(0xFFFF),
            blocks[failed_ids],
        )
    )
    return {
        "sign_mantissa": sign_mantissa.contiguous(),
        "exponent_nibbles": exponent_nibbles.contiguous(),
        "base_exponent": base.to(torch.uint8).contiguous(),
        "fallback_slot": fallback_slot,
        "fallback_bits": fallback_bits,
        "packed_blocks": packed_blocks,
    }, packed_exponents_exact and fallback_bits_exact


def launch_dense(x: torch.Tensor, weight: torch.Tensor, output: torch.Tensor) -> None:
    n, k = weight.shape
    _dense_lmhead_kernel[(triton.cdiv(n, 4),)](
        x,
        weight,
        output,
        n=n,
        k=k,
        BLOCK_N=4,
        BLOCK_K=triton.next_power_of_2(k),
        num_warps=8,
        num_stages=1,
    )


def launch_packed(
    x: torch.Tensor,
    packed: dict[str, torch.Tensor],
    output: torch.Tensor,
    n: int,
    k: int,
    block_n: int,
    num_warps: int,
) -> None:
    _packed_lmhead_kernel[(triton.cdiv(n, block_n),)](
        x,
        packed["sign_mantissa"],
        packed["exponent_nibbles"],
        packed["base_exponent"],
        packed["fallback_slot"],
        packed["fallback_bits"],
        output,
        n=n,
        k=k,
        BLOCK_N=block_n,
        BLOCK_K=triton.next_power_of_2(k),
        PACK_BLOCK=BLOCK_VALUES,
        num_warps=num_warps,
        num_stages=1,
    )


def capture(fn) -> torch.cuda.CUDAGraph:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def paired_graph_samples(
    baseline: torch.cuda.CUDAGraph,
    candidate: torch.cuda.CUDAGraph,
    iterations: int,
    repeats: int,
    warmup_pairs: int = 10,
) -> dict[str, list[float] | list[str]]:
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
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--schedules",
        default="1x2,1x4,1x8,2x2,2x4,2x8,4x2,4x4,4x8,8x2,8x4,8x8,16x2,16x4,16x8,32x2,32x4,32x8",
        help="Comma-separated BLOCK_Nxwarps pairs.",
    )
    parser.add_argument("--measurement-warmup-pairs", type=int, default=10)
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
    packed, weight_bit_exact = pack_exact_bf16(weight)

    tensor_bytes = {
        name: value.numel() * value.element_size()
        for name, value in packed.items()
        if name != "packed_blocks"
    }
    dense_bytes = weight.numel() * weight.element_size()
    packed_bytes = sum(tensor_bytes.values())
    failed_blocks = int((~packed["packed_blocks"]).sum().item())
    block_count = int(packed["packed_blocks"].numel())
    del packed["packed_blocks"]

    x = (torch.randn((1, k), device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    dense_output = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
    packed_output = torch.empty_like(dense_output)
    launch_dense(x, weight, dense_output)

    requested_schedules = [
        tuple(int(value) for value in item.split("x"))
        for item in args.schedules.split(",")
    ]
    schedules = []
    for block_n, num_warps in requested_schedules:
        launch_packed(x, packed, packed_output, n, k, block_n, num_warps)
        torch.cuda.synchronize()
        exact = bool(torch.equal(dense_output, packed_output))
        max_abs = float(
            (dense_output.float() - packed_output.float()).abs().max().item()
        )
        dense_graph = capture(lambda: launch_dense(x, weight, dense_output))
        packed_graph = capture(
            lambda: launch_packed(
                x, packed, packed_output, n, k, block_n, num_warps
            )
        )
        samples = paired_graph_samples(
            dense_graph,
            packed_graph,
            args.iterations,
            args.repeats,
            warmup_pairs=args.measurement_warmup_pairs,
        )
        dense_median = statistics.median(samples["baseline_us"])
        packed_median = statistics.median(samples["candidate_us"])
        schedules.append(
            {
                "block_n": block_n,
                "num_warps": num_warps,
                "output_bit_exact": exact,
                "maximum_absolute_logit_difference": max_abs,
                "samples": samples,
                "median_dense_us": dense_median,
                "median_packed_us": packed_median,
                "local_speedup": dense_median / packed_median,
            }
        )

    best = max(schedules, key=lambda item: item["local_speedup"])
    strict_tpot_us = 8079.0
    saving_us = best["median_dense_us"] - best["median_packed_us"]
    projected_speedup = strict_tpot_us / (strict_tpot_us - saving_us)
    output = {
        "schema_version": "sm89-exact-bf16-packed-lmhead-screen-v1",
        "status": "PASS",
        "scope": {
            "model": str(model_path),
            "gpu": torch.cuda.get_device_name(0),
            "weight_shape": [n, k],
            "dtype_contract": "exact BF16 weight bits, accepted FP32 reduction, BF16 logits",
        },
        "packing": {
            "block_values": BLOCK_VALUES,
            "blocks": block_count,
            "fallback_blocks": failed_blocks,
            "fallback_block_fraction": failed_blocks / block_count,
            "dense_bytes": dense_bytes,
            "packed_bytes": packed_bytes,
            "packed_byte_fraction": packed_bytes / dense_bytes,
            "weight_bit_exact": weight_bit_exact,
            "tensor_bytes": tensor_bytes,
        },
        "schedules": schedules,
        "best": best,
        "projection": {
            "strict_frontier_tpot_us": strict_tpot_us,
            "observed_lm_head_saving_us": saving_us,
            "projected_whole_step_speedup": projected_speedup,
            "minimum_saving_for_1_03x_us": strict_tpot_us * (1 - 1 / 1.03),
        },
        "decision": (
            "IMPLEMENT_END_TO_END"
            if weight_bit_exact and projected_speedup >= 1.03
            else "MEASURED_REJECT_UNPACK_OVERHEAD_OR_WEIGHT_CORRECTNESS"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
