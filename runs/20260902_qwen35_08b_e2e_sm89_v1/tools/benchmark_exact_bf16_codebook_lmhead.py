#!/usr/bin/env python3
"""Screen an exact 12-bit BF16 codebook lm-head on SM89.

The 4096 most common BF16 bit patterns are stored once as their exact FP32
widening. Two dictionary indices occupy three bytes. A 128-value block that
contains any rarer pattern falls back to the original BF16 payload, preserving
all weight values exactly.
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
    launch_dense,
    paired_graph_samples,
)


CODEBOOK_SIZE = 4096
FALLBACK_BLOCK = 128


@triton.jit
def _codebook_lmhead_kernel(
    x_ptr,
    packed_indices_ptr,
    palette_ptr,
    fallback_slot_ptr,
    fallback_weight_ptr,
    output_ptr,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FALLBACK_BLOCK_VALUES: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    valid = (offsets_n[:, None] < n) & (offsets_k[None, :] < k)
    linear = offsets_n[:, None] * k + offsets_k[None, :]

    pair = linear // 2
    packed_offset = pair * 3
    byte0 = tl.load(packed_indices_ptr + packed_offset, mask=valid, other=0).to(
        tl.int32
    )
    byte1 = tl.load(
        packed_indices_ptr + packed_offset + 1, mask=valid, other=0
    ).to(tl.int32)
    byte2 = tl.load(
        packed_indices_ptr + packed_offset + 2, mask=valid, other=0
    ).to(tl.int32)
    even_index = byte0 | ((byte1 & 0xF) << 8)
    odd_index = (byte1 >> 4) | (byte2 << 4)
    codebook_index = tl.where((linear & 1) == 0, even_index, odd_index)
    dictionary_weight = tl.load(
        palette_ptr + codebook_index, mask=valid, other=0.0
    )

    block_id = linear // FALLBACK_BLOCK_VALUES
    in_block = linear % FALLBACK_BLOCK_VALUES
    slot = tl.load(fallback_slot_ptr + block_id, mask=valid, other=-1)
    fallback_weight = tl.load(
        fallback_weight_ptr + slot * FALLBACK_BLOCK_VALUES + in_block,
        mask=valid & (slot >= 0),
        other=0.0,
    ).to(tl.float32)
    weight = tl.where(slot >= 0, fallback_weight, dictionary_weight)
    x = tl.load(x_ptr + offsets_k, mask=offsets_k < k, other=0.0)
    accum = tl.sum(weight * x[None, :].to(tl.float32), axis=1)
    tl.store(output_ptr + offsets_n, accum, mask=offsets_n < n)


def pack_codebook(weight: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict]:
    bits = weight.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).flatten()
    counts = torch.bincount(bits, minlength=65536)
    patterns = torch.argsort(counts, descending=True)[:CODEBOOK_SIZE]
    lookup = torch.full((65536,), -1, dtype=torch.int32, device=weight.device)
    lookup[patterns] = torch.arange(
        CODEBOOK_SIZE, dtype=torch.int32, device=weight.device
    )
    indices = lookup[bits]
    block_indices = indices.reshape(-1, FALLBACK_BLOCK)
    packed_blocks = (block_indices >= 0).all(dim=1)
    safe_indices = torch.where(indices >= 0, indices, 0).reshape(-1, 2)
    byte0 = safe_indices[:, 0] & 0xFF
    byte1 = (safe_indices[:, 0] >> 8) | ((safe_indices[:, 1] & 0xF) << 4)
    byte2 = safe_indices[:, 1] >> 4
    packed_indices = torch.stack((byte0, byte1, byte2), dim=1).to(
        torch.uint8
    ).flatten().contiguous()

    # Index the palette with the same rank mapping and widen BF16 exactly.
    palette_bits = patterns.to(torch.int16)
    palette = palette_bits.view(torch.bfloat16).float().contiguous()
    failed_ids = torch.nonzero(~packed_blocks, as_tuple=False).flatten()
    fallback_slot = torch.full(
        (packed_blocks.numel(),), -1, dtype=torch.int32, device=weight.device
    )
    fallback_slot[failed_ids] = torch.arange(
        failed_ids.numel(), dtype=torch.int32, device=weight.device
    )
    fallback_weight = (
        weight.flatten()
        .reshape(-1, FALLBACK_BLOCK)[failed_ids]
        .contiguous()
    )

    bits_blocks = bits.reshape(-1, FALLBACK_BLOCK)
    dictionary_exact = True
    for start in range(0, packed_blocks.numel(), 4096):
        end = min(start + 4096, packed_blocks.numel())
        selected_blocks = packed_blocks[start:end]
        if bool(selected_blocks.any()):
            decoded = patterns[
                block_indices[start:end][selected_blocks].to(torch.int64)
            ]
            if not torch.equal(decoded, bits_blocks[start:end][selected_blocks]):
                dictionary_exact = False
                break
    fallback_exact = bool(
        torch.equal(
            fallback_weight.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF),
            bits_blocks[failed_ids],
        )
    )

    covered_values = int((indices >= 0).sum().item())
    stats = {
        "codebook_size": CODEBOOK_SIZE,
        "index_bits": 12,
        "fallback_block_values": FALLBACK_BLOCK,
        "blocks": int(packed_blocks.numel()),
        "fallback_blocks": int(failed_ids.numel()),
        "covered_value_fraction": covered_values / indices.numel(),
        "passing_block_fraction": float(packed_blocks.float().mean().item()),
        "weight_bit_exact": dictionary_exact and fallback_exact,
    }
    return {
        "packed_indices": packed_indices,
        "palette": palette,
        "fallback_slot": fallback_slot,
        "fallback_weight": fallback_weight,
    }, stats


def launch_codebook(
    x: torch.Tensor,
    packed: dict[str, torch.Tensor],
    output: torch.Tensor,
    n: int,
    k: int,
    block_n: int,
    num_warps: int,
) -> None:
    _codebook_lmhead_kernel[(triton.cdiv(n, block_n),)](
        x,
        packed["packed_indices"],
        packed["palette"],
        packed["fallback_slot"],
        packed["fallback_weight"],
        output,
        n=n,
        k=k,
        BLOCK_N=block_n,
        BLOCK_K=triton.next_power_of_2(k),
        FALLBACK_BLOCK_VALUES=FALLBACK_BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    packed, packing_stats = pack_codebook(weight)
    tensor_bytes = {
        name: value.numel() * value.element_size() for name, value in packed.items()
    }
    dense_bytes = weight.numel() * weight.element_size()
    packed_bytes = sum(tensor_bytes.values())

    x = (torch.randn((1, k), device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    dense_output = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
    codebook_output = torch.empty_like(dense_output)
    launch_dense(x, weight, dense_output)
    schedules = []
    for block_n in (1, 2, 4, 8, 16):
        for num_warps in (2, 4, 8):
            launch_codebook(x, packed, codebook_output, n, k, block_n, num_warps)
            torch.cuda.synchronize()
            exact = bool(torch.equal(dense_output, codebook_output))
            max_abs = float(
                (dense_output.float() - codebook_output.float()).abs().max().item()
            )
            dense_graph = capture(lambda: launch_dense(x, weight, dense_output))
            codebook_graph = capture(
                lambda: launch_codebook(
                    x, packed, codebook_output, n, k, block_n, num_warps
                )
            )
            samples = paired_graph_samples(
                dense_graph, codebook_graph, args.iterations, args.repeats
            )
            dense_median = statistics.median(samples["baseline_us"])
            candidate_median = statistics.median(samples["candidate_us"])
            schedules.append(
                {
                    "block_n": block_n,
                    "num_warps": num_warps,
                    "output_bit_exact": exact,
                    "maximum_absolute_logit_difference": max_abs,
                    "samples": samples,
                    "median_dense_us": dense_median,
                    "median_codebook_us": candidate_median,
                    "local_speedup": dense_median / candidate_median,
                }
            )

    best = max(schedules, key=lambda item: item["local_speedup"])
    strict_tpot_us = 8079.0
    saving_us = best["median_dense_us"] - best["median_codebook_us"]
    projected_speedup = strict_tpot_us / (strict_tpot_us - saving_us)
    result = {
        "schema_version": "sm89-exact-bf16-codebook-lmhead-screen-v1",
        "status": "PASS",
        "scope": {
            "model": str(model_path),
            "gpu": torch.cuda.get_device_name(0),
            "weight_shape": [n, k],
            "dtype_contract": "exact BF16 codebook and fallback weights, FP32 reduction, BF16 logits",
        },
        "packing": {
            **packing_stats,
            "dense_bytes": dense_bytes,
            "packed_bytes": packed_bytes,
            "packed_byte_fraction": packed_bytes / dense_bytes,
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
            if best["output_bit_exact"] and projected_speedup >= 1.03
            else "MEASURED_REJECT_LOOKUP_OR_PACKING_OVERHEAD"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
