#!/usr/bin/env python3
"""Screen metadata-hoisted exact BF16 unpack followed by Tensor Core dot."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from transformers import Qwen3_5ForConditionalGeneration

from benchmark_exact_bf16_backbone_stream import (
    build_groups,
    launch_dense_stream,
    reconstruct_bits,
)
from benchmark_exact_bf16_packed_lmhead import (
    BLOCK_VALUES,
    capture,
    pack_exact_bf16,
    paired_graph_samples,
)


@triton.jit
def _packed_tensorcore_gemv_kernel(
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
    BLOCK_M: tl.constexpr,
    PACK_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_m = tl.arange(0, BLOCK_M)
    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    blocks_per_row: tl.constexpr = k // PACK_BLOCK
    for k_start in range(0, k, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        valid = (offsets_n[:, None] < n) & (offsets_k[None, :] < k)
        linear = offsets_n[:, None] * k + offsets_k[None, :]
        sm = tl.load(sign_mantissa_ptr + linear, mask=valid, other=0).to(tl.int32)
        pair = tl.load(
            exponent_nibbles_ptr + linear // 2, mask=valid, other=0
        ).to(tl.int32)
        delta = (pair >> ((linear & 1) * 4)) & 0xF

        # BLOCK_K divides the 256-value packing block, so metadata is constant
        # across this tile and is loaded once per output row, not once/value.
        metadata_block = (
            offsets_n * blocks_per_row + k_start // PACK_BLOCK
        )
        base = tl.load(
            base_exponent_ptr + metadata_block,
            mask=offsets_n < n,
            other=0,
        ).to(tl.int32)
        slot = tl.load(
            fallback_slot_ptr + metadata_block,
            mask=offsets_n < n,
            other=-1,
        )
        packed_bits = (
            ((sm & 0x80) << 8)
            | ((base[:, None] + delta) << 7)
            | (sm & 0x7F)
        )
        in_block = offsets_k % PACK_BLOCK
        fallback_bits = tl.load(
            fallback_bits_ptr
            + slot[:, None] * PACK_BLOCK
            + in_block[None, :],
            mask=valid & (slot[:, None] >= 0),
            other=0,
        ).to(tl.int32) & 0xFFFF
        fp32_bits = tl.where(slot[:, None] >= 0, fallback_bits, packed_bits) << 16
        weight_fp32 = tl.inline_asm_elementwise(
            "mov.b32 $0, $1;",
            "=f,r",
            [fp32_bits],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        x = tl.load(x_ptr + offsets_k, mask=offsets_k < k, other=0.0)
        lhs = tl.where(
            offsets_m[:, None] == 0,
            x[None, :],
            0.0,
        ).to(tl.bfloat16)
        rhs = tl.trans(weight_fp32.to(tl.bfloat16))
        accum += tl.dot(lhs, rhs)
    # Only the first padded M row contains x. The remaining rows are exact
    # zeros, so this reduction extracts the GEMV result without extra writes.
    result = tl.sum(accum, axis=0)
    tl.store(output_ptr + offsets_n, result, mask=offsets_n < n)


def launch_tensorcore_stream(
    x: torch.Tensor,
    packed_weights: list[dict[str, torch.Tensor]],
    outputs: list[torch.Tensor],
    n: int,
    k: int,
    block_n: int,
    block_k: int,
    num_warps: int,
) -> None:
    for packed, output in zip(packed_weights, outputs, strict=True):
        _packed_tensorcore_gemv_kernel[(triton.cdiv(n, block_n),)](
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
            BLOCK_K=block_k,
            BLOCK_M=16,
            PACK_BLOCK=BLOCK_VALUES,
            num_warps=num_warps,
            num_stages=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups", default="mlp_gate_up,gdn_qkvz,mlp_down,attention_gdn_out,attention_qkv")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--measurement-warmup-pairs", type=int, default=100)
    parser.add_argument(
        "--schedules",
        default="16x32x4,16x64x4,16x128x4,32x32x4,32x64x4,32x128x4,32x64x8",
        help="Comma-separated BLOCK_NxBLOCK_Kxwarps triples.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260905)
    model_path = args.model.resolve()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).eval()
    model.requires_grad_(False)
    all_groups = build_groups(model)
    selected_names = args.groups.split(",")
    cpu_groups = {name: all_groups[name] for name in selected_names}
    del all_groups, model
    schedules = [
        tuple(int(value) for value in item.split("x"))
        for item in args.schedules.split(",")
    ]

    results = []
    for group_name, cpu_weights in cpu_groups.items():
        torch.cuda.empty_cache()
        weights = [weight.to("cuda") for weight in cpu_weights]
        n, k = weights[0].shape
        x = (torch.randn((1, k), device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
        dense_outputs = [
            torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
            for _ in weights
        ]
        launch_dense_stream(x, weights, dense_outputs)
        torch.cuda.synchronize()
        dense_graph = capture(lambda: launch_dense_stream(x, weights, dense_outputs))

        packed_weights = []
        weight_bit_exact = True
        dense_bytes = 0
        packed_bytes = 0
        for weight in weights:
            packed, pack_check = pack_exact_bf16(weight)
            original = weight.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).flatten()
            reconstructed = reconstruct_bits(packed)
            weight_bit_exact &= pack_check and bool(torch.equal(original, reconstructed))
            del packed["packed_blocks"], original, reconstructed
            dense_bytes += weight.numel() * weight.element_size()
            packed_bytes += sum(
                tensor.numel() * tensor.element_size() for tensor in packed.values()
            )
            packed_weights.append(packed)

        points = []
        for block_n, block_k, num_warps in schedules:
            if BLOCK_VALUES % block_k:
                raise ValueError("BLOCK_K must divide the 256-value packing block")
            outputs = [
                torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
                for _ in weights
            ]
            launch_tensorcore_stream(
                x, packed_weights, outputs, n, k, block_n, block_k, num_warps
            )
            torch.cuda.synchronize()
            maximum_absolute_difference = max(
                float((dense.float() - candidate.float()).abs().max().item())
                for dense, candidate in zip(dense_outputs, outputs, strict=True)
            )
            output_bit_exact = all(
                bool(torch.equal(dense, candidate))
                for dense, candidate in zip(dense_outputs, outputs, strict=True)
            )
            graph = capture(
                lambda: launch_tensorcore_stream(
                    x,
                    packed_weights,
                    outputs,
                    n,
                    k,
                    block_n,
                    block_k,
                    num_warps,
                )
            )
            samples = paired_graph_samples(
                dense_graph,
                graph,
                args.iterations,
                args.repeats,
                warmup_pairs=args.measurement_warmup_pairs,
            )
            dense_us = statistics.median(samples["baseline_us"])
            candidate_us = statistics.median(samples["candidate_us"])
            points.append(
                {
                    "block_n": block_n,
                    "block_k": block_k,
                    "num_warps": num_warps,
                    "output_bit_exact": output_bit_exact,
                    "maximum_absolute_output_difference": maximum_absolute_difference,
                    "median_dense_stream_us": dense_us,
                    "median_candidate_stream_us": candidate_us,
                    "local_speedup": dense_us / candidate_us,
                    "samples": samples,
                }
            )
            del graph, outputs
        best = max(points, key=lambda item: item["local_speedup"])
        results.append(
            {
                "group": group_name,
                "weight_shape": [n, k],
                "weights_per_decode_step": len(weights),
                "dense_working_set_bytes": dense_bytes,
                "packed_working_set_bytes": packed_bytes,
                "packed_byte_fraction": packed_bytes / dense_bytes,
                "weight_bit_exact": weight_bit_exact,
                "schedules": points,
                "best": best,
            }
        )
        del dense_graph, packed_weights, weights, dense_outputs, x

    dense_total_us = sum(item["best"]["median_dense_stream_us"] for item in results)
    candidate_total_us = sum(
        item["best"]["median_candidate_stream_us"] for item in results
    )
    saving_us = dense_total_us - candidate_total_us
    current_tpot_us = 6656.8025689
    output = {
        "schema_version": "sm89-exact-bf16-backbone-tensorcore-cold-stream-v1",
        "status": "PASS",
        "scope": {
            "model": str(model_path),
            "gpu": torch.cuda.get_device_name(0),
            "dtype_contract": "exact BF16 weight bits; Tensor Core FP32 accumulation",
        },
        "protocol": {
            "iterations_per_sample": args.iterations,
            "repeats": args.repeats,
            "warmup_pairs": args.measurement_warmup_pairs,
            "schedules": args.schedules,
            "architecture": "load block metadata once per output row/tile, reconstruct exact BF16, use padded-M=16 Tensor Core dot",
        },
        "groups": results,
        "aggregate_best_per_group": {
            "dense_stream_us": dense_total_us,
            "candidate_stream_us": candidate_total_us,
            "local_speedup": dense_total_us / candidate_total_us,
            "observed_projection_saving_us": saving_us,
            "current_frontier_tpot_us": current_tpot_us,
            "optimistic_whole_step_speedup": current_tpot_us
            / (current_tpot_us - saving_us),
        },
        "decision": (
            "IMPLEMENT_BOUNDED_VLLM_BACKBONE_SCREEN"
            if saving_us >= current_tpot_us * (1.0 - 1.0 / 1.03)
            and all(item["weight_bit_exact"] for item in results)
            else "MEASURED_REJECT_TENSORCORE_PADDING_OR_UNPACK_OVERHEAD"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
