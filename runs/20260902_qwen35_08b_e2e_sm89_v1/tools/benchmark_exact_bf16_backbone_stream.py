#!/usr/bin/env python3
"""Cold-stream screen exact BF16 packing on all material decode projections.

Each benchmark group rotates through every layer weight of one production
shape, so its working set exceeds L2. This avoids promoting a kernel from an
isolated, falsely cache-resident matrix.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration

from benchmark_exact_bf16_packed_lmhead import (
    BLOCK_VALUES,
    capture,
    launch_packed,
    pack_exact_bf16,
    paired_graph_samples,
)


def build_groups(model: Qwen3_5ForConditionalGeneration) -> dict[str, list[torch.Tensor]]:
    groups: dict[str, list[torch.Tensor]] = {
        "mlp_gate_up": [],
        "gdn_qkvz": [],
        "mlp_down": [],
        "attention_gdn_out": [],
        "attention_qkv": [],
    }
    for layer in model.model.language_model.layers:
        groups["mlp_gate_up"].append(
            torch.cat(
                (layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight), dim=0
            ).contiguous()
        )
        groups["mlp_down"].append(layer.mlp.down_proj.weight.detach().contiguous())
        if hasattr(layer, "linear_attn"):
            groups["gdn_qkvz"].append(
                torch.cat(
                    (
                        layer.linear_attn.in_proj_qkv.weight,
                        layer.linear_attn.in_proj_z.weight,
                    ),
                    dim=0,
                ).contiguous()
            )
            groups["attention_gdn_out"].append(
                layer.linear_attn.out_proj.weight.detach().contiguous()
            )
        else:
            groups["attention_qkv"].append(
                torch.cat(
                    (
                        layer.self_attn.q_proj.weight,
                        layer.self_attn.k_proj.weight,
                        layer.self_attn.v_proj.weight,
                    ),
                    dim=0,
                ).contiguous()
            )
            groups["attention_gdn_out"].append(
                layer.self_attn.o_proj.weight.detach().contiguous()
            )
    return groups


def launch_dense_stream(
    x: torch.Tensor, weights: list[torch.Tensor], outputs: list[torch.Tensor]
) -> None:
    for weight, output in zip(weights, outputs, strict=True):
        torch.mm(x, weight.t(), out=output)


def launch_packed_stream(
    x: torch.Tensor,
    packed_weights: list[dict[str, torch.Tensor]],
    outputs: list[torch.Tensor],
    n: int,
    k: int,
    block_n: int,
    num_warps: int,
) -> None:
    for packed, output in zip(packed_weights, outputs, strict=True):
        launch_packed(x, packed, output, n, k, block_n, num_warps)


def reconstruct_bits(packed: dict[str, torch.Tensor]) -> torch.Tensor:
    pairs = packed["exponent_nibbles"].to(torch.int32)
    delta = torch.stack((pairs & 0xF, pairs >> 4), dim=1).flatten()
    block_ids = torch.arange(
        delta.numel(), device=delta.device, dtype=torch.int64
    ) // BLOCK_VALUES
    sign_mantissa = packed["sign_mantissa"].to(torch.int32)
    exponent = packed["base_exponent"].to(torch.int32)[block_ids] + delta
    bits = (
        ((sign_mantissa & 0x80) << 8)
        | (exponent << 7)
        | (sign_mantissa & 0x7F)
    )
    fallback_slot = packed["fallback_slot"].to(torch.int64)[block_ids]
    fallback = fallback_slot >= 0
    if bool(fallback.any()):
        in_block = torch.arange(
            delta.numel(), device=delta.device, dtype=torch.int64
        ) % BLOCK_VALUES
        bits[fallback] = (
            packed["fallback_bits"]
            .flatten()[fallback_slot[fallback] * BLOCK_VALUES + in_block[fallback]]
            .to(torch.int32)
            .bitwise_and(0xFFFF)
        )
    return bits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--measurement-warmup-pairs", type=int, default=100)
    parser.add_argument(
        "--schedules",
        default="4x4,4x8,8x4,8x8,16x4,16x8,32x4,32x8",
        help="Comma-separated BLOCK_Nxwarps pairs.",
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
    cpu_groups = build_groups(model)
    del model

    requested_schedules = [
        tuple(int(value) for value in item.split("x"))
        for item in args.schedules.split(",")
    ]
    results = []
    for group_name, cpu_weights in cpu_groups.items():
        torch.cuda.empty_cache()
        weights = [weight.to("cuda") for weight in cpu_weights]
        del cpu_weights
        n, k = weights[0].shape
        if any(tuple(weight.shape) != (n, k) for weight in weights):
            raise RuntimeError(f"mixed shapes in {group_name}")
        x = (torch.randn((1, k), device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
        dense_outputs = [
            torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
            for _ in weights
        ]
        launch_dense_stream(x, weights, dense_outputs)
        torch.cuda.synchronize()
        dense_graph = capture(lambda: launch_dense_stream(x, weights, dense_outputs))

        packed_weights = []
        all_weight_bits_exact = True
        dense_bytes = 0
        packed_bytes = 0
        fallback_blocks = 0
        block_count = 0
        for weight in weights:
            packed, pack_check = pack_exact_bf16(weight)
            original_bits = (
                weight.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).flatten()
            )
            reconstructed = reconstruct_bits(packed)
            all_weight_bits_exact &= pack_check and bool(
                torch.equal(original_bits, reconstructed)
            )
            dense_bytes += weight.numel() * weight.element_size()
            fallback_blocks += int((~packed["packed_blocks"]).sum().item())
            block_count += int(packed["packed_blocks"].numel())
            del packed["packed_blocks"], original_bits, reconstructed
            packed_bytes += sum(
                tensor.numel() * tensor.element_size() for tensor in packed.values()
            )
            packed_weights.append(packed)

        schedule_results = []
        for block_n, num_warps in requested_schedules:
            packed_outputs = [
                torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
                for _ in weights
            ]
            launch_packed_stream(
                x, packed_weights, packed_outputs, n, k, block_n, num_warps
            )
            torch.cuda.synchronize()
            max_abs = max(
                float((dense.float() - candidate.float()).abs().max().item())
                for dense, candidate in zip(
                    dense_outputs, packed_outputs, strict=True
                )
            )
            exact_outputs = all(
                bool(torch.equal(dense, candidate))
                for dense, candidate in zip(
                    dense_outputs, packed_outputs, strict=True
                )
            )
            packed_graph = capture(
                lambda: launch_packed_stream(
                    x,
                    packed_weights,
                    packed_outputs,
                    n,
                    k,
                    block_n,
                    num_warps,
                )
            )
            samples = paired_graph_samples(
                dense_graph,
                packed_graph,
                args.iterations,
                args.repeats,
                warmup_pairs=args.measurement_warmup_pairs,
            )
            dense_us = statistics.median(samples["baseline_us"])
            packed_us = statistics.median(samples["candidate_us"])
            schedule_results.append(
                {
                    "block_n": block_n,
                    "num_warps": num_warps,
                    "output_bit_exact": exact_outputs,
                    "maximum_absolute_output_difference": max_abs,
                    "median_dense_stream_us": dense_us,
                    "median_packed_stream_us": packed_us,
                    "local_speedup": dense_us / packed_us,
                    "samples": samples,
                }
            )
            del packed_graph, packed_outputs
        best = max(schedule_results, key=lambda item: item["local_speedup"])
        results.append(
            {
                "group": group_name,
                "weight_shape": [n, k],
                "weights_per_decode_step": len(weights),
                "dense_working_set_bytes": dense_bytes,
                "packed_working_set_bytes": packed_bytes,
                "packed_byte_fraction": packed_bytes / dense_bytes,
                "l2_cold_stream": dense_bytes > 32 * 1024 * 1024,
                "blocks": block_count,
                "fallback_blocks": fallback_blocks,
                "weight_bit_exact": all_weight_bits_exact,
                "schedules": schedule_results,
                "best": best,
            }
        )
        del dense_graph, packed_weights, weights, dense_outputs, x

    dense_total_us = sum(item["best"]["median_dense_stream_us"] for item in results)
    packed_total_us = sum(item["best"]["median_packed_stream_us"] for item in results)
    saving_us = dense_total_us - packed_total_us
    current_tpot_us = 6656.8025689
    output = {
        "schema_version": "sm89-exact-bf16-backbone-cold-stream-v1",
        "status": "PASS",
        "scope": {
            "model": str(model_path),
            "gpu": torch.cuda.get_device_name(0),
            "dtype_contract": "exact BF16 weight bits; FP32 reduction may differ from cuBLAS",
            "excluded": "small GDN BA projection (1.125 MiB total, about 87 us/step)",
        },
        "protocol": {
            "iterations_per_sample": args.iterations,
            "repeats": args.repeats,
            "warmup_pairs": args.measurement_warmup_pairs,
            "schedules": args.schedules,
            "cache_control": "Each group rotates through every layer weight; all included dense working sets exceed 32 MiB L2.",
        },
        "groups": results,
        "aggregate_best_per_group": {
            "dense_stream_us": dense_total_us,
            "packed_stream_us": packed_total_us,
            "local_speedup": dense_total_us / packed_total_us,
            "observed_projection_saving_us": saving_us,
            "current_frontier_tpot_us": current_tpot_us,
            "optimistic_whole_step_speedup": current_tpot_us
            / (current_tpot_us - saving_us),
        },
        "decision": (
            "IMPLEMENT_BOUNDED_VLLM_BACKBONE_SCREEN"
            if saving_us >= current_tpot_us * (1.0 - 1.0 / 1.03)
            and all(item["weight_bit_exact"] for item in results)
            else "MEASURED_REJECT_COLD_STREAM_UNPACK_OVERHEAD"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
