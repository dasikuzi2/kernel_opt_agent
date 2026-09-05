#!/usr/bin/env python3
"""Measure exact base-plus-delta packing feasibility for active decode weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration


BLOCK_VALUES = 256
BASELINE_US = {
    "lm_head": 2057.5395,
    "gdn_qkvz": 18 * 80.463,
    "gdn_ba": 18 * 4.848,
    "gdn_out": 18 * 24.063,
    "attention_qkv": 6 * 52.357,
    "attention_out": 6 * 24.063,
    "mlp_gate_up": 24 * 70.969,
    "mlp_down": 24 * 37.702,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def group_for_parameter(name: str) -> str | None:
    if name.endswith("embed_tokens.weight"):
        return "lm_head"
    if ".mlp.gate_proj.weight" in name or ".mlp.up_proj.weight" in name:
        return "mlp_gate_up"
    if ".mlp.down_proj.weight" in name:
        return "mlp_down"
    if ".linear_attn.in_proj_qkv.weight" in name or ".linear_attn.in_proj_z.weight" in name:
        return "gdn_qkvz"
    if ".linear_attn.in_proj_b.weight" in name or ".linear_attn.in_proj_a.weight" in name:
        return "gdn_ba"
    if ".linear_attn.out_proj.weight" in name:
        return "gdn_out"
    if any(f".self_attn.{part}_proj.weight" in name for part in ("q", "k", "v")):
        return "attention_qkv"
    if ".self_attn.o_proj.weight" in name:
        return "attention_out"
    return None


def tensor_stats(weight: torch.Tensor) -> dict[str, int]:
    weight = weight.detach().contiguous()
    if weight.dtype != torch.bfloat16:
        raise RuntimeError(f"expected BF16, got {weight.dtype}")
    values = weight.numel()
    if values % BLOCK_VALUES:
        raise RuntimeError(f"{values} values are not divisible by {BLOCK_VALUES}")
    bits = weight.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).flatten()
    exponent = bits.bitwise_right_shift(7).bitwise_and(0xFF)
    blocks = exponent.reshape(-1, BLOCK_VALUES)
    passing = (blocks.amax(dim=1) - blocks.amin(dim=1)) <= 15
    return {
        "values": values,
        "blocks": passing.numel(),
        "passing_blocks": int(passing.sum().item()),
        "fallback_blocks": int((~passing).sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--current-frontier-tpot-ms", type=float, default=6.6568025689)
    args = parser.parse_args()

    model_path = args.model.resolve()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).eval()
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensors": 0, "values": 0, "blocks": 0, "passing_blocks": 0, "fallback_blocks": 0}
    )
    shapes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, weight in model.named_parameters():
        group = group_for_parameter(name)
        if group is None:
            continue
        stats = tensor_stats(weight)
        totals[group]["tensors"] += 1
        for key, value in stats.items():
            totals[group][key] += value
        shapes[group]["x".join(str(value) for value in weight.shape)] += 1

    groups = []
    for group, stats in totals.items():
        values = stats["values"]
        blocks = stats["blocks"]
        fallback_blocks = stats["fallback_blocks"]
        dense_bytes = values * 2
        packed_bytes = (
            values
            + values // 2
            + blocks
            + blocks * 4
            + fallback_blocks * BLOCK_VALUES * 2
        )
        byte_fraction = packed_bytes / dense_bytes
        baseline_us = BASELINE_US[group]
        optimistic_saving_us = baseline_us * (1.0 - byte_fraction)
        groups.append(
            {
                "group": group,
                **stats,
                "component_shapes": dict(shapes[group]),
                "dense_bytes": dense_bytes,
                "packed_bytes": packed_bytes,
                "packed_byte_fraction": byte_fraction,
                "optimistic_local_speedup": 1.0 / byte_fraction,
                "profiled_baseline_us_per_decode_step": baseline_us,
                "optimistic_saving_us_per_decode_step": optimistic_saving_us,
            }
        )
    groups.sort(key=lambda item: item["optimistic_saving_us_per_decode_step"], reverse=True)

    backbone = [item for item in groups if item["group"] != "lm_head"]
    backbone_dense = sum(item["dense_bytes"] for item in backbone)
    backbone_packed = sum(item["packed_bytes"] for item in backbone)
    optimistic_backbone_saving_us = sum(
        item["optimistic_saving_us_per_decode_step"] for item in backbone
    )
    current_tpot_us = args.current_frontier_tpot_ms * 1000
    output = {
        "schema_version": "sm89-exact-bf16-backbone-compression-feasibility-v1",
        "status": "PASS",
        "scope": {
            "model": str(model_path),
            "config_sha256": sha256(model_path / "config.json"),
            "dtype": "bfloat16",
            "block_values": BLOCK_VALUES,
            "active_path": "text batch-1 decode projection weights",
        },
        "format": "8-bit exact sign+mantissa, 4-bit block-local exponent delta, 8-bit base and int32 fallback slot per block, dense BF16 fallback payload",
        "groups": groups,
        "aggregate_backbone": {
            "dense_bytes": backbone_dense,
            "packed_bytes": backbone_packed,
            "packed_byte_fraction": backbone_packed / backbone_dense,
            "optimistic_projection_saving_us_per_decode_step": optimistic_backbone_saving_us,
            "current_frontier_tpot_us": current_tpot_us,
            "optimistic_whole_step_speedup": current_tpot_us
            / (current_tpot_us - optimistic_backbone_saving_us),
        },
        "materiality_gate": {
            "minimum_whole_step_speedup": 1.03,
            "passes": optimistic_backbone_saving_us
            >= current_tpot_us * (1.0 - 1.0 / 1.03),
            "warning": "The upper bound assumes latency scales with bytes and charges no unpack instructions. GPU stream microbenchmarks must rotate through all layer weights to defeat false L2-hot wins.",
        },
        "decision": "RUN_COLD_STREAM_GPU_SCREEN",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
