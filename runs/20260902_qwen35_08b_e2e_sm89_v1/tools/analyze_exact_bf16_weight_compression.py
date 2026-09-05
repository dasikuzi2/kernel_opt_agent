#!/usr/bin/env python3
"""Screen lossless BF16 bit packing before writing a GPU decode kernel.

BF16 stores one sign bit, eight exponent bits and seven mantissa bits.  This
tool measures whether the exponent field of the real lm-head weights can be
represented by a smaller exact index.  It also evaluates block-local
base-plus-delta layouts with an exact dense fallback for blocks whose exponent
range is too wide.  No floating-point value is changed by either layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def entropy_from_counts(counts: torch.Tensor) -> float:
    counts = counts[counts > 0].double()
    probabilities = counts / counts.sum()
    return float((-(probabilities * probabilities.log2())).sum().item())


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), q).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-sizes", default="64,128,256,512,1024")
    parser.add_argument("--strict-frontier-tpot-ms", type=float, default=8.079)
    parser.add_argument("--lm-head-ms", type=float, default=2.0575)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model_path = args.model.resolve()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).eval().to("cuda")
    weight = model.lm_head.weight.detach()
    weight_shape = list(weight.shape)
    if weight.dtype != torch.bfloat16:
        raise RuntimeError(f"expected BF16 lm-head, got {weight.dtype}")
    # Keep only the tied output matrix alive before allocating bit-field views.
    weight = weight.contiguous()
    del model
    torch.cuda.empty_cache()

    bits = weight.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).flatten()
    del weight
    exponent = bits.bitwise_right_shift(7).bitwise_and(0xFF)
    sign_mantissa = bits.bitwise_and(0x7F).bitwise_or(
        bits.bitwise_right_shift(8).bitwise_and(0x80)
    )
    full_counts = torch.bincount(bits, minlength=65536).cpu()
    exponent_counts = torch.bincount(exponent, minlength=256).cpu()
    sign_mantissa_counts = torch.bincount(sign_mantissa, minlength=256).cpu()
    del sign_mantissa

    nonzero_exponents = torch.nonzero(exponent_counts, as_tuple=False).flatten()
    exponent_values = [int(value) for value in nonzero_exponents.tolist()]
    exponent_histogram = {
        str(value): int(exponent_counts[value].item()) for value in exponent_values
    }
    exponent_index_bits = math.ceil(math.log2(max(1, len(exponent_values))))
    total_values = int(exponent.numel())
    global_palette_bits = (
        total_values * (8 + exponent_index_bits) + len(exponent_values) * 8
    )
    global_palette_byte_fraction = global_palette_bits / (total_values * 16)

    ranked_patterns = torch.argsort(full_counts, descending=True)
    codebook_layouts = []
    for codebook_size in (256, 1024, 4096):
        index_bits = int(math.log2(codebook_size))
        selected = ranked_patterns[:codebook_size].to(device="cuda")
        covered_lookup = torch.zeros(65536, dtype=torch.bool, device="cuda")
        covered_lookup[selected] = True
        covered_values = covered_lookup[bits]
        covered_value_fraction = float(covered_values.float().mean().item())
        for block_size in (64, 128, 256, 512, 1024):
            block_count = total_values // block_size
            packed_blocks = covered_values.reshape(-1, block_size).all(dim=1)
            passing_blocks = int(packed_blocks.sum().item())
            fallback_blocks = block_count - passing_blocks
            # Keep a fixed-width packed-index array for direct addressing and
            # one int32 fallback slot per block. Failed blocks additionally
            # retain their dense BF16 payload. Bit padding and lookup work are
            # deliberately free in this optimistic gate.
            index_bytes = math.ceil(total_values * index_bits / 8)
            total_bytes = (
                index_bytes
                + codebook_size * 2
                + block_count * 4
                + fallback_blocks * block_size * 2
            )
            byte_fraction = total_bytes / (total_values * 2)
            saving_ms = args.lm_head_ms * (1.0 - byte_fraction)
            whole_step_speedup = args.strict_frontier_tpot_ms / (
                args.strict_frontier_tpot_ms - saving_ms
            )
            codebook_layouts.append(
                {
                    "codebook_size": codebook_size,
                    "index_bits": index_bits,
                    "block_size": block_size,
                    "covered_value_fraction": covered_value_fraction,
                    "passing_block_fraction": passing_blocks / block_count,
                    "fallback_blocks": fallback_blocks,
                    "optimistic_total_byte_fraction": byte_fraction,
                    "optimistic_lm_head_saving_ms": saving_ms,
                    "optimistic_whole_step_speedup": whole_step_speedup,
                }
            )
    del bits

    block_sizes = [int(value) for value in args.block_sizes.split(",")]
    layouts = []
    for block_size in block_sizes:
        if total_values % block_size:
            raise ValueError(f"block size {block_size} does not divide weight")
        blocks = exponent.reshape(-1, block_size)
        exponent_min = blocks.amin(dim=1)
        exponent_range = blocks.amax(dim=1) - exponent_min
        range_float = exponent_range.float()
        delta_layouts = []
        for delta_bits in range(3, 9):
            passing = exponent_range < (1 << delta_bits)
            passing_blocks = int(passing.sum().item())
            block_count = int(passing.numel())
            # One flag bit per block. Packed blocks store an eight-bit base and
            # exact sign+mantissa plus exponent delta; failed blocks store all
            # original sixteen bits. Padding/alignment and decode work are not
            # charged, so this remains an optimistic feasibility bound.
            packed_block_bits = block_size * (8 + delta_bits) + 8 + 1
            dense_block_bits = block_size * 16 + 1
            total_bits = (
                passing_blocks * packed_block_bits
                + (block_count - passing_blocks) * dense_block_bits
            )
            byte_fraction = total_bits / (total_values * 16)
            saving_ms = args.lm_head_ms * (1.0 - byte_fraction)
            whole_step_speedup = args.strict_frontier_tpot_ms / (
                args.strict_frontier_tpot_ms - saving_ms
            )
            delta_layouts.append(
                {
                    "delta_bits": delta_bits,
                    "packed_bits_per_value": 8 + delta_bits,
                    "passing_block_fraction": passing_blocks / block_count,
                    "optimistic_total_byte_fraction": byte_fraction,
                    "optimistic_lm_head_saving_ms": saving_ms,
                    "optimistic_whole_step_speedup": whole_step_speedup,
                }
            )
        layouts.append(
            {
                "block_size": block_size,
                "exponent_range": {
                    "min": int(exponent_range.min().item()),
                    "median": percentile(range_float, 0.5),
                    "p90": percentile(range_float, 0.9),
                    "p99": percentile(range_float, 0.99),
                    "max": int(exponent_range.max().item()),
                },
                "base_delta_layouts": delta_layouts,
            }
        )

    best_base_delta = min(
        (
            {"block_size": layout["block_size"], **candidate}
            for layout in layouts
            for candidate in layout["base_delta_layouts"]
        ),
        key=lambda value: value["optimistic_total_byte_fraction"],
    )
    best_codebook = min(
        codebook_layouts,
        key=lambda value: value["optimistic_total_byte_fraction"],
    )
    best = min(
        (
            {"family": "base_delta", **best_base_delta},
            {"family": "codebook", **best_codebook},
        ),
        key=lambda value: value["optimistic_total_byte_fraction"],
    )
    output = {
        "schema_version": "sm89-exact-bf16-weight-compression-feasibility-v1",
        "status": "PASS",
        "scope": {
            "model": str(model_path),
            "config_sha256": sha256(model_path / "config.json"),
            "gpu": torch.cuda.get_device_name(0),
            "tensor": "lm_head.weight",
            "dtype": "bfloat16",
            "weight_shape": weight_shape,
            "values": total_values,
            "dense_bytes": total_values * 2,
        },
        "bit_statistics": {
            "distinct_bf16_patterns": int((full_counts > 0).sum().item()),
            "full_pattern_entropy_bits_per_value": entropy_from_counts(full_counts),
            "distinct_exponents": len(exponent_values),
            "exponent_values": exponent_values,
            "exponent_histogram": exponent_histogram,
            "exponent_entropy_bits_per_value": entropy_from_counts(exponent_counts),
            "sign_mantissa_entropy_bits_per_value": entropy_from_counts(
                sign_mantissa_counts
            ),
        },
        "global_exponent_palette": {
            "index_bits": exponent_index_bits,
            "packed_bits_per_value": 8 + exponent_index_bits,
            "optimistic_total_byte_fraction": global_palette_byte_fraction,
        },
        "block_local_base_delta": layouts,
        "exact_pattern_codebook": codebook_layouts,
        "best_base_delta_layout": best_base_delta,
        "best_codebook_layout": best_codebook,
        "best_optimistic_layout": best,
        "materiality_gate": {
            "minimum_whole_step_speedup": 1.03,
            "passes": best["optimistic_whole_step_speedup"] >= 1.03,
            "warning": "The projection assumes lm-head latency scales exactly with bytes and charges zero unpacking, alignment or instruction overhead.",
        },
        "decision": (
            "IMPLEMENT_EXACT_GPU_UNPACK_SCREEN"
            if best["optimistic_whole_step_speedup"] >= 1.03
            else "STOP_EXACT_WEIGHT_COMPRESSION_BELOW_MATERIALITY"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
