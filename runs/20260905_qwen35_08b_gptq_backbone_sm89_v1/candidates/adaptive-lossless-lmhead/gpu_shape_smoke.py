#!/usr/bin/env python3
"""Exercise a non-model-specific, non-power-of-two shape on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.model_executor.kernels.linear.unquantized.lossless_packed_lm_head import (
    _PACK_BLOCK,
    _STATE_PREFIX,
    _packed_bf16_lm_head_impl,
    choose_launch_config,
    pack_bf16_weight,
    try_apply_lossless_packed_lm_head,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("gpu_shape_smoke_result.json"),
    )
    args = parser.parse_args()
    torch.manual_seed(20260905)
    n, k = 33001, 1023
    weight = (torch.rand((n, k), device="cuda") * 2.0 - 1.0).to(torch.bfloat16)
    # Force one fallback block containing special values and a wide exponent span.
    special = torch.tensor(
        [0x0000, 0x8000, 0x0001, 0x3F80, 0xBF80, 0x4000, 0xC000],
        dtype=torch.uint16,
        device="cuda",
    ).view(torch.int16)
    weight.view(torch.int16).reshape(-1)[: special.numel()] = special
    packed = pack_bf16_weight(weight, max_packed_fraction=0.90)
    if packed is None:
        raise RuntimeError("synthetic generic shape unexpectedly failed eligibility")
    plan, tensors = packed
    sign_mantissa, exponent_nibbles, base_exponent, fallback_slot, fallback_bits = tensors

    flat = weight.view(torch.int16).reshape(-1)
    sample_indices = torch.cat(
        (
            torch.arange(0, 4096, device="cuda", dtype=torch.int64),
            torch.arange(flat.numel() - 4096, flat.numel(), device="cuda", dtype=torch.int64),
        )
    )
    block = sample_indices // _PACK_BLOCK
    within = sample_indices % _PACK_BLOCK
    slot = fallback_slot[block].to(torch.int64)
    sm = sign_mantissa[sample_indices].to(torch.int32)
    pair = exponent_nibbles[sample_indices // 2].to(torch.int32)
    delta = (pair >> ((sample_indices & 1).to(torch.int32) * 4)) & 0xF
    rebuilt = ((sm & 0x80) << 8) | (
        (base_exponent[block].to(torch.int32) + delta) << 7
    ) | (sm & 0x7F)
    fallback = fallback_bits[slot.clamp_min(0), within].to(torch.int32) & 0xFFFF
    rebuilt = torch.where(slot >= 0, fallback, rebuilt).to(torch.int16)
    if not torch.equal(rebuilt, flat[sample_indices]):
        raise RuntimeError("sampled BF16 bit reconstruction mismatch")

    x = torch.randn((1, k), dtype=torch.bfloat16, device="cuda")
    launch = choose_launch_config(k)
    actual = _packed_bf16_lm_head_impl(
        x, *tensors, n, k, launch.block_n, launch.num_warps
    )
    expected = torch.nn.functional.linear(x, weight)
    layer = SimpleNamespace(
        _vllm_lossless_lm_head_meta=(n, k, launch.block_n, launch.num_warps)
    )
    for name, value in zip(
        ("sign_mantissa", "exponent_nibbles", "base_exponent", "fallback_slot", "fallback_bits"),
        tensors,
    ):
        setattr(layer, _STATE_PREFIX + name, value)
    if try_apply_lossless_packed_lm_head(layer, x.repeat(2, 1), None) is not None:
        raise RuntimeError("M=2 did not fall back")
    if try_apply_lossless_packed_lm_head(layer, x, torch.zeros(n, device="cuda")) is not None:
        raise RuntimeError("biased projection did not fall back")
    max_abs_delta = float((actual.float() - expected.float()).abs().max())
    expected_peak = float(expected.float().abs().max())

    result = {
        "status": "PASS",
        "claim_scope": "SYNTHETIC_CORRECTNESS_AND_FALLBACK_ONLY",
        "shape": [n, k],
        "packed_fraction": plan.packed_fraction,
        "fallback_blocks": plan.fallback_blocks,
        "sampled_weight_bits_exact": int(sample_indices.numel()),
        "argmax_equal": bool(actual.argmax() == expected.argmax()),
        "max_abs_logit_delta": max_abs_delta,
        "max_delta_over_logit_peak": max_abs_delta / expected_peak,
        "m2_fallback": True,
        "bias_fallback": True,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
