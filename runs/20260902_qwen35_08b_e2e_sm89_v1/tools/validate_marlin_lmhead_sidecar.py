#!/usr/bin/env python3
"""Prove online and sidecar Marlin lm-head materializations are bit-identical."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--tensor-key",
        default="model.language_model.embed_tokens.weight",
    )
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--expect-sidecar-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    sidecar = args.sidecar.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    actual_sidecar_sha256 = sha256_file(sidecar)
    expected_sidecar_sha256 = args.expect_sidecar_sha256.lower()
    if actual_sidecar_sha256 != expected_sidecar_sha256:
        raise ValueError(
            f"sidecar SHA mismatch: expected {expected_sidecar_sha256}, "
            f"got {actual_sidecar_sha256}"
        )

    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        checkpoint_weight = handle.get_tensor(args.tensor_key)
    weight = checkpoint_weight.to(torch.bfloat16).to(args.device).contiguous()

    from vllm.model_executor.layers import utils

    os.environ.pop("VLLM_SM89_MARLIN_W4_SIDECAR", None)
    os.environ.pop("VLLM_SM89_MARLIN_W4_SIDECAR_SHA256", None)
    utils._SM89_MARLIN_W4_CACHE.clear()
    torch.cuda.synchronize()
    online_started = time.perf_counter()
    online = utils._get_sm89_marlin_w4_weight(weight)
    torch.cuda.synchronize()
    online_seconds = time.perf_counter() - online_started

    os.environ["VLLM_SM89_MARLIN_W4_SIDECAR"] = str(sidecar)
    os.environ["VLLM_SM89_MARLIN_W4_SIDECAR_SHA256"] = expected_sidecar_sha256
    utils._SM89_MARLIN_W4_CACHE.clear()
    torch.cuda.synchronize()
    sidecar_started = time.perf_counter()
    loaded = utils._get_sm89_marlin_w4_weight(weight)
    torch.cuda.synchronize()
    sidecar_seconds = time.perf_counter() - sidecar_started

    packed_equal = torch.equal(online[0], loaded[0])
    scales_equal = torch.equal(online[1], loaded[1])
    g_idx_equal = torch.equal(online[2], loaded[2])
    sort_indices_equal = torch.equal(online[3], loaded[3])

    torch.manual_seed(20260905)
    x = torch.randn((2, weight.shape[1]), device=args.device, dtype=torch.bfloat16)
    utils._SM89_MARLIN_W4_CACHE.clear()
    os.environ.pop("VLLM_SM89_MARLIN_W4_SIDECAR", None)
    os.environ.pop("VLLM_SM89_MARLIN_W4_SIDECAR_SHA256", None)
    online_logits = utils.sm89_marlin_w4_rerank_impl(x, weight).clone()
    utils._SM89_MARLIN_W4_CACHE.clear()
    os.environ["VLLM_SM89_MARLIN_W4_SIDECAR"] = str(sidecar)
    os.environ["VLLM_SM89_MARLIN_W4_SIDECAR_SHA256"] = expected_sidecar_sha256
    sidecar_logits = utils.sm89_marlin_w4_rerank_impl(x, weight).clone()
    logits_equal = torch.equal(online_logits, sidecar_logits)
    argmax_equal = torch.equal(
        online_logits.argmax(dim=-1),
        sidecar_logits.argmax(dim=-1),
    )

    utils._SM89_MARLIN_W4_CACHE.clear()
    os.environ["VLLM_SM89_MARLIN_W4_SIDECAR_SHA256"] = "0" * 64
    wrong_sha_rejected = False
    try:
        utils._get_sm89_marlin_w4_weight(weight)
    except ValueError as error:
        wrong_sha_rejected = "SHA-256 mismatch" in str(error)
    finally:
        os.environ["VLLM_SM89_MARLIN_W4_SIDECAR_SHA256"] = expected_sidecar_sha256

    checks = {
        "packed_weight_bit_equal": packed_equal,
        "permuted_scales_bit_equal": scales_equal,
        "g_idx_equal": g_idx_equal,
        "sort_indices_equal": sort_indices_equal,
        "reranked_logits_bit_equal": logits_equal,
        "argmax_equal": argmax_equal,
        "wrong_sidecar_sha256_rejected": wrong_sha_rejected,
    }
    if not all(checks.values()):
        raise ValueError(f"sidecar equivalence failed: {checks}")

    utils_path = Path(utils.__file__).resolve()
    payload = {
        "schema_version": "qwen35-sm89-marlin-lmhead-sidecar-validation-v1",
        "status": "PASS",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "utils_path": str(utils_path),
            "utils_sha256": sha256_file(utils_path),
        },
        "inputs": {
            "checkpoint": str(checkpoint),
            "tensor_key": args.tensor_key,
            "sidecar": str(sidecar),
            "sidecar_sha256": actual_sidecar_sha256,
            "dense_shape": list(weight.shape),
            "dense_dtype": str(weight.dtype),
            "random_seed": 20260905,
            "test_rows": 2,
        },
        "checks": checks,
        "materialization_seconds": {
            "online_quantize_repack": online_seconds,
            "sidecar_hash_and_load": sidecar_seconds,
            "speedup": online_seconds / sidecar_seconds,
        },
        "scope": {
            "proves_same_packed_tensors": True,
            "proves_same_rerank_function_for_test_input": True,
            "does_not_prove_global_performance": True,
            "does_not_remove_dense_weight": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
