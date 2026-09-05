#!/usr/bin/env python3
"""Build a model-bound Marlin W4 lm-head sidecar with production vLLM APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SCHEMA = "qwen35-sm89-marlin-lmhead-sidecar-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    chunk = 8 * 1024 * 1024
    for start in range(0, raw.numel(), chunk):
        digest.update(memoryview(raw[start : start + chunk].numpy()))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--tensor-key",
        default="model.language_model.embed_tokens.weight",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly replace an existing SHA-bound sidecar",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if output == checkpoint:
        raise ValueError("sidecar output must not overwrite the source checkpoint")
    if output.parent == checkpoint.parent:
        raise ValueError(
            "sidecar output must be outside the model directory because vLLM "
            "discovers every .safetensors file there as a checkpoint shard"
        )
    if output.exists() and not args.force:
        raise FileExistsError(
            f"sidecar already exists: {output}; pass --force to replace it"
        )
    if args.group_size != 128:
        raise ValueError("this deployment candidate is frozen to group_size=128")

    load_started = time.perf_counter()
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        if args.tensor_key not in handle.keys():
            raise KeyError(f"tensor is missing: {args.tensor_key}")
        checkpoint_nk_cpu = handle.get_tensor(args.tensor_key)
    if checkpoint_nk_cpu.dtype not in (torch.float16, torch.bfloat16) or checkpoint_nk_cpu.ndim != 2:
        raise ValueError(
            f"expected a 2-D FP16/BF16 tensor, got {checkpoint_nk_cpu.dtype} "
            f"shape={tuple(checkpoint_nk_cpu.shape)}"
        )
    load_seconds = time.perf_counter() - load_started
    checkpoint_tensor_sha256 = sha256_tensor(checkpoint_nk_cpu)
    dense_nk_cpu = checkpoint_nk_cpu.to(torch.bfloat16).contiguous()
    dense_sha256 = sha256_tensor(dense_nk_cpu)

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        gptq_quantize_weights,
        pack_rows,
    )
    from vllm.scalar_type import scalar_types

    quant_started = time.perf_counter()
    dense_kn = dense_nk_cpu.to(args.device).t().contiguous()
    size_k, size_n = dense_kn.shape
    _, qweight, scales, g_idx, _ = gptq_quantize_weights(
        dense_kn,
        scalar_types.uint4b8,
        args.group_size,
        False,
    )
    if g_idx.numel() != 0:
        raise ValueError("unexpected activation-order g_idx for frozen symmetric W4")
    qweight = pack_rows(qweight, 4, size_k, size_n)
    empty_perm = torch.empty(0, dtype=torch.int, device=args.device)
    packed = ops.gptq_marlin_repack(
        qweight,
        perm=empty_perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=4,
        is_a_8bit=False,
    )
    permuted_scales = marlin_permute_scales(
        scales,
        size_k,
        size_n,
        args.group_size,
        is_a_8bit=False,
    )
    torch.cuda.synchronize()
    quantize_seconds = time.perf_counter() - quant_started

    tensors = {
        "packed_weight": packed.detach().cpu().contiguous(),
        "permuted_scales": permuted_scales.detach().cpu().contiguous(),
    }
    metadata = {
        "schema_version": SCHEMA,
        "source_checkpoint": checkpoint.name,
        "source_tensor_key": args.tensor_key,
        "checkpoint_tensor_sha256": checkpoint_tensor_sha256,
        "checkpoint_tensor_dtype": str(checkpoint_nk_cpu.dtype),
        "source_tensor_sha256": dense_sha256,
        "dense_shape": json.dumps(list(dense_nk_cpu.shape)),
        "dense_dtype": str(dense_nk_cpu.dtype),
        "quant_type": "uint4b8",
        "group_size": str(args.group_size),
        "size_k": str(size_k),
        "size_n": str(size_n),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.safetensors")
    if temporary.exists():
        raise FileExistsError(f"stale temporary sidecar exists: {temporary}")
    save_started = time.perf_counter()
    save_file(tensors, temporary, metadata=metadata)
    temporary.replace(output)
    save_seconds = time.perf_counter() - save_started

    payload = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "checkpoint": str(checkpoint),
            "tensor_key": args.tensor_key,
            "tensor_shape": list(checkpoint_nk_cpu.shape),
            "checkpoint_tensor_dtype": str(checkpoint_nk_cpu.dtype),
            "checkpoint_tensor_sha256": checkpoint_tensor_sha256,
            "runtime_tensor_dtype": str(dense_nk_cpu.dtype),
            "runtime_tensor_sha256": dense_sha256,
        },
        "sidecar": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "packed_weight_shape": list(tensors["packed_weight"].shape),
            "packed_weight_dtype": str(tensors["packed_weight"].dtype),
            "permuted_scales_shape": list(tensors["permuted_scales"].shape),
            "permuted_scales_dtype": str(tensors["permuted_scales"].dtype),
        },
        "timings_seconds": {
            "checkpoint_tensor_load": load_seconds,
            "quantize_repack": quantize_seconds,
            "sidecar_save": save_seconds,
        },
        "contract": {
            "runtime_use_requires_exact_sidecar_sha256": True,
            "runtime_use_requires_dense_shape_and_metadata_match": True,
            "packed_equivalence_requires_separate_online_vs_sidecar_validation": True,
            "dense_weight_retained": True,
            "reason_dense_weight_retained": "tied input embedding and BF16 shortlist rerank both consume the original rows",
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
