#!/usr/bin/env python3
"""Inspect GPTQ tensors and derive vLLM's shared-input fused projection shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from safetensors import safe_open


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _shape_map(model_path: Path) -> dict[str, tuple[int, ...]]:
    result = {}
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if name.endswith(".qweight"):
                result[name] = tuple(handle.get_slice(name).get_shape())
    return result


def _linear_shape(qweight_shape: tuple[int, ...], pack_factor: int) -> tuple[int, int]:
    if len(qweight_shape) != 2:
        raise ValueError(f"qweight must be rank two, got {qweight_shape}")
    packed_k, output_features = qweight_shape
    return output_features, packed_k * pack_factor


def inspect(model_dir: Path, pack_factor: int = 8) -> dict:
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    qweights = _shape_map(weights_path)
    layers: dict[int, dict[str, tuple[int, int]]] = {}
    prefix = "model.language_model.layers."
    for name, packed_shape in qweights.items():
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix) :]
        layer_text, suffix = remainder.split(".", 1)
        layers.setdefault(int(layer_text), {})[suffix.removesuffix(".qweight")] = (
            _linear_shape(packed_shape, pack_factor)
        )

    derived = []
    group_specs = (
        ("mlp_gate_up", ("mlp.gate_proj", "mlp.up_proj")),
        ("mlp_down", ("mlp.down_proj",)),
        (
            "linear_attention_qkvz",
            ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
        ),
        (
            "linear_attention_ba",
            ("linear_attn.in_proj_b", "linear_attn.in_proj_a"),
        ),
        ("linear_attention_out", ("linear_attn.out_proj",)),
        (
            "full_attention_qkv",
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        ),
        ("full_attention_out", ("self_attn.o_proj",)),
    )
    for layer, tensors in sorted(layers.items()):
        for fused_name, members in group_specs:
            present = [member for member in members if member in tensors]
            if not present:
                continue
            if len(present) != len(members):
                raise ValueError(
                    f"layer {layer} has partial {fused_name}: {present}, expected {members}"
                )
            shapes = [tensors[member] for member in members]
            input_features = {shape[1] for shape in shapes}
            if len(input_features) != 1:
                raise ValueError(f"layer {layer} {fused_name} inputs differ: {shapes}")
            derived.append(
                {
                    "layer": layer,
                    "projection": fused_name,
                    "members": list(members),
                    "decode_shape_m_n_k": [1, sum(shape[0] for shape in shapes), shapes[0][1]],
                }
            )

    counts = Counter(
        (item["projection"], tuple(item["decode_shape_m_n_k"])) for item in derived
    )
    grouped = [
        {
            "projection": projection,
            "decode_shape_m_n_k": list(shape),
            "calls_per_decode": count,
        }
        for (projection, shape), count in sorted(counts.items())
    ]
    return {
        "schema_version": "gptq-projection-shape-inventory-v1",
        "status": "PASS",
        "claim_scope": "CHECKPOINT_AND_VLLM_SHARED_INPUT_FUSION_LAYOUT",
        "model": {
            "path": str(model_dir),
            "config_sha256": digest(config_path),
            "weights_sha256": digest(weights_path),
            "quantization_config": config.get("quantization_config"),
            "hidden_size": text_config.get("hidden_size"),
            "intermediate_size": text_config.get("intermediate_size"),
            "num_hidden_layers": text_config.get("num_hidden_layers"),
            "layer_types": text_config.get("layer_types"),
        },
        "packing": {
            "qweight_pack_factor": pack_factor,
            "shape_rule": "qweight[packed_K, N] -> decode GEMM [M=1,N,K=packed_K*pack_factor]",
        },
        "projection_groups": grouped,
        "calls_per_decode": sum(item["calls_per_decode"] for item in grouped),
        "per_layer_projection_groups": derived,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pack-factor", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.model.resolve(), args.pack_factor)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
