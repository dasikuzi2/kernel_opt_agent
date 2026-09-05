#!/usr/bin/env python3
"""Create a compact, hash-bound inventory of the frozen Qwen3.5 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from math import prod
from pathlib import Path

from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def category(key: str) -> str:
    if "visual" in key:
        return "vision"
    if "linear_attn" in key:
        return "gated_delta_net"
    if "self_attn" in key:
        return "full_attention"
    if ".mlp." in key:
        return "mlp"
    if "embed_tokens" in key or "lm_head" in key:
        return "embedding_and_lm_head"
    if "norm" in key:
        return "normalization"
    return "other_text"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = args.model.resolve()
    config_path = model / "config.json"
    weight_path = model / "model.safetensors-00001-of-00001.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config["text_config"]

    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensors": 0, "parameters": 0, "storage_bytes": 0}
    )
    dtypes: dict[str, int] = defaultdict(int)
    key_examples: dict[str, list[str]] = defaultdict(list)
    with safe_open(weight_path, framework="pt", device="cpu") as weights:
        for key in weights.keys():
            view = weights.get_slice(key)
            count = prod(view.get_shape())
            dtype = str(view.get_dtype())
            element_bytes = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1}.get(dtype)
            if element_bytes is None:
                raise ValueError(f"unhandled storage dtype {dtype} for {key}")
            group = category(key)
            groups[group]["tensors"] += 1
            groups[group]["parameters"] += count
            groups[group]["storage_bytes"] += count * element_bytes
            dtypes[dtype] += count
            if len(key_examples[group]) < 3:
                key_examples[group].append(key)

    layer_types = text_config["layer_types"]
    payload = {
        "schema_version": "qwen35-model-snapshot-inventory-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "modelscope_id": "Qwen/Qwen3.5-0.8B",
        "local_path": str(model),
        "identity": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "weights": {
                "path": str(weight_path),
                "size_bytes": weight_path.stat().st_size,
                "sha256": sha256(weight_path),
            },
        },
        "architecture": {
            "architectures": config["architectures"],
            "hidden_size": text_config["hidden_size"],
            "layers": text_config["num_hidden_layers"],
            "layer_types": layer_types,
            "gated_delta_net_layers": layer_types.count("linear_attention"),
            "full_attention_layers": layer_types.count("full_attention"),
            "linear_num_key_heads": text_config["linear_num_key_heads"],
            "linear_num_value_heads": text_config["linear_num_value_heads"],
            "linear_key_head_dim": text_config["linear_key_head_dim"],
            "linear_value_head_dim": text_config["linear_value_head_dim"],
            "vocab_size": text_config["vocab_size"],
        },
        "parameter_groups": dict(sorted(groups.items())),
        "parameter_dtypes": dict(sorted(dtypes.items())),
        "total_parameters": sum(group["parameters"] for group in groups.values()),
        "total_storage_bytes": sum(group["storage_bytes"] for group in groups.values()),
        "key_examples": dict(sorted(key_examples.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
