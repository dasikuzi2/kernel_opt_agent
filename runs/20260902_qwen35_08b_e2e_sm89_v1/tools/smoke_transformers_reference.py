#!/usr/bin/env python3
"""Load the frozen Qwen3.5 checkpoint and record a deterministic CUDA smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _prompt_ids(tokenizer, length):
    seed_text = "Kernel optimization must preserve semantics while reducing global latency. "
    seed = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed:
        raise RuntimeError("tokenizer produced an empty seed")
    values = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
    return torch.tensor([values], dtype=torch.long, device="cuda")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--new-tokens", type=int, default=2)
    args = parser.parse_args()
    model_path = args.model.resolve()
    torch.manual_seed(20260902)
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).eval().to("cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    input_ids = _prompt_ids(tokenizer, args.prompt_tokens)

    def generate():
        with torch.inference_mode():
            return model.generate(
                input_ids=input_ids,
                max_new_tokens=args.new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

    first_start = time.perf_counter()
    first = generate()
    torch.cuda.synchronize()
    first_seconds = time.perf_counter() - first_start
    warm_start = time.perf_counter()
    second = generate()
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - warm_start
    deterministic = bool(torch.equal(first, second))
    generated = second[0, args.prompt_tokens:].tolist()
    result = {
        "schema_version": "qwen35-transformers-reference-smoke-v1",
        "status": "PASS" if deterministic and len(generated) == args.new_tokens else "FAIL",
        "claim_scope": "CORRECTNESS_AND_ENVIRONMENT_SMOKE_NOT_VLLM_BASELINE",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "weight_sha256": _sha256(model_path / "model.safetensors-00001-of-00001.safetensors"),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "input": {"prompt_tokens": args.prompt_tokens, "new_tokens": args.new_tokens},
        "output": {"generated_token_ids": generated, "deterministic_repeat": deterministic},
        "timing": {
            "model_load_seconds": load_seconds,
            "first_generate_seconds": first_seconds,
            "warm_generate_seconds": warm_seconds,
            "acceptance_use": "diagnostic only",
        },
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
