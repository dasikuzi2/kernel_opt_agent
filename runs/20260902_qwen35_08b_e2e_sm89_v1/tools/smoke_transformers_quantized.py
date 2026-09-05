#!/usr/bin/env python3
"""Check whether a quantized text checkpoint is sane outside vLLM."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sanity(token_ids: list[int]) -> dict:
    distinct = len(set(token_ids))
    minimum = min(8, max(len(token_ids) // 4, 2))
    return {
        "status": "PASS" if distinct >= minimum else "FAIL",
        "distinct_token_count": distinct,
        "minimum_distinct_tokens": minimum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--new-tokens", type=int, default=32)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer.resolve(), local_files_only=True
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain in one paragraph why GPU memory bandwidth matters for token decoding."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model.resolve(),
        dtype=torch.bfloat16,
        local_files_only=True,
    ).eval().to("cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    new_ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
    result = {
        "schema_version": "qwen35-transformers-quantized-smoke-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "load_seconds": load_seconds,
        "generated_token_ids": new_ids,
        "decoded": tokenizer.decode(new_ids),
        "generation_sanity": sanity(new_ids),
    }
    result["status"] = result["generation_sanity"]["status"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
