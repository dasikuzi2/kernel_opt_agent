#!/usr/bin/env python3
"""Validate every discovery-baseline token against the Transformers reference."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


def exact_prompt(tokenizer, length: int) -> list[int]:
    seed_text = "Kernel optimization must preserve semantics while reducing global latency. "
    seed = tokenizer.encode(seed_text, add_special_tokens=False)
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--vllm-baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.vllm_baseline.read_text(encoding="utf-8"))
    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).eval().to("cuda")

    results = []
    for case in baseline["cases"]:
        prompt = exact_prompt(tokenizer, int(case["prompt_tokens"]))
        input_ids = torch.tensor([prompt], dtype=torch.int64, device="cuda")
        attention_mask = torch.ones_like(input_ids)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(case["generated_tokens"]),
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        actual = output[0, len(prompt):].tolist()
        expected = list(case["generated_token_ids"])
        first_mismatch = next(
            (index for index, (left, right) in enumerate(zip(actual, expected)) if left != right),
            None,
        )
        exact = actual == expected
        results.append({
            "case_id": case["case_id"],
            "status": "PASS" if exact else "FAIL",
            "prompt_tokens": len(prompt),
            "expected_tokens": len(expected),
            "actual_tokens": len(actual),
            "first_mismatch_index": first_mismatch,
            "transformers_seconds": elapsed,
            "expected_vllm_token_ids": expected,
            "actual_transformers_token_ids": actual,
        })

    payload = {
        "schema_version": "qwen35-vllm-transformers-parity-v1",
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "EXACT_GREEDY_TOKEN_PARITY_FOR_DISCOVERY_WORKLOAD",
        "vllm_baseline": str(args.vllm_baseline.resolve()),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "cases": [
        {key: value for key, value in case.items() if key not in {"expected_vllm_token_ids", "actual_transformers_token_ids"}}
        for case in results
    ]}, indent=2))


if __name__ == "__main__":
    main()
