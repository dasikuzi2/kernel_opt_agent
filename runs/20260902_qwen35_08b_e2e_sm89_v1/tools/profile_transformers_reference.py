#!/usr/bin/env python3
"""Bounded operator profile for the diagnostic Transformers reference path."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration


def exact_prompt(vocab_size: int, length: int) -> torch.Tensor:
    # Avoid tokenizer and text-shape drift: deterministic valid non-special ids.
    values = (torch.arange(length, dtype=torch.int64) * 1543 + 17) % (vocab_size - 256)
    return (values + 128).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).eval().to("cuda")
    prompt = exact_prompt(model.config.text_config.vocab_size, args.prompt_tokens).cuda()
    attention_mask = torch.ones_like(prompt)

    def generate() -> None:
        model.generate(
            prompt,
            attention_mask=attention_mask,
            pad_token_id=model.config.text_config.eos_token_id,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )

    with torch.inference_mode():
        generate()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            generate()
            torch.cuda.synchronize()
        profiled_wall_seconds = time.perf_counter() - started

    rows = []
    for event in prof.key_averages():
        device_us = float(getattr(event, "self_device_time_total", 0.0))
        cpu_us = float(event.self_cpu_time_total)
        if device_us <= 0.0 and cpu_us <= 0.0:
            continue
        rows.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "self_device_us": device_us,
                "self_cpu_us": cpu_us,
                "device_memory_bytes": int(getattr(event, "self_device_memory_usage", 0)),
                "input_shapes": str(event.input_shapes),
            }
        )
    rows.sort(key=lambda item: item["self_device_us"], reverse=True)
    aten_rows = [item for item in rows if item["name"].startswith("aten::")]
    kernel_rows = [item for item in rows if not item["name"].startswith("aten::")]
    payload = {
        "schema_version": "qwen35-transformers-operator-profile-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "DIAGNOSTIC_REFERENCE_PATH_ONLY_NOT_VLLM_PRODUCTION_BASELINE",
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "input": {"prompt_tokens": args.prompt_tokens, "generated_tokens": 1},
        "measurement": {
            "profiled_wall_seconds": profiled_wall_seconds,
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
            "aten_self_device_us": sum(item["self_device_us"] for item in aten_rows),
            "kernel_self_device_us": sum(item["self_device_us"] for item in kernel_rows),
            "all_self_cpu_us": sum(item["self_cpu_us"] for item in rows),
            "note": "ATen and device-kernel rows are separate views and must not be summed together.",
        },
        "top_aten_by_self_device_time": aten_rows[: args.top],
        "top_device_kernels_by_self_device_time": kernel_rows[: args.top],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
