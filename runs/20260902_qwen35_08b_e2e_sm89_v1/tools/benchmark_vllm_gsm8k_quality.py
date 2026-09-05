#!/usr/bin/env python3
"""Run a frozen GSM8K answer-quality screen through vLLM decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path


REFERENCE_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalize_number(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.replace(",", "")
    match = re.fullmatch(r"(-?)([0-9]+)(?:\.([0-9]+))?", value)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    integer = integer.lstrip("0") or "0"
    fraction = (fraction or "").rstrip("0")
    if integer == "0" and not fraction:
        sign = ""
    return f"{sign}{integer}.{fraction}" if fraction else f"{sign}{integer}"


def extract_prediction(text: str) -> str | None:
    matches = NUMBER_RE.findall(text)
    return normalize_number(matches[-1]) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--selection-seed", type=int, default=20260905)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=536870912)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--quantization",
        choices=(
            "none",
            "fp8",
            "fp8_per_tensor",
            "fp8_per_block",
            "fp8_per_channel",
            "auto_gptq",
            "gptq_marlin",
        ),
        default="none",
    )
    parser.add_argument("--expect-source-sha256", action="append", default=[])
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    observed_dataset_hash = sha256(dataset_path)
    if observed_dataset_hash != args.dataset_sha256.lower():
        raise RuntimeError(
            f"dataset hash mismatch: expected {args.dataset_sha256}, "
            f"observed {observed_dataset_hash}"
        )
    guarded_sources: dict[str, str] = {}
    for spec in args.expect_source_sha256:
        source_raw, expected = spec.rsplit("=", 1)
        source = Path(source_raw).expanduser().resolve()
        observed = sha256(source)
        if observed != expected.lower():
            raise RuntimeError(
                f"source hash mismatch for {source}: expected {expected}, "
                f"observed {observed}"
            )
        guarded_sources[str(source)] = observed

    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
    ]
    if args.samples < 1 or args.samples > len(rows):
        raise ValueError(f"samples must be in [1, {len(rows)}]")
    rng = random.Random(args.selection_seed)
    selected_indices = sorted(rng.sample(range(len(rows)), args.samples))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    cases = []
    for index in selected_indices:
        row = rows[index]
        reference_match = REFERENCE_RE.search(row["answer"])
        if reference_match is None:
            raise ValueError(f"missing GSM8K reference marker at row {index}")
        request = (
            "Solve the following grade-school math problem. Return only the "
            "final numeric answer without units or explanation.\n\n" + row["question"]
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": request}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        cases.append(
            {
                "dataset_index": index,
                "question_sha256": hashlib.sha256(
                    row["question"].encode("utf-8")
                ).hexdigest(),
                "reference": normalize_number(reference_match.group(1)),
                "prompt_token_ids": tokenizer.encode(
                    rendered, add_special_tokens=False
                ),
            }
        )

    init_started = time.perf_counter()
    llm_kwargs = {
        "model": str(model_path),
        "tokenizer": str(model_path),
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "max_num_seqs": args.max_num_seqs,
        "language_model_only": True,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "seed": args.selection_seed,
    }
    if args.quantization != "none":
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(
        **llm_kwargs,
    )
    init_seconds = time.perf_counter() - init_started
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.selection_seed,
    )

    # Warm both short and long prompt paths without contaminating scored rows.
    for case in (cases[0], cases[-1]):
        llm.generate(
            {"prompt_token_ids": case["prompt_token_ids"]},
            params,
            use_tqdm=False,
        )

    results = []
    started = time.perf_counter()
    for case in cases:
        output = llm.generate(
            {"prompt_token_ids": case["prompt_token_ids"]},
            params,
            use_tqdm=False,
        )[0].outputs[0]
        text = output.text
        prediction = extract_prediction(text)
        results.append(
            {
                "dataset_index": case["dataset_index"],
                "question_sha256": case["question_sha256"],
                "reference": case["reference"],
                "prediction": prediction,
                "correct": prediction == case["reference"],
                "generated_token_ids": list(output.token_ids),
                "generated_text": text,
            }
        )
    elapsed_seconds = time.perf_counter() - started

    payload = {
        "schema_version": "vllm-gsm8k-decode-quality-v1",
        "status": "PASS",
        "dataset": {
            "path": str(dataset_path),
            "sha256": observed_dataset_hash,
            "upstream_commit": "3101c7d5072418e28b9008a6636bde82a006892c",
            "selection_seed": args.selection_seed,
            "selected_indices": selected_indices,
        },
        "model": str(model_path),
        "environment": {
            "vllm_sm89_bf16_lm_head": os.environ.get(
                "VLLM_SM89_BF16_LM_HEAD", "0"
            ),
            "vllm_sm89_bf16_gemv": os.environ.get(
                "VLLM_SM89_BF16_GEMV", "none(default)"
            ),
            "vllm_sm89_segmented_gdn_projection": os.environ.get(
                "VLLM_SM89_SEGMENTED_GDN_PROJECTION", "0"
            ),
            "vllm_sm89_exact_packed_lm_head": os.environ.get(
                "VLLM_SM89_EXACT_PACKED_LM_HEAD", "0"
            ),
            "vllm_sm89_marlin_w4_rerank": os.environ.get(
                "VLLM_SM89_MARLIN_W4_RERANK", "0"
            ),
            "vllm_sm89_marlin_w4_scan_only": os.environ.get(
                "VLLM_SM89_MARLIN_W4_SCAN_ONLY", "0"
            ),
            "vllm_sm89_int8_groupwise_lm_head": os.environ.get(
                "VLLM_SM89_INT8_GROUPWISE_LM_HEAD", "off(default)"
            ),
            "vllm_sm89_int8_bf16_rerank_topk": os.environ.get(
                "VLLM_SM89_INT8_BF16_RERANK_TOPK", "off(default)"
            ),
            "vllm_cache_root": os.environ.get("VLLM_CACHE_ROOT"),
            "guarded_source_sha256": guarded_sources,
        },
        "controls": {
            "dtype": "bfloat16",
            "quantization": args.quantization,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "sample_count": len(results),
            "engine_initialization_seconds": init_seconds,
            "scored_wall_seconds": elapsed_seconds,
        },
        "accuracy": sum(item["correct"] for item in results) / len(results),
        "correct_count": sum(item["correct"] for item in results),
        "parseable_count": sum(item["prediction"] is not None for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "accuracy": payload["accuracy"],
                "correct_count": payload["correct_count"],
                "parseable_count": payload["parseable_count"],
                "sample_count": len(results),
                "scored_wall_seconds": elapsed_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
