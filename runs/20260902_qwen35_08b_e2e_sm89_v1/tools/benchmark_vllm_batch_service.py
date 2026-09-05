#!/usr/bin/env python3
"""Measure vLLM offline batch service curves with deterministic decode."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


PROMPTS = [
    "请用简洁的语言解释为什么GPU显存带宽会限制大模型逐token推理。",
    "Write a Python function that merges two sorted integer lists without calling sorted.",
    "A shop discounts an item by 20% and then raises the discounted price by 25%. Is the final price equal to the original? Explain.",
    "Rewrite this sentence to be clearer: The service, after it was restarted due to an error that happened unexpectedly, was available again.",
    "Design a bounded retry policy for a distributed job worker and list the main failure modes.",
    "把下面这句话翻译成英文：优化之前必须先确认真正的瓶颈，而不是只看某一个算子的耗时。",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-sizes", default="1,2,3,4,8")
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--quantization", default="fp8_per_block")
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=536870912)
    parser.add_argument("--expect-source-sha256", action="append", default=[])
    parser.add_argument(
        "--expect-vllm-cache-root",
        type=Path,
        help="Fail before engine startup unless VLLM_CACHE_ROOT resolves here.",
    )
    parser.add_argument(
        "--expect-marlin-w4-rerank",
        choices=("off", "on"),
        help="Fail before engine startup unless the Marlin-W4 rerank switch matches.",
    )
    parser.add_argument(
        "--expect-marlin-w4-scan-only",
        choices=("off", "on"),
        help="Fail before engine startup unless the Marlin-W4 scan-only switch matches.",
    )
    parser.add_argument(
        "--expect-bf16-lm-head",
        choices=("off", "on"),
        help="Fail before engine startup unless the SM89 BF16 lm-head switch matches.",
    )
    parser.add_argument(
        "--expect-exact-packed-lm-head",
        choices=("off", "on"),
        help="Fail before engine startup unless exact BF16 packed lm-head matches.",
    )
    args = parser.parse_args()

    actual_vllm_cache_root_raw = os.environ.get("VLLM_CACHE_ROOT")
    actual_vllm_cache_root = (
        str(Path(actual_vllm_cache_root_raw).expanduser().resolve())
        if actual_vllm_cache_root_raw
        else None
    )
    if args.expect_vllm_cache_root is not None:
        expected_vllm_cache_root = str(
            args.expect_vllm_cache_root.expanduser().resolve()
        )
        if actual_vllm_cache_root != expected_vllm_cache_root:
            raise RuntimeError(
                "vLLM cache-root mismatch: expected "
                f"{expected_vllm_cache_root}, got {actual_vllm_cache_root!r}"
            )

    actual_marlin_w4_rerank = (
        "on"
        if os.environ.get("VLLM_SM89_MARLIN_W4_RERANK", "0") == "1"
        else "off"
    )
    actual_marlin_w4_scan_only = (
        "on"
        if os.environ.get("VLLM_SM89_MARLIN_W4_SCAN_ONLY", "0") == "1"
        else "off"
    )
    actual_bf16_lm_head = (
        "on" if os.environ.get("VLLM_SM89_BF16_LM_HEAD", "0") == "1" else "off"
    )
    actual_exact_packed_lm_head = (
        "on"
        if os.environ.get("VLLM_SM89_EXACT_PACKED_LM_HEAD", "0") == "1"
        else "off"
    )
    if (
        args.expect_marlin_w4_rerank is not None
        and actual_marlin_w4_rerank != args.expect_marlin_w4_rerank
    ):
        raise RuntimeError(
            "candidate path is unreachable: expected Marlin-W4 rerank "
            f"{args.expect_marlin_w4_rerank}, got {actual_marlin_w4_rerank}"
        )
    if (
        args.expect_marlin_w4_scan_only is not None
        and actual_marlin_w4_scan_only != args.expect_marlin_w4_scan_only
    ):
        raise RuntimeError(
            "candidate path is unreachable: expected Marlin-W4 scan-only "
            f"{args.expect_marlin_w4_scan_only}, got {actual_marlin_w4_scan_only}"
        )
    if (
        args.expect_bf16_lm_head is not None
        and actual_bf16_lm_head != args.expect_bf16_lm_head
    ):
        raise RuntimeError(
            "candidate path is unreachable: expected SM89 BF16 lm-head "
            f"{args.expect_bf16_lm_head}, got {actual_bf16_lm_head}"
        )
    if (
        args.expect_exact_packed_lm_head is not None
        and actual_exact_packed_lm_head != args.expect_exact_packed_lm_head
    ):
        raise RuntimeError(
            "candidate path is unreachable: expected exact BF16 packed lm-head "
            f"{args.expect_exact_packed_lm_head}, got {actual_exact_packed_lm_head}"
        )

    guarded_sources = {}
    for spec in args.expect_source_sha256:
        source_raw, expected = spec.rsplit("=", 1)
        source = Path(source_raw).expanduser().resolve()
        observed = sha256(source)
        if observed != expected.lower():
            raise RuntimeError(
                f"source hash mismatch for {source}: expected {expected}, observed {observed}"
            )
        guarded_sources[str(source)] = observed

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch
    import vllm

    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompts = []
    for text in PROMPTS:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(tokenizer.encode(rendered, add_special_tokens=False))
    prompt_token_ids_sha256 = [
        hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for ids in prompts
    ]

    init_started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        quantization=None if args.quantization == "none" else args.quantization,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_num_seqs=max(int(value) for value in args.batch_sizes.split(",")),
        language_model_only=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=20260902,
    )
    init_seconds = time.perf_counter() - init_started
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.new_tokens,
        ignore_eos=True,
        seed=20260902,
    )

    def run_batch(batch_size: int, phase: str, iteration: int) -> dict:
        batch_prompts = [
            {"prompt_token_ids": prompts[(iteration + index) % len(prompts)]}
            for index in range(batch_size)
        ]
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(batch_prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        request_metrics = []
        token_ids = []
        for output in outputs:
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            token_ids.append(ids)
            metrics = output.metrics
            if metrics is None:
                raise RuntimeError("vLLM request metrics unavailable")
            decode_intervals = max(len(ids) - 1, 1)
            decode_seconds = max(
                float(metrics.last_token_ts - metrics.first_token_ts), 0.0
            )
            request_metrics.append(
                {
                    "ttft_ms": float(metrics.first_token_latency) * 1000.0,
                    "tpot_ms": decode_seconds * 1000.0 / decode_intervals,
                    "engine_metrics": dataclasses.asdict(metrics),
                }
            )
        generated = sum(len(ids) for ids in token_ids)
        return {
            "batch_size": batch_size,
            "phase": phase,
            "iteration": iteration,
            "wall_ms": wall_seconds * 1000.0,
            "aggregate_output_tokens_per_second": generated / wall_seconds,
            "median_request_ttft_ms": median(
                [item["ttft_ms"] for item in request_metrics]
            ),
            "median_request_tpot_ms": median(
                [item["tpot_ms"] for item in request_metrics]
            ),
            "generated_token_ids": token_ids,
            "request_metrics": request_metrics,
        }

    samples = []
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    for iteration in range(args.warmups):
        for batch_size in batch_sizes:
            samples.append(run_batch(batch_size, "warmup", iteration))
    for iteration in range(args.trials):
        order = (
            batch_sizes if iteration % 2 == 0 else list(reversed(batch_sizes))
        )
        for batch_size in order:
            samples.append(run_batch(batch_size, "measure", iteration))

    service_curve = []
    for batch_size in batch_sizes:
        measured = [
            sample
            for sample in samples
            if sample["phase"] == "measure" and sample["batch_size"] == batch_size
        ]
        service_curve.append(
            {
                "batch_size": batch_size,
                "median_wall_ms": median([sample["wall_ms"] for sample in measured]),
                "median_aggregate_output_tokens_per_second": median(
                    [
                        sample["aggregate_output_tokens_per_second"]
                        for sample in measured
                    ]
                ),
                "median_request_ttft_ms": median(
                    [sample["median_request_ttft_ms"] for sample in measured]
                ),
                "median_request_tpot_ms": median(
                    [sample["median_request_tpot_ms"] for sample in measured]
                ),
                "all_requests_exact_length": all(
                    len(ids) == args.new_tokens
                    for sample in measured
                    for ids in sample["generated_token_ids"]
                ),
            }
        )

    payload = {
        "schema_version": "qwen35-vllm-batch-service-v1",
        "status": "PASS",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "workload_identity": {
            "harness_sha256": sha256(Path(__file__).resolve()),
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "prompt_count": len(prompts),
            "request_selection": "prompt[(iteration + request_index) % prompt_count]",
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "vllm_cache_root": actual_vllm_cache_root,
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "vllm_sm89_bf16_lm_head": os.environ.get(
                "VLLM_SM89_BF16_LM_HEAD", "0(default)"
            ),
            "vllm_sm89_exact_packed_lm_head": os.environ.get(
                "VLLM_SM89_EXACT_PACKED_LM_HEAD", "0(default)"
            ),
            "vllm_sm89_marlin_w4_rerank": os.environ.get(
                "VLLM_SM89_MARLIN_W4_RERANK", "0(default)"
            ),
            "vllm_sm89_marlin_w4_scan_only": os.environ.get(
                "VLLM_SM89_MARLIN_W4_SCAN_ONLY", "0(default)"
            ),
            "guarded_source_sha256": guarded_sources,
        },
        "controls": {
            "quantization": args.quantization,
            "new_tokens": args.new_tokens,
            "warmups": args.warmups,
            "trials": args.trials,
            "ignore_eos": True,
            "sampling": "greedy",
            "engine_initialization_seconds": init_seconds,
            "case_order": "batch sizes forward/reverse by trial",
            "expected_vllm_cache_root": (
                str(args.expect_vllm_cache_root.expanduser().resolve())
                if args.expect_vllm_cache_root is not None
                else None
            ),
            "expected_marlin_w4_rerank": args.expect_marlin_w4_rerank,
            "actual_marlin_w4_rerank": actual_marlin_w4_rerank,
            "expected_marlin_w4_scan_only": args.expect_marlin_w4_scan_only,
            "actual_marlin_w4_scan_only": actual_marlin_w4_scan_only,
            "expected_bf16_lm_head": args.expect_bf16_lm_head,
            "actual_bf16_lm_head": actual_bf16_lm_head,
            "expected_exact_packed_lm_head": args.expect_exact_packed_lm_head,
            "actual_exact_packed_lm_head": actual_exact_packed_lm_head,
        },
        "service_curve": service_curve,
        "raw_samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "service_curve": service_curve}, indent=2))


if __name__ == "__main__":
    main()
