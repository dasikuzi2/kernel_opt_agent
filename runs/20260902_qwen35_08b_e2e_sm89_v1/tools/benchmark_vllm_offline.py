#!/usr/bin/env python3
"""Bounded, fresh-state vLLM discovery baseline for frozen Qwen3.5 cases."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SYNTHETIC_CASES = (
    ("prompt-128-generate-128", 128, 0.2),
    ("prompt-512-generate-128", 512, 0.3),
    ("prompt-2048-generate-128", 2048, 0.5),
)

NATURAL_REQUESTS = (
    (
        "zh-explanation",
        "请用通俗但准确的语言解释：为什么大语言模型逐 token 解码通常受显存带宽限制？给出一个简单的数量级估算。",
    ),
    (
        "python-code",
        "Write a complete Python function that merges overlapping half-open intervals. Include type hints, a concise explanation, and three edge-case tests.",
    ),
    (
        "reasoning",
        "A shop discounts an item by 20%, then raises the discounted price by 25%. Explain step by step whether the final price equals the original price.",
    ),
    (
        "editing",
        "Rewrite the following paragraph to be concise and professional without losing any facts: Our team ran several experiments over the last two weeks, but because each trial used a different environment and no frozen baseline, the reported improvements cannot yet be compared reliably.",
    ),
    (
        "systems-design",
        "Design a bounded GPU-kernel optimization loop for an autonomous agent. Focus on candidate diversity, fail-fast checks, correctness gates, measurement noise, and stopping rules.",
    ),
    (
        "translation",
        "Translate into natural Chinese and briefly explain the technical meaning: Speculative decoding reduces latency only when accepted draft tokens amortize the verification cost.",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def weight_manifest_sha256(model_path: Path) -> str:
    """Hash every safetensors shard without assuming a checkpoint filename."""
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors weights found under {model_path}")
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(shard.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(shard).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def exact_prompt(tokenizer, length: int) -> list[int]:
    seed_text = "Kernel optimization must preserve semantics while reducing global latency. "
    seed = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed:
        raise RuntimeError("tokenizer produced an empty seed")
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def generation_sanity(token_ids: list[int]) -> dict:
    """Catch deterministic-but-useless short cycles before scoring speed."""
    distinct = len(set(token_ids))
    distinct_fraction = distinct / max(len(token_ids), 1)
    min_distinct = min(8, max(len(token_ids) // 4, 2))
    sane = len(token_ids) > 0 and distinct >= min_distinct
    return {
        "status": "PASS" if sane else "FAIL",
        "distinct_token_count": distinct,
        "distinct_token_fraction": distinct_fraction,
        "minimum_distinct_tokens": min_distinct,
        "reason": None if sane else "degenerate low-diversity token cycle",
    }


def nvcc_release() -> str | None:
    executable = shutil.which("nvcc")
    if executable is None:
        return None
    result = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False
    )
    match = re.search(r"release\s+(\d+\.\d+)", result.stdout + result.stderr)
    return match.group(1) if match else None


def gpu_telemetry() -> dict | None:
    """Take a cheap point sample so power-state drift cannot stay invisible."""
    fields = (
        "pstate,power.draw,clocks.current.graphics,clocks.current.memory,"
        "temperature.gpu,utilization.gpu"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [
            value.strip()
            for value in completed.stdout.splitlines()[0].split(",")
        ]
        if len(values) != 6:
            raise ValueError(f"unexpected nvidia-smi row: {values!r}")
        return {
            "pstate": values[0],
            "power_w": float(values[1]),
            "graphics_clock_mhz": float(values[2]),
            "memory_clock_mhz": float(values[3]),
            "temperature_c": float(values[4]),
            "utilization_percent": float(values[5]),
        }
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return None


def expected_source_hashes(specs: list[str]) -> dict[str, str]:
    """Validate PATH=SHA256 guards before an expensive engine startup."""
    observed = {}
    for spec in specs:
        try:
            raw_path, expected = spec.rsplit("=", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid --expect-source-sha256 {spec!r}; expected PATH=SHA256"
            ) from exc
        path = Path(raw_path).expanduser().resolve()
        expected = expected.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid SHA256 in source guard: {expected!r}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                "candidate source mismatch: "
                f"{path} expected {expected}, observed {actual}"
            )
        observed[str(path)] = actual
    return observed


def require_empty_vllm_cache_root(cache_root_raw: str | None) -> str:
    """Reject a candidate run that could silently reuse a compiled graph."""
    if not cache_root_raw:
        raise RuntimeError(
            "--require-empty-vllm-cache-root needs an explicit VLLM_CACHE_ROOT"
        )
    cache_root = Path(cache_root_raw).expanduser().resolve()
    if cache_root.exists() and any(cache_root.iterdir()):
        raise RuntimeError(
            f"candidate path may be stale: VLLM_CACHE_ROOT is not empty: {cache_root}"
        )
    return str(cache_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--operator-contract",
        type=Path,
        help="Hash-bind the frozen operator contract governing this run.",
    )
    parser.add_argument(
        "--workload-contract",
        type=Path,
        help="Hash-bind the frozen workload contract governing this run.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional tokenizer/template path; useful for incomplete quantized checkpoints.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt-suite", choices=("synthetic", "natural"), default="synthetic"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--cudagraph-mode",
        choices=("default", "none"),
        default="default",
        help="Keep torch.compile enabled while independently disabling CUDA Graphs.",
    )
    parser.add_argument(
        "--expect-gdn-decode-kernel",
        choices=("cuda", "triton"),
        help="Fail before engine startup unless the requested candidate path is selected.",
    )
    parser.add_argument(
        "--expect-sm89-lm-head",
        choices=("stock", "triton"),
        help="Fail before engine startup unless the requested lm_head path is selected.",
    )
    parser.add_argument(
        "--expect-exact-packed-lm-head",
        choices=("off", "on"),
        help="Fail unless the exact packed-BF16 lm_head switch matches.",
    )
    parser.add_argument(
        "--expect-marlin-w4-rerank",
        choices=("off", "on"),
        help="Fail unless the SM89 Marlin-W4 shortlist/rerank switch matches.",
    )
    parser.add_argument(
        "--expect-marlin-w4-scan-only",
        choices=("off", "on"),
        help="Fail unless the SM89 Marlin-W4 output-head scan-only switch matches.",
    )
    parser.add_argument(
        "--gpu-telemetry",
        action="store_true",
        help="Record nvidia-smi point samples immediately before and after each request.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Wrap only measured requests in cudaProfilerStart/Stop for nsys capture.",
    )
    parser.add_argument(
        "--expect-source-sha256",
        action="append",
        default=[],
        metavar="PATH=SHA256",
        help="Fail before engine startup unless a candidate source file has this hash.",
    )
    parser.add_argument(
        "--require-empty-vllm-cache-root",
        action="store_true",
        help=(
            "Require VLLM_CACHE_ROOT to name an absent or empty directory, proving "
            "that a source candidate is compiled into a fresh graph."
        ),
    )
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--speculative-tokens",
        type=int,
        default=0,
        help="Enable the model's native MTP proposer with this many draft tokens.",
    )
    parser.add_argument(
        "--ngram-speculative-tokens",
        type=int,
        default=0,
        help="Enable the zero-weight n-gram proposer with this many draft tokens.",
    )
    parser.add_argument("--ngram-prompt-lookup-min", type=int, default=2)
    parser.add_argument("--ngram-prompt-lookup-max", type=int, default=5)
    parser.add_argument(
        "--chunked-prefill",
        choices=("default", "on", "off"),
        default="default",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "fp8"),
        default="auto",
    )
    parser.add_argument(
        "--custom-ops",
        choices=("default", "all", "none"),
        default="default",
    )
    parser.add_argument(
        "--fuse-norm-quant",
        action="store_true",
        help="Enable vLLM's RMSNorm-to-quantization graph rewrite.",
    )
    parser.add_argument(
        "--fuse-act-quant",
        action="store_true",
        help="Enable vLLM's activation-to-quantization graph rewrite.",
    )
    parser.add_argument(
        "--fuse-attn-quant",
        action="store_true",
        help="Enable vLLM's attention-to-quantization graph rewrite.",
    )
    parser.add_argument(
        "--quantization",
        choices=(
            "none",
            "fp8",
            "fp8_per_tensor",
            "fp8_per_block",
            "fp8_per_channel",
            "int8_per_channel_weight_only",
            "compressed-tensors",
            "auto_gptq",
            "gptq_marlin",
        ),
        default="none",
    )
    parser.add_argument(
        "--linear-backend",
        choices=(
            "auto",
            "marlin",
            "triton",
            "conch",
            "exllama",
            "humming",
            "allspark",
            "machete",
            "cutlass",
        ),
        default="auto",
        help="Route vLLM through one registered linear-kernel family.",
    )
    parser.add_argument(
        "--lm-head-backend",
        choices=("torch", "lossless_packed"),
        default="torch",
        help="Select the dedicated unquantized lm-head backend.",
    )
    parser.add_argument(
        "--lm-head-max-packed-fraction",
        type=float,
        default=0.90,
        help="Fallback unless the lossless auxiliary layout is at most this fraction of dense BF16 bytes.",
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.trials < 1:
        raise ValueError("warmups and trials must both be positive")
    for label, contract_path in (
        ("operator", args.operator_contract),
        ("workload", args.workload_contract),
    ):
        if contract_path is not None and not contract_path.is_file():
            raise FileNotFoundError(
                f"{label} contract must be a readable file: {contract_path}"
            )
    if not 0.5 <= args.lm_head_max_packed_fraction <= 1.0:
        raise ValueError("lm-head-max-packed-fraction must be in [0.5, 1.0]")
    if args.max_num_seqs is not None and args.max_num_seqs < 1:
        raise ValueError("max-num-seqs must be positive")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens < 1:
        raise ValueError("max-num-batched-tokens must be positive")
    if args.speculative_tokens < 0:
        raise ValueError("speculative-tokens must be non-negative")
    if args.ngram_speculative_tokens < 0:
        raise ValueError("ngram-speculative-tokens must be non-negative")
    if args.speculative_tokens and args.ngram_speculative_tokens:
        raise ValueError("MTP and n-gram speculation are mutually exclusive")
    guarded_sources = expected_source_hashes(args.expect_source_sha256)
    cache_root_raw = os.environ.get("VLLM_CACHE_ROOT")
    if args.require_empty_vllm_cache_root:
        cache_root_raw = require_empty_vllm_cache_root(cache_root_raw)
    actual_gdn_kernel = os.environ.get("VLLM_GDN_DECODE_KERNEL", "cuda").lower()
    if (
        args.expect_gdn_decode_kernel is not None
        and actual_gdn_kernel != args.expect_gdn_decode_kernel
    ):
        raise RuntimeError(
            "candidate path is unreachable: "
            f"expected VLLM_GDN_DECODE_KERNEL={args.expect_gdn_decode_kernel}, "
            f"got {actual_gdn_kernel!r}"
        )
    actual_sm89_lm_head = (
        "triton"
        if os.environ.get("VLLM_SM89_BF16_LM_HEAD", "0") == "1"
        else "stock"
    )
    if (
        args.expect_sm89_lm_head is not None
        and actual_sm89_lm_head != args.expect_sm89_lm_head
    ):
        raise RuntimeError(
            "candidate path is unreachable: "
            f"expected SM89 lm_head={args.expect_sm89_lm_head}, "
            f"got {actual_sm89_lm_head}"
        )
    actual_exact_packed_lm_head = (
        "on"
        if os.environ.get("VLLM_SM89_EXACT_PACKED_LM_HEAD", "0") == "1"
        else "off"
    )
    if (
        args.expect_exact_packed_lm_head is not None
        and actual_exact_packed_lm_head != args.expect_exact_packed_lm_head
    ):
        raise RuntimeError(
            "candidate path is unreachable: expected exact-packed lm_head "
            f"{args.expect_exact_packed_lm_head}, got {actual_exact_packed_lm_head}"
        )
    actual_marlin_w4_rerank = (
        "on"
        if os.environ.get("VLLM_SM89_MARLIN_W4_RERANK", "0") == "1"
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
    actual_marlin_w4_scan_only = (
        "on"
        if os.environ.get("VLLM_SM89_MARLIN_W4_SCAN_ONLY", "0") == "1"
        else "off"
    )
    if (
        args.expect_marlin_w4_scan_only is not None
        and actual_marlin_w4_scan_only != args.expect_marlin_w4_scan_only
    ):
        raise RuntimeError(
            "candidate path is unreachable: expected Marlin-W4 scan-only "
            f"{args.expect_marlin_w4_scan_only}, got {actual_marlin_w4_scan_only}"
        )
    if args.ngram_prompt_lookup_min < 1:
        raise ValueError("ngram-prompt-lookup-min must be positive")
    if args.ngram_prompt_lookup_max < args.ngram_prompt_lookup_min:
        raise ValueError("ngram-prompt-lookup-max must be >= lookup-min")
    if args.kv_cache_memory_bytes is not None:
        if args.kv_cache_memory_bytes < 1:
            raise ValueError("kv-cache-memory-bytes must be positive")
        if args.max_num_seqs is None:
            raise ValueError(
                "fixed KV cache requires explicit --max-num-seqs so the high "
                "engine default cannot cause an avoidable late Mamba-cache failure"
            )
    import torch

    selected_nvcc = nvcc_release()
    if args.kv_cache_dtype == "fp8":
        runtime_cuda = str(torch.version.cuda or "")
        if selected_nvcc is None or selected_nvcc != runtime_cuda:
            raise RuntimeError(
                "FP8 KV cache requires FlashInfer JIT in this environment, but "
                f"nvcc={selected_nvcc!r} and torch CUDA runtime={runtime_cuda!r}; "
                "align compiler and runtime headers before launching the engine"
            )

    import transformers
    import triton
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = args.model.resolve()
    tokenizer_path = (
        args.tokenizer.resolve() if args.tokenizer is not None else model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if args.prompt_suite == "synthetic":
        cases = list(SYNTHETIC_CASES)
        prompts = {
            case_id: exact_prompt(tokenizer, length)
            for case_id, length, _ in cases
        }
    else:
        prompts = {}
        for case_id, request in NATURAL_REQUESTS:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": request}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts[case_id] = tokenizer.encode(rendered, add_special_tokens=False)
            if not prompts[case_id] or not all(
                isinstance(token_id, int) for token_id in prompts[case_id]
            ):
                raise TypeError(f"chat template produced invalid token IDs for {case_id}")
        weight = 1.0 / len(NATURAL_REQUESTS)
        cases = [
            (case_id, len(prompts[case_id]), weight)
            for case_id, _ in NATURAL_REQUESTS
        ]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.new_tokens,
        ignore_eos=True,
        detokenize=False,
        seed=20260902,
    )

    init_started = time.perf_counter()
    engine_overrides = {}
    if args.linear_backend != "auto":
        engine_overrides["linear_backend"] = args.linear_backend
    if args.lm_head_backend != "torch":
        engine_overrides["kernel_config"] = {
            "lm_head_backend": args.lm_head_backend,
            "lm_head_max_packed_fraction": args.lm_head_max_packed_fraction,
        }
    if args.max_num_seqs is not None:
        engine_overrides["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        engine_overrides["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.chunked_prefill != "default":
        engine_overrides["enable_chunked_prefill"] = args.chunked_prefill == "on"
    compilation_config = {}
    if args.cudagraph_mode != "default":
        compilation_config["cudagraph_mode"] = args.cudagraph_mode.upper()
    if args.custom_ops != "default":
        compilation_config["custom_ops"] = [args.custom_ops]
    if args.fuse_norm_quant or args.fuse_act_quant or args.fuse_attn_quant:
        compilation_config["pass_config"] = {
            "fuse_norm_quant": args.fuse_norm_quant,
            "fuse_act_quant": args.fuse_act_quant,
            "fuse_attn_quant": args.fuse_attn_quant,
        }
    if compilation_config:
        engine_overrides["compilation_config"] = compilation_config
    if args.speculative_tokens:
        engine_overrides["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.speculative_tokens,
        }
        # Acceptance is the central feasibility signal for speculative decode.
        # Collecting only latency lets repetitive synthetic text look like a
        # general architecture win and gives the optimizer no way to separate
        # proposer cost from rejected drafts.
        engine_overrides["per_request_spec_decode_metrics"] = "summary"
    elif args.ngram_speculative_tokens:
        engine_overrides["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": args.ngram_speculative_tokens,
            "prompt_lookup_min": args.ngram_prompt_lookup_min,
            "prompt_lookup_max": args.ngram_prompt_lookup_max,
        }
        engine_overrides["per_request_spec_decode_metrics"] = "summary"

    llm = LLM(
        model=str(model_path),
        tokenizer=str(tokenizer_path),
        dtype="bfloat16",
        quantization=None if args.quantization == "none" else args.quantization,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        enforce_eager=args.enforce_eager,
        language_model_only=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        kv_cache_dtype=args.kv_cache_dtype,
        seed=20260902,
        **engine_overrides,
    )
    init_seconds = time.perf_counter() - init_started

    def request(case_id: str, phase: str, iteration: int) -> dict:
        telemetry_before = gpu_telemetry() if args.gpu_telemetry else None
        started = time.perf_counter()
        outputs = llm.generate({"prompt_token_ids": prompts[case_id]}, params, use_tqdm=False)
        wall_seconds = time.perf_counter() - started
        telemetry_after = gpu_telemetry() if args.gpu_telemetry else None
        output = outputs[0]
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)
        spec_decode_metrics = (
            completion.spec_decode_metrics.to_dict()
            if completion.spec_decode_metrics is not None
            else None
        )
        stats = output.metrics
        if stats is None:
            raise RuntimeError("vLLM request metrics are unavailable with disable_log_stats=False")
        raw_stats = dataclasses.asdict(stats)
        decode_intervals = max(len(token_ids) - 1, 1)
        decode_seconds = max(float(stats.last_token_ts - stats.first_token_ts), 0.0)
        return {
            "case_id": case_id,
            "phase": phase,
            "iteration": iteration,
            "prompt_tokens": len(prompts[case_id]),
            "generated_tokens": len(token_ids),
            "generated_token_ids": token_ids,
            "end_to_end_ms": wall_seconds * 1000.0,
            "ttft_ms": float(stats.first_token_latency) * 1000.0,
            "tpot_ms": decode_seconds * 1000.0 / decode_intervals,
            "output_tokens_per_second": decode_intervals / decode_seconds if decode_seconds > 0 else None,
            "engine_metrics": raw_stats,
            "spec_decode_metrics": spec_decode_metrics,
            "gpu_telemetry_before": telemetry_before,
            "gpu_telemetry_after": telemetry_after,
        }

    samples: list[dict] = []
    # Rotate the case order so clock/thermal drift cannot systematically favor one shape.
    case_ids = [item[0] for item in cases]
    for iteration in range(args.warmups):
        order = case_ids[iteration % len(case_ids):] + case_ids[: iteration % len(case_ids)]
        for case_id in order:
            samples.append(request(case_id, "warmup", iteration))
    if args.cuda_profiler_range:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
    for iteration in range(args.trials):
        shift = (iteration + args.warmups) % len(case_ids)
        order = case_ids[shift:] + case_ids[:shift]
        for case_id in order:
            samples.append(request(case_id, "measure", iteration))
    if args.cuda_profiler_range:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    summaries = []
    for case_id, prompt_tokens, weight in cases:
        measured = [sample for sample in samples if sample["case_id"] == case_id and sample["phase"] == "measure"]
        reference_ids = measured[0]["generated_token_ids"]
        exact_repeat = all(sample["generated_token_ids"] == reference_ids for sample in measured)
        sanity = generation_sanity(reference_ids)
        spec_metrics = [
            sample["spec_decode_metrics"]
            for sample in measured
            if sample["spec_decode_metrics"] is not None
        ]
        spec_summary = None
        if spec_metrics:
            total_steps = sum(item["num_spec_steps"] for item in spec_metrics)
            total_accepted = sum(
                item["num_accepted_draft_tokens"] for item in spec_metrics
            )
            total_drafted = sum(item["num_draft_tokens"] for item in spec_metrics)
            spec_summary = {
                "num_spec_steps": total_steps,
                "num_accepted_draft_tokens": total_accepted,
                "num_draft_tokens": total_drafted,
                "mean_acceptance_length": (
                    1.0 + total_accepted / total_steps if total_steps else 1.0
                ),
                "draft_acceptance_rate": (
                    total_accepted / total_drafted if total_drafted else 0.0
                ),
            }
        summaries.append({
            "case_id": case_id,
            "weight": weight,
            "prompt_tokens": prompt_tokens,
            "prompt_token_ids_sha256": hashlib.sha256(
                json.dumps(prompts[case_id], separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "generated_tokens": args.new_tokens,
            "correctness": (
                "PASS"
                if exact_repeat
                and len(reference_ids) == args.new_tokens
                and sanity["status"] == "PASS"
                else "FAIL"
            ),
            "generation_sanity": sanity,
            "generated_token_ids": reference_ids,
            "median_end_to_end_ms": median([sample["end_to_end_ms"] for sample in measured]),
            "median_ttft_ms": median([sample["ttft_ms"] for sample in measured]),
            "median_tpot_ms": median([sample["tpot_ms"] for sample in measured]),
            "median_output_tokens_per_second": median([sample["output_tokens_per_second"] for sample in measured]),
            "spec_decode_metrics": spec_summary,
        })

    payload = {
        "schema_version": "qwen35-vllm-discovery-baseline-v1",
        "status": "PASS" if all(case["correctness"] == "PASS" for case in summaries) else "FAIL",
        "claim_scope": (
            "CONTRACT_BOUND_DISCOVERY_BASELINE_ONLY_NOT_QUALIFICATION"
            if args.operator_contract is not None and args.workload_contract is not None
            else (
                "DISCOVERY_BASELINE_ONLY_NOT_QUALIFICATION"
                if args.quantization == "none"
                else "EXPLORATORY_QUANTIZED_DISCOVERY_REQUIRES_NEW_NUMERICS_CONTRACT"
            )
        ),
        "contracts": {
            "operator": (
                {
                    "path": str(args.operator_contract.resolve()),
                    "sha256": sha256(args.operator_contract.resolve()),
                }
                if args.operator_contract is not None
                else None
            ),
            "workload": (
                {
                    "path": str(args.workload_contract.resolve()),
                    "sha256": sha256(args.workload_contract.resolve()),
                }
                if args.workload_contract is not None
                else None
            ),
        },
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "tokenizer_path": str(tokenizer_path),
            "config_sha256": sha256(model_path / "config.json"),
            "weight_manifest_sha256": weight_manifest_sha256(model_path),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvcc_release": selected_nvcc,
            "triton": triton.__version__,
            "transformers": transformers.__version__,
            "linear_backend": args.linear_backend,
            "lm_head_backend": args.lm_head_backend,
            "lm_head_max_packed_fraction": args.lm_head_max_packed_fraction,
            "vllm_use_v2_model_runner": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
            "vllm_use_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
            "vllm_gdn_decode_kernel": os.environ.get("VLLM_GDN_DECODE_KERNEL", "cuda(default)"),
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
            "vllm_sm89_int8_groupwise_lm_head": os.environ.get(
                "VLLM_SM89_INT8_GROUPWISE_LM_HEAD", "off(default)"
            ),
            "vllm_sm89_int8_bf16_rerank_topk": os.environ.get(
                "VLLM_SM89_INT8_BF16_RERANK_TOPK", "off(default)"
            ),
            "vllm_sm89_fused_swiglu_down": os.environ.get(
                "VLLM_SM89_FUSED_SWIGLU_DOWN", "0(default)"
            ),
            "vllm_sm89_bf16_gemv": os.environ.get(
                "VLLM_SM89_BF16_GEMV", "none(default)"
            ),
            "vllm_sm89_segmented_gdn_projection": os.environ.get(
                "VLLM_SM89_SEGMENTED_GDN_PROJECTION", "0(default)"
            ),
            "vllm_enable_fla_packed_recurrent_decode": os.environ.get(
                "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "1(default)"
            ),
            "vllm_cache_root": cache_root_raw,
            "guarded_source_sha256": guarded_sources,
        },
        "controls": {
            "language_model_only": True,
            "dtype": "bfloat16",
            "quantization": args.quantization,
            "lm_head_backend": args.lm_head_backend,
            "lm_head_max_packed_fraction": args.lm_head_max_packed_fraction,
            "prompt_suite": args.prompt_suite,
            "max_model_len": 4096,
            "enable_prefix_caching": False,
            "enforce_eager": args.enforce_eager,
            "cudagraph_mode": args.cudagraph_mode,
            "expected_gdn_decode_kernel": args.expect_gdn_decode_kernel,
            "actual_gdn_decode_kernel": actual_gdn_kernel,
            "expected_sm89_lm_head": args.expect_sm89_lm_head,
            "actual_sm89_lm_head": actual_sm89_lm_head,
            "expected_exact_packed_lm_head": args.expect_exact_packed_lm_head,
            "actual_exact_packed_lm_head": actual_exact_packed_lm_head,
            "expected_marlin_w4_rerank": args.expect_marlin_w4_rerank,
            "actual_marlin_w4_rerank": actual_marlin_w4_rerank,
            "expected_marlin_w4_scan_only": args.expect_marlin_w4_scan_only,
            "actual_marlin_w4_scan_only": actual_marlin_w4_scan_only,
            "gpu_telemetry": args.gpu_telemetry,
            "cuda_profiler_range": args.cuda_profiler_range,
            "require_empty_vllm_cache_root": args.require_empty_vllm_cache_root,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "speculative_tokens": args.speculative_tokens,
            "per_request_spec_decode_metrics": (
                "summary"
                if args.speculative_tokens or args.ngram_speculative_tokens
                else "none"
            ),
            "ngram_speculative_tokens": args.ngram_speculative_tokens,
            "ngram_prompt_lookup_min": args.ngram_prompt_lookup_min,
            "ngram_prompt_lookup_max": args.ngram_prompt_lookup_max,
            "chunked_prefill": args.chunked_prefill,
            "kv_cache_dtype": args.kv_cache_dtype,
            "custom_ops": args.custom_ops,
            "fuse_norm_quant": args.fuse_norm_quant,
            "fuse_act_quant": args.fuse_act_quant,
            "fuse_attn_quant": args.fuse_attn_quant,
            "warmups": args.warmups,
            "trials": args.trials,
            "case_order": "rotated per iteration",
            "sampling": (
                f"greedy temperature=0, ignore_eos=True, exact {args.new_tokens} tokens"
            ),
            "engine_initialization_seconds": init_seconds,
        },
        "cases": summaries,
        "raw_samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "controls": payload["controls"], "cases": summaries}, indent=2))


if __name__ == "__main__":
    main()
