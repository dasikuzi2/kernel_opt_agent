#!/usr/bin/env python3
"""Summarize the measured vLLM/llama.cpp latency-quality frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_e2e(payload: dict) -> float:
    aggregate = payload.get("aggregate")
    if aggregate:
        return float(aggregate["weighted_median_end_to_end_ms"])
    return sum(
        float(case["weight"]) * float(case["median_end_to_end_ms"])
        for case in payload["cases"]
    )


def weighted_tps(payload: dict) -> float:
    aggregate = payload.get("aggregate")
    if aggregate:
        return float(aggregate["weighted_median_output_tokens_per_second"])
    return sum(
        float(case["weight"]) * float(case["median_output_tokens_per_second"])
        for case in payload["cases"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-fast", required=True, type=Path)
    parser.add_argument("--vllm-slow", required=True, type=Path)
    parser.add_argument("--q8-fast", required=True, type=Path)
    parser.add_argument("--q4-fast", required=True, type=Path)
    parser.add_argument("--q4-slow", required=True, type=Path)
    parser.add_argument("--llamacpp-bf16", required=True, type=Path)
    parser.add_argument("--q8-mtp", required=True, type=Path)
    parser.add_argument("--power-probe", required=True, type=Path)
    parser.add_argument("--ppl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = {
        "vllm_fast": args.vllm_fast,
        "vllm_slow": args.vllm_slow,
        "q8_fast": args.q8_fast,
        "q4_fast": args.q4_fast,
        "q4_slow": args.q4_slow,
        "llamacpp_bf16": args.llamacpp_bf16,
        "q8_mtp": args.q8_mtp,
        "power_probe": args.power_probe,
        "ppl": args.ppl,
    }
    payloads = {label: load(path) for label, path in paths.items()}
    rows = {}
    for label in (
        "vllm_fast",
        "vllm_slow",
        "q8_fast",
        "q4_fast",
        "q4_slow",
        "llamacpp_bf16",
        "q8_mtp",
    ):
        rows[label] = {
            "weighted_e2e_ms": weighted_e2e(payloads[label]),
            "weighted_decode_tokens_per_second": weighted_tps(payloads[label]),
            "status": payloads[label]["status"],
            "evidence": str(paths[label]),
            "evidence_sha256": sha256(paths[label]),
        }

    vllm_fast = rows["vllm_fast"]["weighted_e2e_ms"]
    vllm_slow = rows["vllm_slow"]["weighted_e2e_ms"]
    q8_fast = rows["q8_fast"]["weighted_e2e_ms"]
    q4_fast = rows["q4_fast"]["weighted_e2e_ms"]
    q4_slow = rows["q4_slow"]["weighted_e2e_ms"]
    ppl = {
        item["label"]: item for item in payloads["ppl"]["results"]
    }
    telemetry = payloads["power_probe"]["gpu_telemetry"]
    summary = {
        "schema_version": "qwen35-runtime-frontier-v1",
        "status": "DISCOVERY_COMPLETE_QUALIFICATION_PENDING",
        "claim_scope": "RTX_4060_LAPTOP_BATCH1_NATURAL_SUITE_128_OUTPUT_TOKENS",
        "decision": {
            "strict_bf16": (
                "Keep vLLM. The llama.cpp BF16 route is slower, and no strict "
                "token-parity vLLM-beating kernel was found."
            ),
            "quality_first_quantized": (
                "Q8_0 llama.cpp is the leading discovery candidate; its "
                "8-chunk corpus perplexity is within 0.16% of llama.cpp BF16."
            ),
            "latency_first_quantized": (
                "Q4_0 llama.cpp is fastest in the high-power observation but "
                "raises corpus perplexity by 9.59%."
            ),
            "speculation": (
                "Reject MTP-1 and natural-prompt n-gram speculation for this "
                "0.8B batch-1 workload."
            ),
        },
        "comparisons": {
            "high_power_observation": {
                "vllm_bf16_ms": vllm_fast,
                "llamacpp_q8_ms": q8_fast,
                "llamacpp_q4_ms": q4_fast,
                "q8_speedup_vs_vllm": vllm_fast / q8_fast,
                "q4_speedup_vs_vllm": vllm_fast / q4_fast,
                "qualification": "not power-controlled",
            },
            "degraded_power_adjacent_observation": {
                "vllm_bf16_ms": vllm_slow,
                "llamacpp_q4_ms": q4_slow,
                "q4_speedup_vs_vllm": vllm_slow / q4_slow,
                "qualification": "adjacent runs, not randomized interleaving",
            },
            "quality_screen": {
                "bf16_perplexity": ppl["bf16"]["perplexity"],
                "q8_perplexity": ppl["q8_0"]["perplexity"],
                "q4_perplexity": ppl["q4_0"]["perplexity"],
                "q8_relative_to_bf16": ppl["q8_0"][
                    "relative_perplexity_vs_bf16"
                ],
                "q4_relative_to_bf16": ppl["q4_0"][
                    "relative_perplexity_vs_bf16"
                ],
                "qualification": "8-chunk in-domain corpus discovery only",
            },
        },
        "measurements": rows,
        "measurement_risk": {
            "observed_absolute_latency_drift": vllm_slow / vllm_fast,
            "power_probe_sample_count": len(telemetry),
            "power_probe_median_power_w": statistics.median(
                sample["power_w"] for sample in telemetry
            ),
            "power_probe_median_graphics_clock_mhz": statistics.median(
                sample["graphics_clock_mhz"] for sample in telemetry
            ),
            "power_probe_median_memory_clock_mhz": statistics.median(
                sample["memory_clock_mhz"] for sample in telemetry
            ),
            "power_probe_median_gpu_utilization_percent": statistics.median(
                sample["gpu_utilization_percent"] for sample in telemetry
            ),
            "required_next_gate": (
                "Lock power mode, record clocks/power per sample, randomize "
                "runtime order, then run at least 3 warmups and 10 trials."
            ),
        },
        "evidence": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["comparisons"], indent=2))


if __name__ == "__main__":
    main()
