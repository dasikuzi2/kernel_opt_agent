#!/usr/bin/env python3
"""Summarize the bounded SM89 vLLM candidate search without overclaiming."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CANDIDATES = (
    ("frozen-vllm", "vllm_discovery_baseline_w1_n3.json", "REFERENCE"),
    ("cuda-maxseq1-fastloop", "vllm_confirm_fastloop_cuda_a_w1_n7.json", "RETAIN_FAST_LOOP"),
    ("triton-gdn-decode", "vllm_confirm_fastloop_triton_w1_n7.json", "SCREENED_OUT"),
    ("cuda-maxseq80", "vllm_confirm_maxseq80_cache512m_repeat_w1_n7.json", "INCONCLUSIVE_LATENCY"),
    ("chunked-prefill-off", "vllm_screen_fastloop_chunkoff_w1_n3.json", "SCREENED_OUT"),
    ("custom-ops-all", "vllm_screen_fastloop_customops_all_w1_n3.json", "SCREENED_OUT"),
    ("cuda-maxseq1-auto-profile", "vllm_confirm_maxseq1_autoprofile_w1_n3.json", "REFERENCE_FAST_LOOP"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted(cases: list[dict], field: str) -> float:
    return sum(float(case["weight"]) * float(case[field]) for case in cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    traces = args.traces.resolve()
    loaded = {}
    for candidate_id, filename, decision in CANDIDATES:
        path = traces / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded[candidate_id] = (path, payload, decision)

    baseline = loaded["frozen-vllm"][1]
    baseline_tokens = {
        case["case_id"]: case["generated_token_ids"] for case in baseline["cases"]
    }
    baseline_e2e = weighted(baseline["cases"], "median_end_to_end_ms")
    rows = []
    for candidate_id, (path, payload, decision) in loaded.items():
        cases = payload["cases"]
        exact = all(
            case["generated_token_ids"] == baseline_tokens[case["case_id"]]
            for case in cases
        )
        weighted_e2e = weighted(cases, "median_end_to_end_ms")
        rows.append(
            {
                "candidate_id": candidate_id,
                "decision": decision,
                "trace": {
                    "path": path.relative_to(args.output.resolve().parents[1]).as_posix(),
                    "sha256": digest(path),
                },
                "controls": payload["controls"],
                "exact_tokens_vs_frozen_baseline": exact,
                "weighted_median_end_to_end_ms": weighted_e2e,
                "weighted_median_ttft_ms": weighted(cases, "median_ttft_ms"),
                "weighted_median_tpot_ms": weighted(cases, "median_tpot_ms"),
                "cross_session_speedup_vs_frozen_baseline": baseline_e2e / weighted_e2e,
            }
        )

    by_id = {row["candidate_id"]: row for row in rows}
    fast = by_id["cuda-maxseq1-fastloop"]
    auto = by_id["cuda-maxseq1-auto-profile"]
    wide = by_id["cuda-maxseq80"]
    triton = by_id["triton-gdn-decode"]
    payload = {
        "schema_version": "qwen35-sm89-vllm-candidate-search-v1",
        "status": "PASS",
        "claim_scope": "DISCOVERY_ONLY_NOT_PAIRED_QUALIFICATION",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "conclusion": {
            "strict_bf16_inference_winner": "frozen-vllm",
            "fast_iteration_configuration": "cuda-maxseq1-fastloop",
            "new_strict_bf16_speedup_claim": "NO_MEASURABLE_GAIN_OVER_FROZEN_VLLM",
            "reason": (
                "No candidate beat the frozen vLLM baseline across sessions. "
                "The mature fused CUDA GDN decode remains faster than the Triton alternative."
            ),
        },
        "derived_comparisons": {
            "cached_fixed_cache_init_speedup_vs_cached_auto_profile": (
                float(auto["controls"]["engine_initialization_seconds"])
                / float(fast["controls"]["engine_initialization_seconds"])
            ),
            "maxseq1_init_speedup_vs_maxseq80": (
                float(wide["controls"]["engine_initialization_seconds"])
                / float(fast["controls"]["engine_initialization_seconds"])
            ),
            "cuda_weighted_e2e_speedup_vs_triton_gdn": (
                triton["weighted_median_end_to_end_ms"]
                / fast["weighted_median_end_to_end_ms"]
            ),
            "cuda_weighted_tpot_speedup_vs_triton_gdn": (
                triton["weighted_median_tpot_ms"]
                / fast["weighted_median_tpot_ms"]
            ),
        },
        "invalid_or_technical_paths": [
            {
                "candidate_id": "default-maxseq-with-512m-cache",
                "status": "TECHNICAL_FAILURE",
                "reason": "max_num_seqs=256 exceeds the 80 Mamba cache blocks available under the fixed cache budget",
            },
            {
                "candidate_id": "fp8-kv-cache",
                "status": "TECHNICAL_FAILURE",
                "reason": (
                    "FlashInfer JIT failed: system nvcc 12.0 rejects a CUDA-13 flag; "
                    "the venv nvcc 13.3 then conflicts with the installed CUDA 13.0 runtime headers"
                ),
                "performance_conclusion": "NONE",
            },
        ],
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], **payload["conclusion"], **payload["derived_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
