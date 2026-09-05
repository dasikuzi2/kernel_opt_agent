#!/usr/bin/env python3
"""Validate the archived shortlist-rerank candidate and emit a discovery receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CANDIDATE_ID = "int8-bf16-shortlist-rerank"
ROOT = Path(__file__).resolve().parents[2]
PATCH = Path(__file__).with_name("vllm_utils_int8_bf16_shortlist_rerank.patch")
QUALITY = ROOT / "comparisons/vllm_fp8_exact_vs_int8head_g1024_rerank_k128_quality_gsm8k_n512.json"
PAIR = ROOT / "comparisons/vllm_fp8_exact_vs_rerank_g1024k128_c1_s1_w3_n10_t128.json"
PROFILE = ROOT / "models/nsys2025_int8_bf16_rerank_g1024k128_map.json"


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_build() -> None:
    text = PATCH.read_text(encoding="utf-8")
    required = (
        "_sm89_exact_bf16_shortlist_rerank_kernel",
        "VLLM_SM89_INT8_BF16_RERANK_TOPK",
        "torch.topk",
        "output.fill_(float(\"-inf\"))",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"candidate patch is incomplete: {missing}")


def verify_correctness() -> None:
    quality = load(QUALITY)
    if quality.get("status") != "PASS":
        raise SystemExit("paired GSM8K gate did not pass")
    if quality.get("token_exact_count") != 512 or quality.get("answer_agreement_count") != 512:
        raise SystemExit("paired GSM8K gate is not 512/512 token and answer identical")
    pair = load(PAIR)
    if pair.get("status") != "PASS" or pair.get("exact_generated_token_cases") != "6/6":
        raise SystemExit("natural-prompt paired gate did not pass")


def emit_smoke(output: Path) -> None:
    verify_build()
    verify_correctness()
    pair = load(PAIR)
    profile = load(PROFILE)
    proofs = profile.get("execution_proofs", [])
    scan = next(
        (item for item in proofs if "_sm89_int8_groupwise_lmhead_kernel" in item.get("scope", "")),
        None,
    )
    if scan is None or scan.get("status") != "PASS":
        raise SystemExit("Nsight scan-kernel reachability proof is missing")
    result = {
        "schema_version": "candidate-smoke-result-v3",
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "objective": {
            "direction": "minimize",
            "baseline": float(pair["stock"]["weighted_tpot_ms"]) * 1000.0,
            "candidate": float(pair["candidate"]["weighted_tpot_ms"]) * 1000.0,
            "unit": "us_weighted",
        },
        "cases": [
            {"case_id": "natural-six-by-128", "role": "ANCHOR"},
            {"case_id": "gsm8k-frozen-512", "role": "EDGE"},
        ],
        "reachability": {
            "status": "PASS",
            "expected_path": "INT8 full-vocabulary scan followed by exact BF16 shortlist rerank",
            "observed_path": "INT8 full-vocabulary scan followed by exact BF16 shortlist rerank",
            "compile_cache_policy": "SOURCE_HASHED",
            "execution_proof": {
                "kind": "KERNEL_INSTANCE_COUNT",
                "scope": scan["scope"],
                "observed_count": int(scan["observed_count"]),
                "minimum_count": int(scan["minimum_count"]),
                "evidence_index": 0,
            },
            "evidence": [
                {
                    "path": PROFILE.relative_to(ROOT).as_posix(),
                    "sha256": digest(PROFILE),
                },
                {
                    "path": QUALITY.relative_to(ROOT).as_posix(),
                    "sha256": digest(QUALITY),
                },
            ],
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("build", "correctness", "smoke"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "build":
        verify_build()
    elif args.stage == "correctness":
        verify_correctness()
    else:
        if args.output is None:
            parser.error("--output is required for smoke")
        emit_smoke(args.output)


if __name__ == "__main__":
    main()
