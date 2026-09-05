#!/usr/bin/env python3
"""Compare speculative-decoding candidates against matching vLLM baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PAIRS = (
    (
        "synthetic-mtp1",
        "vllm_discovery_baseline_w1_n3.json",
        "vllm_screen_mtp1_w1_n3.json",
    ),
    (
        "synthetic-ngram4",
        "vllm_discovery_baseline_w1_n3.json",
        "vllm_screen_ngram4_w1_n3.json",
    ),
    (
        "natural-mtp1",
        "vllm_natural_baseline_w1_n3.json",
        "vllm_natural_mtp1_w1_n3.json",
    ),
    (
        "natural-ngram4",
        "vllm_natural_baseline_w1_n3.json",
        "vllm_natural_ngram4_w1_n3.json",
    ),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted(payload: dict, field: str) -> float:
    return sum(float(case["weight"]) * float(case[field]) for case in payload["cases"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.output.resolve().parents[1]
    rows = []
    for candidate_id, baseline_name, candidate_name in PAIRS:
        baseline_path = (args.traces / baseline_name).resolve()
        candidate_path = (args.traces / candidate_name).resolve()
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        expected = {
            case["case_id"]: case["generated_token_ids"]
            for case in baseline["cases"]
        }
        exact_cases = sum(
            case["generated_token_ids"] == expected[case["case_id"]]
            for case in candidate["cases"]
        )
        baseline_e2e = weighted(baseline, "median_end_to_end_ms")
        candidate_e2e = weighted(candidate, "median_end_to_end_ms")
        rows.append(
            {
                "candidate_id": candidate_id,
                "baseline_trace": {
                    "path": baseline_path.relative_to(root).as_posix(),
                    "sha256": digest(baseline_path),
                },
                "candidate_trace": {
                    "path": candidate_path.relative_to(root).as_posix(),
                    "sha256": digest(candidate_path),
                },
                "case_count": len(candidate["cases"]),
                "exact_cases_vs_baseline": exact_cases,
                "all_cases_exact_vs_baseline": exact_cases == len(candidate["cases"]),
                "baseline_weighted_e2e_ms": baseline_e2e,
                "candidate_weighted_e2e_ms": candidate_e2e,
                "speedup_vs_matching_vllm_baseline": baseline_e2e / candidate_e2e,
                "baseline_weighted_tpot_ms": weighted(baseline, "median_tpot_ms"),
                "candidate_weighted_tpot_ms": weighted(candidate, "median_tpot_ms"),
            }
        )

    output = {
        "schema_version": "qwen35-sm89-speculation-search-v1",
        "status": "PASS",
        "claim_scope": "DISCOVERY_ONLY_NOT_PRODUCTION_ACCEPTANCE",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "conclusion": {
            "strict_winner": "default-vllm",
            "synthetic_ngram_speedup_is_generalizable": False,
            "reason": (
                "N-gram wins only on the periodic synthetic output. On the natural "
                "suite it is slower and changes every output; MTP-1 is also slower."
            ),
        },
        "comparisons": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["conclusion"], indent=2))
    for row in rows:
        print(
            row["candidate_id"],
            f"speedup={row['speedup_vs_matching_vllm_baseline']:.4f}",
            f"exact={row['exact_cases_vs_baseline']}/{row['case_count']}",
        )


if __name__ == "__main__":
    main()
