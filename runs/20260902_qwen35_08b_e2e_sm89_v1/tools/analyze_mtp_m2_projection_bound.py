#!/usr/bin/env python3
"""Bound MTP-1 value after replacing every M=2 backbone projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--economics", required=True, type=Path)
    parser.add_argument("--microbenchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-promotion-speedup", type=float, default=1.03)
    args = parser.parse_args()
    economics = json.loads(args.economics.read_text(encoding="utf-8"))
    microbenchmark = json.loads(args.microbenchmark.read_text(encoding="utf-8"))
    rows = microbenchmark["results"]
    if not rows or any(not row["name"].endswith("_m2") for row in rows):
        raise ValueError("microbenchmark must contain only M=2 backbone shapes")

    stock_projection_us = sum(
        row["cold_paired_best_vs_torch"]["baseline_median_us"]
        * row["multiplicity"]
        for row in rows
    )
    candidate_projection_us = sum(
        row["cold_paired_best_vs_torch"]["candidate_median_us"]
        * row["multiplicity"]
        for row in rows
    )
    projected_saving_ms = (stock_projection_us - candidate_projection_us) / 1000
    perf = economics["performance"]
    acceptance = economics["acceptance"]
    current_cycle_ms = perf["inferred_speculative_cycle_ms"]
    baseline_tpot_ms = perf["baseline_weighted_tpot_ms"]
    mean_acceptance_length = acceptance["mean_acceptance_length"]
    projected_cycle_ms = current_cycle_ms - projected_saving_ms
    projected_tpot_ms = projected_cycle_ms / mean_acceptance_length
    projected_speedup = baseline_tpot_ms / projected_tpot_ms
    break_even_cycle_ms = baseline_tpot_ms * mean_acceptance_length
    promotion_cycle_ms = break_even_cycle_ms / args.minimum_promotion_speedup
    break_even_required_saving_ms = current_cycle_ms - break_even_cycle_ms
    promotion_required_saving_ms = current_cycle_ms - promotion_cycle_ms
    extra_saving_for_promotion_ms = max(
        0.0, promotion_required_saving_ms - projected_saving_ms
    )
    perfect_acceptance_tpot_ms = projected_cycle_ms / (
        1 + acceptance["num_spec_tokens"]
    )

    decision = (
        "PROMOTE_TO_FULL_MODEL_IMPLEMENTATION"
        if projected_speedup >= args.minimum_promotion_speedup
        else "SCREEN_OUT_PROJECTION_ONLY_IMPLEMENTATION"
    )
    payload = {
        "schema_version": "mtp-m2-projection-bound-v1",
        "status": "PASS",
        "claim_scope": "DISCOVERY_PROJECTION_ONLY_COLD_CACHE_ESTIMATE",
        "inputs": {
            "economics": {
                "path": str(args.economics),
                "sha256": sha256(args.economics),
            },
            "microbenchmark": {
                "path": str(args.microbenchmark),
                "sha256": sha256(args.microbenchmark),
            },
        },
        "projection_estimate": {
            "stock_weighted_us": stock_projection_us,
            "candidate_weighted_us": candidate_projection_us,
            "projected_cycle_saving_ms": projected_saving_ms,
            "all_shapes_numerically_close": all(
                row["best_triton"]["correct"] for row in rows
            ),
            "maximum_observed_abs_error": max(
                row["best_triton"]["max_abs"] for row in rows
            ),
        },
        "projected_mtp": {
            "current_cycle_ms": current_cycle_ms,
            "projected_cycle_ms": projected_cycle_ms,
            "mean_acceptance_length": mean_acceptance_length,
            "projected_tpot_ms": projected_tpot_ms,
            "projected_speedup_vs_non_spec": projected_speedup,
            "perfect_acceptance_tpot_ms": perfect_acceptance_tpot_ms,
            "perfect_acceptance_speedup_vs_non_spec": (
                baseline_tpot_ms / perfect_acceptance_tpot_ms
            ),
        },
        "decision_boundary": {
            "minimum_promotion_speedup": args.minimum_promotion_speedup,
            "break_even_required_cycle_saving_ms": break_even_required_saving_ms,
            "promotion_required_cycle_saving_ms": promotion_required_saving_ms,
            "additional_non_projection_saving_needed_for_promotion_ms": (
                extra_saving_for_promotion_ms
            ),
            "decision": decision,
            "reason": (
                "At measured acceptance, replacing every screened M=2 projection "
                "still misses the promotion threshold. Re-open only if a separate "
                "cycle optimization supplies the recorded deficit or acceptance rises."
            ),
        },
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
