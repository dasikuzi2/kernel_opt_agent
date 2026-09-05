#!/usr/bin/env python3
"""Decide whether speculative decoding can amortize its measured cycle cost."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted(cases: list[dict], field: str) -> float:
    return sum(float(case["weight"]) * float(case[field]) for case in cases)


def analyze(
    baseline_path: Path,
    speculative_path: Path,
    minimum_promotion_speedup: float,
) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    speculative = json.loads(speculative_path.read_text(encoding="utf-8"))
    baseline_cases = {case["case_id"]: case for case in baseline["cases"]}
    speculative_cases = {case["case_id"]: case for case in speculative["cases"]}
    if baseline_cases.keys() != speculative_cases.keys():
        raise ValueError("baseline and speculative traces cover different cases")

    exact_cases = 0
    for case_id, base_case in baseline_cases.items():
        spec_case = speculative_cases[case_id]
        if base_case.get("prompt_token_ids_sha256") != spec_case.get(
            "prompt_token_ids_sha256"
        ):
            raise ValueError(f"prompt identity differs for {case_id}")
        if base_case["generated_token_ids"] == spec_case["generated_token_ids"]:
            exact_cases += 1

    metrics = [case.get("spec_decode_metrics") for case in speculative_cases.values()]
    if any(item is None for item in metrics):
        raise ValueError(
            "speculative trace lacks acceptance metrics; rerun with per-request "
            "spec decode metrics enabled"
        )
    non_null_metrics = [item for item in metrics if item is not None]
    num_steps = sum(int(item["num_spec_steps"]) for item in non_null_metrics)
    num_accepted = sum(
        int(item["num_accepted_draft_tokens"]) for item in non_null_metrics
    )
    num_drafted = sum(int(item["num_draft_tokens"]) for item in non_null_metrics)
    if num_steps <= 0 or num_drafted <= 0:
        raise ValueError("speculative trace contains no measured draft steps")

    num_spec_tokens_values = {
        int(sample["spec_decode_metrics"]["num_spec_tokens"])
        for sample in speculative.get("raw_samples", [])
        if sample.get("phase") == "measure"
        and sample.get("spec_decode_metrics") is not None
    }
    if not num_spec_tokens_values:
        configured = int(speculative["controls"].get("speculative_tokens", 0))
        num_spec_tokens_values = {configured}
    if len(num_spec_tokens_values) != 1:
        raise ValueError("trace mixes different speculative token counts")
    num_spec_tokens = num_spec_tokens_values.pop()
    if num_spec_tokens <= 0:
        raise ValueError("speculative token count must be positive")

    baseline_cases_list = list(baseline_cases.values())
    speculative_cases_list = list(speculative_cases.values())
    baseline_tpot = weighted(baseline_cases_list, "median_tpot_ms")
    speculative_tpot = weighted(speculative_cases_list, "median_tpot_ms")
    baseline_e2e = weighted(baseline_cases_list, "median_end_to_end_ms")
    speculative_e2e = weighted(speculative_cases_list, "median_end_to_end_ms")
    mean_acceptance_length = 1.0 + num_accepted / num_steps
    draft_acceptance_rate = num_accepted / num_drafted

    # TPOT is measured per emitted interval. Multiplying it by the observed
    # output tokens per verify step estimates one proposer+verify cycle. This
    # deliberately assumes that cycle cost stays constant as acceptance rises;
    # it is a screening upper bound, not a production performance certificate.
    inferred_cycle_ms = speculative_tpot * mean_acceptance_length
    perfect_acceptance_length = 1.0 + num_spec_tokens
    perfect_acceptance_tpot_floor = inferred_cycle_ms / perfect_acceptance_length
    perfect_acceptance_speedup_ceiling = baseline_tpot / perfect_acceptance_tpot_floor
    required_mean_acceptance_length = inferred_cycle_ms / baseline_tpot
    required_draft_acceptance_rate = (
        required_mean_acceptance_length - 1.0
    ) / num_spec_tokens
    measured_speedup = baseline_tpot / speculative_tpot
    e2e_speedup = baseline_e2e / speculative_e2e

    reasons = []
    if exact_cases != len(baseline_cases):
        reasons.append("generated token IDs differ from the frozen baseline")
    if measured_speedup < minimum_promotion_speedup:
        reasons.append("measured TPOT does not clear the promotion threshold")
    if perfect_acceptance_speedup_ceiling < minimum_promotion_speedup:
        reasons.append(
            "even the constant-cycle perfect-acceptance ceiling misses the threshold"
        )
    decision = "PROMOTE_TO_QUALIFICATION" if not reasons else "SCREEN_OUT"

    return {
        "schema_version": "speculation-economics-v1",
        "status": "PASS",
        "claim_scope": "DISCOVERY_SCREENING_ONLY_CONSTANT_CYCLE_APPROXIMATION",
        "inputs": {
            "baseline": {
                "path": str(baseline_path),
                "sha256": sha256(baseline_path),
            },
            "speculative": {
                "path": str(speculative_path),
                "sha256": sha256(speculative_path),
            },
        },
        "correctness": {
            "exact_cases": exact_cases,
            "case_count": len(baseline_cases),
            "all_cases_exact": exact_cases == len(baseline_cases),
        },
        "acceptance": {
            "num_spec_tokens": num_spec_tokens,
            "num_spec_steps": num_steps,
            "num_draft_tokens": num_drafted,
            "num_accepted_draft_tokens": num_accepted,
            "draft_acceptance_rate": draft_acceptance_rate,
            "mean_acceptance_length": mean_acceptance_length,
        },
        "performance": {
            "baseline_weighted_tpot_ms": baseline_tpot,
            "speculative_weighted_tpot_ms": speculative_tpot,
            "measured_tpot_speedup": measured_speedup,
            "baseline_weighted_e2e_ms": baseline_e2e,
            "speculative_weighted_e2e_ms": speculative_e2e,
            "measured_e2e_speedup": e2e_speedup,
            "inferred_speculative_cycle_ms": inferred_cycle_ms,
            "required_mean_acceptance_length_to_break_even": (
                required_mean_acceptance_length
            ),
            "required_draft_acceptance_rate_to_break_even": (
                required_draft_acceptance_rate
            ),
            "perfect_acceptance_tpot_floor_ms": perfect_acceptance_tpot_floor,
            "perfect_acceptance_speedup_ceiling": perfect_acceptance_speedup_ceiling,
        },
        "policy": {
            "minimum_promotion_speedup": minimum_promotion_speedup,
            "decision": decision,
            "reasons": reasons,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--speculative", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-promotion-speedup", type=float, default=1.03)
    args = parser.parse_args()
    if args.minimum_promotion_speedup <= 1.0:
        raise ValueError("minimum promotion speedup must be greater than 1")
    result = analyze(
        args.baseline.resolve(),
        args.speculative.resolve(),
        args.minimum_promotion_speedup,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
