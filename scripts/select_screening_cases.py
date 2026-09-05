#!/usr/bin/env python3
"""Select a small, deterministic architecture-screening set from frozen cases.

The selector covers both sides of a routing boundary and the short/median/long
sequence regimes.  It is deliberately not a replacement for full production
qualification: its output carries DISCOVERY_ONLY claim scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pick_quantiles(cases: list[dict], roles: tuple[str, ...]) -> list[tuple[str, dict]]:
    ordered = sorted(cases, key=lambda item: (
        item["parameters"]["total_seq_len"],
        item["parameters"]["num_seqs"],
        item["id"],
    ))
    if not ordered:
        return []
    indices = (0, len(ordered) // 2, len(ordered) - 1)
    return [(role, ordered[index]) for role, index in zip(roles, indices)]


def select_cases(workload: dict, *, heads: int, sm_count: int,
                 threshold_numerator: int, threshold_denominator: int) -> dict:
    cases = workload.get("cases", [])
    if not cases:
        raise ValueError("workload has no cases")

    def cp_route(case: dict) -> bool:
        parallel_work = case["parameters"]["num_seqs"] * heads
        return parallel_work * threshold_denominator < sm_count * threshold_numerator

    cp_cases = [case for case in cases if cp_route(case)]
    non_cp_cases = [case for case in cases if not cp_route(case)]
    picks = _pick_quantiles(cp_cases, ("CP_SHORTEST", "CP_MEDIAN", "CP_LONGEST"))
    picks += _pick_quantiles(
        non_cp_cases, ("NON_CP_SHORTEST", "NON_CP_MEDIAN", "NON_CP_LONGEST")
    )

    selected = []
    seen = set()
    for role, case in picks:
        if case["id"] in seen:
            continue
        seen.add(case["id"])
        selected.append({
            "id": case["id"],
            "role": role,
            "total_seq_len": case["parameters"]["total_seq_len"],
            "num_seqs": case["parameters"]["num_seqs"],
            "expected_route": "CP" if cp_route(case) else "NON_CP",
        })

    return {
        "schema_version": "architecture-screening-set-v1",
        "claim_scope": "DISCOVERY_ONLY_NOT_PRODUCTION_ACCEPTANCE",
        "selection_policy": {
            "method": "shortest/median/longest on both sides of the routing boundary",
            "heads": heads,
            "sm_count": sm_count,
            "threshold_numerator": threshold_numerator,
            "threshold_denominator": threshold_denominator,
            "route_equation": "num_seqs * heads * denominator < sm_count * numerator",
        },
        "population": {
            "total": len(cases),
            "cp": len(cp_cases),
            "non_cp": len(non_cp_cases),
        },
        "cases": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heads", required=True, type=int)
    parser.add_argument("--sm-count", required=True, type=int)
    parser.add_argument("--threshold-numerator", type=int, default=1)
    parser.add_argument("--threshold-denominator", type=int, default=3)
    args = parser.parse_args()

    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    result = select_cases(
        workload,
        heads=args.heads,
        sm_count=args.sm_count,
        threshold_numerator=args.threshold_numerator,
        threshold_denominator=args.threshold_denominator,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
