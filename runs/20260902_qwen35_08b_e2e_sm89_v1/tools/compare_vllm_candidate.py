#!/usr/bin/env python3
"""Compare two vLLM traces without hiding output or power-state differences."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def weighted(cases: list[dict], field: str) -> float:
    return sum(float(case["weight"]) * float(case[field]) for case in cases)


def telemetry(trace: dict, field: str) -> float | None:
    values = [
        sample["gpu_telemetry_after"][field]
        for sample in trace["raw_samples"]
        if sample["phase"] == "measure"
        and sample.get("gpu_telemetry_after") is not None
    ]
    return float(statistics.median(values)) if values else None


def index_cases(trace: dict) -> dict[str, dict]:
    return {case["case_id"]: case for case in trace["cases"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    stock = json.loads(args.stock.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    stock_cases = index_cases(stock)
    candidate_cases = index_cases(candidate)
    if stock_cases.keys() != candidate_cases.keys():
        raise ValueError("stock and candidate case IDs differ")

    per_case = []
    for case_id, stock_case in stock_cases.items():
        candidate_case = candidate_cases[case_id]
        per_case.append(
            {
                "case_id": case_id,
                "token_exact": stock_case["generated_token_ids"]
                == candidate_case["generated_token_ids"],
                "e2e_speedup": stock_case["median_end_to_end_ms"]
                / candidate_case["median_end_to_end_ms"],
                "tpot_speedup": stock_case["median_tpot_ms"]
                / candidate_case["median_tpot_ms"],
            }
        )

    stock_e2e = weighted(stock["cases"], "median_end_to_end_ms")
    candidate_e2e = weighted(candidate["cases"], "median_end_to_end_ms")
    stock_tpot = weighted(stock["cases"], "median_tpot_ms")
    candidate_tpot = weighted(candidate["cases"], "median_tpot_ms")
    exact = all(row["token_exact"] for row in per_case)
    result = {
        "status": "PASS" if exact else "FAIL",
        "stock": {
            "trace": str(args.stock),
            "weighted_e2e_ms": stock_e2e,
            "weighted_tpot_ms": stock_tpot,
            "median_graphics_clock_mhz": telemetry(stock, "graphics_clock_mhz"),
            "median_power_w": telemetry(stock, "power_w"),
        },
        "candidate": {
            "trace": str(args.candidate),
            "weighted_e2e_ms": candidate_e2e,
            "weighted_tpot_ms": candidate_tpot,
            "median_graphics_clock_mhz": telemetry(
                candidate, "graphics_clock_mhz"
            ),
            "median_power_w": telemetry(candidate, "power_w"),
        },
        "speedup": {
            "e2e": stock_e2e / candidate_e2e,
            "tpot": stock_tpot / candidate_tpot,
        },
        "exact_generated_token_cases": f"{sum(row['token_exact'] for row in per_case)}/{len(per_case)}",
        "per_case": per_case,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not exact:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
