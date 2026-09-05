#!/usr/bin/env python3
"""Fail fast on deterministic-but-degenerate text generation traces.

This is a cheap sanity gate, not a replacement for reference agreement,
perplexity, or downstream task evaluation.  It prevents a performance search
from promoting obvious short token cycles as valid winners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def assess_tokens(token_ids: list[int]) -> dict:
    if not token_ids or not all(isinstance(value, int) for value in token_ids):
        return {
            "status": "FAIL",
            "reason": "generated_token_ids must be a non-empty integer list",
        }
    distinct = len(set(token_ids))
    minimum = min(8, max(len(token_ids) // 4, 2))
    return {
        "status": "PASS" if distinct >= minimum else "FAIL",
        "generated_tokens": len(token_ids),
        "distinct_token_count": distinct,
        "distinct_token_fraction": distinct / len(token_ids),
        "minimum_distinct_tokens": minimum,
        "reason": (
            None
            if distinct >= minimum
            else "degenerate low-diversity token cycle"
        ),
    }


def extract_cases(payload: dict) -> list[tuple[str, list[int]]]:
    if isinstance(payload.get("cases"), list):
        rows = []
        for index, case in enumerate(payload["cases"]):
            case_id = str(case.get("case_id", index))
            rows.append((case_id, case.get("generated_token_ids")))
        return rows
    output = payload.get("output")
    if isinstance(output, dict) and "generated_token_ids" in output:
        return [("output", output["generated_token_ids"])]
    if "generated_token_ids" in payload:
        return [("output", payload["generated_token_ids"])]
    raise ValueError("input contains no cases/output generated_token_ids")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cases = [
        {"case_id": case_id, **assess_tokens(token_ids)}
        for case_id, token_ids in extract_cases(payload)
    ]
    result = {
        "schema_version": "generation-quality-sanity-v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in cases) else "FAIL",
        "claim_scope": "CHEAP_DEGENERATION_GATE_NOT_SEMANTIC_QUALITY_QUALIFICATION",
        "source": {
            "path": str(args.input),
            "sha256": sha256(args.input),
        },
        "cases": cases,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
