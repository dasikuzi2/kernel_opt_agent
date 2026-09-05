#!/usr/bin/env python3
"""Compare paired GSM8K outputs from two vLLM execution paths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mcnemar_exact_p_value(control_only: int, candidate_only: int) -> float:
    discordant = control_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(control_only, candidate_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    control = json.loads(args.control.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if control["dataset"]["sha256"] != candidate["dataset"]["sha256"]:
        raise ValueError("dataset hashes differ")
    if control["dataset"]["selected_indices"] != candidate["dataset"]["selected_indices"]:
        raise ValueError("selected index sets differ")

    control_rows = {item["dataset_index"]: item for item in control["results"]}
    candidate_rows = {item["dataset_index"]: item for item in candidate["results"]}
    if control_rows.keys() != candidate_rows.keys():
        raise ValueError("paired case sets differ")
    pairs = [(control_rows[index], candidate_rows[index]) for index in sorted(control_rows)]
    answer_agreement = sum(a["prediction"] == b["prediction"] for a, b in pairs)
    token_exact = sum(
        a["generated_token_ids"] == b["generated_token_ids"] for a, b in pairs
    )
    both_correct = sum(a["correct"] and b["correct"] for a, b in pairs)
    control_only = sum(a["correct"] and not b["correct"] for a, b in pairs)
    candidate_only = sum(not a["correct"] and b["correct"] for a, b in pairs)
    both_wrong = len(pairs) - both_correct - control_only - candidate_only
    payload = {
        "schema_version": "paired-gsm8k-quality-comparison-v1",
        "status": "PASS",
        "sample_count": len(pairs),
        "control_accuracy": control["accuracy"],
        "candidate_accuracy": candidate["accuracy"],
        "candidate_minus_control_accuracy": candidate["accuracy"] - control["accuracy"],
        "answer_agreement_count": answer_agreement,
        "answer_agreement_rate": answer_agreement / len(pairs),
        "token_exact_count": token_exact,
        "token_exact_rate": token_exact / len(pairs),
        "paired_outcomes": {
            "both_correct": both_correct,
            "control_only_correct": control_only,
            "candidate_only_correct": candidate_only,
            "both_wrong": both_wrong,
        },
        "mcnemar_exact_two_sided_p": mcnemar_exact_p_value(
            control_only, candidate_only
        ),
        "qualification": (
            "This is a bounded task-quality screen, not proof of global model-quality "
            "equivalence. Promotion still requires broader tasks and sampling modes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
