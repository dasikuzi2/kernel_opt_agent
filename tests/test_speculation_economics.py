#!/usr/bin/env python3
"""Dependency-free checks for speculation acceptance/economics screening."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_speculation_economics",
    ROOT / "scripts" / "analyze_speculation_economics.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def trace(tpot: float, tokens: list[int], metrics: dict | None = None) -> dict:
    case = {
        "case_id": "case-a",
        "weight": 1.0,
        "prompt_token_ids_sha256": "prompt-a",
        "generated_token_ids": tokens,
        "median_tpot_ms": tpot,
        "median_end_to_end_ms": tpot * 10,
        "spec_decode_metrics": metrics,
    }
    return {
        "controls": {"speculative_tokens": 1 if metrics else 0},
        "cases": [case],
        "raw_samples": [],
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        baseline_path = root / "baseline.json"
        speculative_path = root / "speculative.json"
        baseline_path.write_text(json.dumps(trace(8.0, [1, 2, 3])))
        speculative_path.write_text(
            json.dumps(
                trace(
                    10.0,
                    [1, 2, 3],
                    {
                        "num_spec_steps": 10,
                        "num_accepted_draft_tokens": 7,
                        "num_draft_tokens": 10,
                    },
                )
            )
        )
        result = MODULE.analyze(baseline_path, speculative_path, 1.03)
        assert result["acceptance"]["draft_acceptance_rate"] == 0.7
        assert result["policy"]["decision"] == "SCREEN_OUT"
        assert result["correctness"]["all_cases_exact"]
        assert result["performance"]["perfect_acceptance_speedup_ceiling"] < 1.03

        missing_metrics = trace(7.0, [1, 2, 3])
        speculative_path.write_text(json.dumps(missing_metrics))
        try:
            MODULE.analyze(baseline_path, speculative_path, 1.03)
        except ValueError as exc:
            assert "acceptance metrics" in str(exc)
        else:
            raise AssertionError("missing speculation acceptance metrics passed")

    print("speculation economics test: PASS")


if __name__ == "__main__":
    main()
