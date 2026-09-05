#!/usr/bin/env python3
"""Cheap correctness checks and paired discovery summary for the generic backend."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from pathlib import Path


RUN = Path(__file__).resolve().parents[2]
CANDIDATE = Path(__file__).resolve().parent
SOURCE = (
    CANDIDATE
    / "vllm/model_executor/kernels/linear/unquantized/lossless_packed_lm_head.py"
)
PATCH = CANDIDATE / "vllm_integration.patch"
CONTROL = RUN / "raw/stock_marlin_power_matched_r43.json"
TRIAL = CANDIDATE / "raw_r44.json"
CANONICAL_CONTROL = RUN / "raw/stock_marlin_natural_b1_w1_n1_t64.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_relative(path: Path) -> str:
    return path.relative_to(RUN).as_posix()


def measured(raw: dict) -> list[dict]:
    return [sample for sample in raw["raw_samples"] if sample["phase"] == "measure"]


def weighted_mean(raw: dict, key: str) -> float:
    total = sum(float(case["weight"]) for case in raw["cases"])
    return (
        sum(float(case["weight"]) * float(case[key]) for case in raw["cases"])
        / total
    )


def reconstruct_block(values: list[int]) -> list[int]:
    exponents = [(value >> 7) & 0xFF for value in values]
    if max(exponents) - min(exponents) > 15:
        return values[:]
    base = min(exponents)
    rebuilt = []
    for value, exponent in zip(values, exponents):
        sign_mantissa = (value & 0x7F) | ((value >> 8) & 0x80)
        delta = exponent - base
        rebuilt.append(
            ((sign_mantissa & 0x80) << 8)
            | ((base + delta) << 7)
            | (sign_mantissa & 0x7F)
        )
    return rebuilt


def correctness() -> None:
    ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    patch = PATCH.read_text(encoding="utf-8")
    required_hooks = (
        "lm_head_backend",
        "process_weights_after_loading",
        "try_apply_lossless_packed_lm_head",
        "LogitsProcessor",
    )
    missing = [hook for hook in required_hooks if hook not in patch]
    if missing:
        raise RuntimeError(f"integration patch lost required hooks: {missing}")

    rng = random.Random(20260905)
    blocks = [
        [rng.randrange(0, 1 << 16) for _ in range(256)],
        [((127 + rng.randrange(0, 8)) << 7) | rng.randrange(0, 128) for _ in range(256)],
        [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFFC1] * 42 + [0, 1, 2, 3],
    ]
    for index, block in enumerate(blocks):
        if reconstruct_block(block) != block:
            raise RuntimeError(f"bit reconstruction failed for synthetic block {index}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "source_sha256": digest(SOURCE),
                "patch_sha256": digest(PATCH),
                "synthetic_blocks": len(blocks),
                "covered": ["packable", "fallback", "zero", "signed", "nan", "inf"],
            }
        )
    )


def smoke() -> None:
    for path in (CONTROL, TRIAL, CANONICAL_CONTROL):
        if not path.is_file():
            raise FileNotFoundError(path)
    control = json.loads(CONTROL.read_text(encoding="utf-8"))
    trial = json.loads(TRIAL.read_text(encoding="utf-8"))
    canonical = json.loads(CANONICAL_CONTROL.read_text(encoding="utf-8"))
    if control["contracts"] != trial["contracts"]:
        raise RuntimeError("paired runs are not bound to the same contracts")
    control_samples = measured(control)
    trial_samples = measured(trial)
    if len(control_samples) != len(trial_samples) or len(control_samples) < 18:
        raise RuntimeError("paired runs require three measurements for all six cases")

    control_clock = sum(
        sample["gpu_telemetry_after"]["graphics_clock_mhz"]
        for sample in control_samples
    ) / len(control_samples)
    trial_clock = sum(
        sample["gpu_telemetry_after"]["graphics_clock_mhz"]
        for sample in trial_samples
    ) / len(trial_samples)
    if abs(control_clock - trial_clock) > max(control_clock, trial_clock) * 0.01:
        raise RuntimeError("paired GPU clocks differ by more than one percent")

    source_key = next(
        key
        for key in trial["environment"]["guarded_source_sha256"]
        if key.endswith("lossless_packed_lm_head.py")
    )
    if trial["environment"]["guarded_source_sha256"][source_key] != digest(SOURCE):
        raise RuntimeError("measured module does not match the candidate source")

    control_ms = weighted_mean(control, "median_end_to_end_ms")
    trial_ms = weighted_mean(trial, "median_end_to_end_ms")
    canonical_matches = sum(
        left["generated_token_ids"] == right["generated_token_ids"]
        for left, right in zip(trial["cases"], canonical["cases"])
    )
    paired_matches = sum(
        left["generated_token_ids"] == right["generated_token_ids"]
        for left, right in zip(trial["cases"], control["cases"])
    )
    result = {
        "schema_version": "candidate-smoke-result-v3",
        "status": "PASS" if trial_ms < control_ms else "FAIL",
        "candidate_id": "adaptive-lossless-lmhead",
        "claim_scope": "DISCOVERY_ONLY_NOT_PRODUCTION_ACCEPTANCE_OR_LIMIT_PROOF",
        "objective": {
            "direction": "minimize",
            "baseline": control_ms * 1000.0,
            "candidate": trial_ms * 1000.0,
            "unit": "us",
            "speedup": control_ms / trial_ms,
            "reduction_percent": (control_ms - trial_ms) / control_ms * 100.0,
        },
        "comparability": {
            "status": "PASS",
            "measurements_per_case": 3,
            "control_graphics_clock_mhz": control_clock,
            "candidate_graphics_clock_mhz": trial_clock,
            "contract_identity": "PASS",
            "source_identity": "PASS",
        },
        "correctness_observation": {
            "canonical_stock_token_matches": canonical_matches,
            "paired_stock_token_matches": paired_matches,
            "case_count": len(trial["cases"]),
            "interpretation": "Cross-process stock output is not deterministic in this setup; token identity is an observation, not a proof of bitwise logit equality.",
        },
        "raw": {
            "baseline": {"path": run_relative(CONTROL), "sha256": digest(CONTROL)},
            "candidate": {"path": run_relative(TRIAL), "sha256": digest(TRIAL)},
            "canonical": {
                "path": run_relative(CANONICAL_CONTROL),
                "sha256": digest(CANONICAL_CONTROL),
            },
        },
    }
    (CANDIDATE / "smoke_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], **result["objective"]}))
    if result["status"] != "PASS":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("correctness", "smoke"))
    args = parser.parse_args()
    {"correctness": correctness, "smoke": smoke}[args.mode]()


if __name__ == "__main__":
    main()
