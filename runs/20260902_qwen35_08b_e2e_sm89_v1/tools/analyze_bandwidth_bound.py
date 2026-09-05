#!/usr/bin/env python3
"""Close a conservative batch-1 decode bandwidth bound from measured evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ACTIVE_GROUPS = (
    "embedding_and_lm_head",
    "full_attention",
    "gated_delta_net",
    "mlp",
    "normalization",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--memory-stream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gate-output", type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    stream = json.loads(args.memory_stream.read_text(encoding="utf-8"))

    groups = inventory["parameter_groups"]
    mandatory_weight_bytes = sum(int(groups[name]["storage_bytes"]) for name in ACTIVE_GROUPS)
    calibrated_bytes_per_second = float(stream["read_only_reduce"]["median_decimal_gb_per_second"]) * 1e9
    weight_stream_floor_ms = mandatory_weight_bytes / calibrated_bytes_per_second * 1000.0

    cases = []
    log_weighted_speedup = 0.0
    total_weight = 0.0
    for case in baseline["cases"]:
        observed_ms = float(case["median_tpot_ms"])
        effective_bytes_per_second = mandatory_weight_bytes / (observed_ms / 1000.0)
        maximum_speedup = observed_ms / weight_stream_floor_ms
        weight = float(case["weight"])
        total_weight += weight
        log_weighted_speedup += weight * math.log(maximum_speedup)
        cases.append({
            "case_id": case["case_id"],
            "weight": weight,
            "observed_tpot_ms": observed_ms,
            "weight_stream_floor_ms": weight_stream_floor_ms,
            "effective_weight_gb_per_second": effective_bytes_per_second / 1e9,
            "fraction_of_calibrated_read_service": effective_bytes_per_second / calibrated_bytes_per_second,
            "maximum_exact_bf16_speedup_ignoring_all_non_weight_work": maximum_speedup,
            "two_x_target_tpot_ms": observed_ms / 2.0,
            "two_x_feasible_under_bound": observed_ms / 2.0 >= weight_stream_floor_ms,
        })

    payload = {
        "schema_version": "qwen35-batch1-bandwidth-bound-v1",
        "status": "PASS",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "CONSERVATIVE_EXACT_BF16_BATCH1_DECODE_BOUND",
        "source_identities": {
            "inventory": str(args.inventory.resolve()),
            "baseline": str(args.baseline.resolve()),
            "memory_stream": str(args.memory_stream.resolve()),
        },
        "assumptions": [
            "every active dense BF16/F32 language weight is consumed at least once per generated token",
            "vision and dormant MTP weights are excluded",
            "the calibrated model-sized read-only stream is an optimistic service ceiling",
            "all activation, recurrent-state, KV-cache, synchronization, launch and arithmetic costs are ignored in the floor",
        ],
        "active_parameter_groups": list(ACTIVE_GROUPS),
        "mandatory_weight_bytes_per_token": mandatory_weight_bytes,
        "calibrated_read_decimal_gb_per_second": calibrated_bytes_per_second / 1e9,
        "weight_stream_floor_ms": weight_stream_floor_ms,
        "weighted_geometric_maximum_exact_bf16_speedup": math.exp(log_weighted_speedup / total_weight),
        "two_x_feasible_for_every_case": all(case["two_x_feasible_under_bound"] for case in cases),
        "cases": cases,
        "decision": "REJECT_EXACT_BF16_2X_AS_PHYSICALLY_INFEASIBLE" if not all(
            case["two_x_feasible_under_bound"] for case in cases
        ) else "EXACT_BF16_2X_NOT_REJECTED_BY_THIS_BOUND",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.gate_output:
        digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
        run_root = args.gate_output.resolve().parent.parent
        evidence_path = args.output.resolve().relative_to(run_root).as_posix()
        gate = {
            "schema_version": "optimization-feasibility-gate-v1",
            "status": "VALID",
            "decision": "TARGET_INFEASIBLE" if not payload["two_x_feasible_for_every_case"] else "TARGET_NOT_REJECTED",
            "target": {"metric": "batch1_tpot_ms", "speedup": 2.0},
            "bound": {
                "maximum_speedup": payload["weighted_geometric_maximum_exact_bf16_speedup"],
                "optimistic_latency_floor_ms": payload["weight_stream_floor_ms"],
                "claim_scope": payload["claim_scope"],
            },
            "evidence": [{"path": evidence_path, "sha256": digest}],
            "required_reframe_options": [
                "reduce mandatory weight bytes with a separately accepted quantization contract",
                "amortize weight reads across multiple useful tokens or requests",
                "change the model or hardware target",
                "replace the 2x goal with the measured exact-BF16 headroom",
            ],
        }
        args.gate_output.parent.mkdir(parents=True, exist_ok=True)
        args.gate_output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
