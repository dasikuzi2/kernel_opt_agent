#!/usr/bin/env python3
"""Build a hash-bound cross-process summary for the SM89 proxy experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def _read(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise ValueError(f"non-PASS input: {path}")
    return value


def _identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _series(rows, path):
    result = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        result.append(float(value))
    return result


def _summary(values):
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "samples": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", nargs=3, required=True, type=Path)
    parser.add_argument("--fusion", nargs=3, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cost = [_read(path) for path in args.cost]
    fusion = [_read(path) for path in args.fusion]
    if len({row["source_identities"]["experiment_source"]["sha256"] for row in cost}) != 1:
        raise ValueError("cost repetitions use different experiment sources")
    if len({row["source_identities"]["experiment_source"]["sha256"] for row in fusion}) != 1:
        raise ValueError("fusion repetitions use different experiment sources")

    cost_metrics = {
        "preallocated_effective_mean_us": ("aggregate", "preallocated_effective_mean_us"),
        "allocated_effective_mean_us": ("aggregate", "allocated_effective_mean_us"),
        "cupti_active_mean_us": ("aggregate", "cupti_active_mean_us"),
        "allocation_path_delta_mean_us": ("aggregate", "allocation_path_delta_mean_us"),
        "active_fraction_of_effective_mean": ("aggregate", "active_fraction_of_effective_mean"),
    }
    fusion_metrics = {
        "mean_effective_speedup": ("aggregate", "mean_effective_speedup"),
        "mean_cupti_active_speedup": ("aggregate", "mean_cupti_active_speedup"),
        "materialized_effective_mean_us": ("aggregate", "materialized_effective_mean_us"),
        "fused_effective_mean_us": ("aggregate", "fused_effective_mean_us"),
        "materialized_active_mean_us": ("aggregate", "materialized_active_mean_us"),
        "fused_active_mean_us": ("aggregate", "fused_active_mean_us"),
    }
    result = {
        "schema_version": "sm89-proxy-repetition-summary-v1",
        "status": "PASS",
        "claim_scope": "DISCOVERY_ONLY_CROSS_PROCESS_REPLICATION_NOT_SM120_QUALIFICATION",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repetitions": 3,
        "input_identities": {
            "cost": [_identity(path) for path in args.cost],
            "fusion": [_identity(path) for path in args.fusion],
        },
        "cost_stack": {name: _summary(_series(cost, path)) for name, path in cost_metrics.items()},
        "producer_consumer_fusion": {name: _summary(_series(fusion, path)) for name, path in fusion_metrics.items()},
        "decision": {
            "status": "PROMOTE_DIRECT_CONSUMPTION_MECHANISM",
            "reason": "all repetitions pass correctness and exceed 1.05x in both effective-timeline and CUPTI-active mean speedup",
            "next_environment": "SM120 target GPU with the production recurrent consumer",
        },
    }
    if min(result["producer_consumer_fusion"]["mean_effective_speedup"]["samples"]) <= 1.05:
        raise ValueError("effective speedup replication gate failed")
    if min(result["producer_consumer_fusion"]["mean_cupti_active_speedup"]["samples"]) <= 1.05:
        raise ValueError("active speedup replication gate failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "cost_stack": result["cost_stack"], "producer_consumer_fusion": result["producer_consumer_fusion"], "decision": result["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
