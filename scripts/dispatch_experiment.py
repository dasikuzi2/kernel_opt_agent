#!/usr/bin/env python3
"""Dispatch only a fully materialized and hash-verified experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_utils import read_object, sha256
from experiment_utils import validate_materialized_experiment
from supervision_utils import validate_supervisor_approval


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    experiment_path = run / "experiments" / args.request_id / "experiment.json"
    errors = validate_materialized_experiment(experiment_path, run)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2, sort_keys=True))
        return 1
    queue_path = run / "models/experiment_queue.json"
    queue = read_object(queue_path)
    request = next((item for item in queue.get("requests", []) if item.get("request_id") == args.request_id), None)
    if request is None:
        raise ValueError(f"request not found: {args.request_id}")
    if request.get("status") != "PLANNED":
        raise ValueError("only a PLANNED request can be dispatched")
    if queue.get("schema_version") not in {"experiment-request-queue-v2", "experiment-request-queue-v3"}:
        raise ValueError("legacy resource-centric requests cannot be dispatched")
    if read_object(experiment_path).get("request_id") != args.request_id:
        raise ValueError("materialized experiment request_id mismatch")
    approval_path = experiment_path.parent / "supervisor_approval.json"
    approval_errors = validate_supervisor_approval(approval_path, run, request, experiment_path)
    if approval_errors:
        print(json.dumps({"status": "FAIL", "errors": approval_errors}, indent=2, sort_keys=True))
        return 1
    request["status"] = "DISPATCHED"
    request["materialized_experiment"] = {"path": str(experiment_path), "sha256": sha256(experiment_path), "status": "MATERIALIZED"}
    request["supervisor_approval"] = {"path": approval_path.relative_to(run).as_posix(), "sha256": sha256(approval_path), "status": "CONSUMED_BY_DISPATCH"}
    atomic_json(queue_path, queue)
    print(json.dumps({"status": "DISPATCHED", "request_id": args.request_id, "experiment": str(experiment_path), "approval": str(approval_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
