#!/usr/bin/env python3
"""Block a RUNNING experiment on immutable external evidence without losing partial results."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(run: Path, path: Path) -> dict:
    resolved = path.resolve()
    if run not in resolved.parents or not resolved.is_file():
        raise ValueError(f"evidence must be an existing file inside the run: {resolved}")
    return {"path": resolved.relative_to(run).as_posix(), "sha256": sha(resolved)}


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--blocking-evidence", type=Path, action="append", required=True)
    parser.add_argument("--partial-result", type=Path, action="append", default=[])
    args = parser.parse_args()
    run = args.run.resolve()
    queue_path = run / "models/experiment_queue.json"
    queue = json.loads(queue_path.read_text())
    request = next((item for item in queue.get("requests", []) if item.get("request_id") == args.request_id), None)
    if request is None or request.get("status") != "RUNNING":
        raise ValueError("only a RUNNING experiment can be externally blocked")
    blockers = [evidence(run, path) for path in args.blocking_evidence]
    partial = [evidence(run, path) for path in args.partial_result]
    for record in blockers:
        data = json.loads((run / record["path"]).read_text())
        if not str(data.get("status", "")).startswith("BLOCKED"):
            raise ValueError(f"blocking evidence does not declare BLOCKED status: {record['path']}")
    for record in partial:
        data = json.loads((run / record["path"]).read_text())
        if data.get("status") != "PASS":
            raise ValueError(f"partial result is not PASS: {record['path']}")
    request["status"] = "BLOCKED"
    request["blocking_evidence"] = blockers
    request["partial_results"] = partial
    request["blocking_reason"] = args.reason
    request["blocked_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(queue_path, queue)
    print(json.dumps({
        "status": "BLOCKED", "request_id": args.request_id,
        "blocking_evidence": blockers, "partial_results": partial,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
