#!/usr/bin/env python3
"""Archive an attempt and force independent supervisor review or global replanning."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--review-evidence", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    evidence = args.review_evidence.resolve()
    if run not in evidence.parents or not evidence.is_file():
        raise ValueError("review evidence must be an existing file inside the run")
    queue_path = run / "models/experiment_queue.json"
    queue = json.loads(queue_path.read_text())
    request = next((item for item in queue.get("requests", []) if item.get("request_id") == args.request_id), None)
    if request is None or request.get("status") not in {"RUNNING", "BLOCKED"}:
        raise ValueError("only a RUNNING or technically BLOCKED completed execution can be revised")
    experiment_dir = run / "experiments" / args.request_id
    receipt_path = experiment_dir / "execution_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt_status = receipt.get("status")
    if receipt_status not in {"PASS", "FAIL"} or receipt.get("request_id") != args.request_id:
        raise ValueError("revision requires a completed PASS or FAIL execution receipt")
    if request.get("status") == "RUNNING" and receipt_status != "PASS":
        raise ValueError("RUNNING requests require a technically PASS receipt before causal revision")
    if request.get("status") == "BLOCKED" and receipt_status != "FAIL":
        raise ValueError("BLOCKED requests require a technically FAIL receipt before technical revision")
    disposition = "REJECTED_FOR_CAUSAL_VALIDITY" if receipt_status == "PASS" else "TECHNICAL_FAILURE_REVISED"
    attempt_root = experiment_dir / "attempts"
    attempt_root.mkdir(parents=True, exist_ok=True)
    attempt_number = 1
    while (attempt_root / f"attempt-{attempt_number:02d}").exists():
        attempt_number += 1
    archive = attempt_root / f"attempt-{attempt_number:02d}"
    archive.mkdir()
    names = [
        "experiment.json", "execution_receipt.json", "result.json", "reproduction.json",
        "correctness.json", "warmup.json", "build.json", "catalog_query_receipt.json",
        "supervisor_approval.json", "execution_logs", "raw", "static",
    ]
    copied = []
    for name in names:
        source = experiment_dir / name
        if not source.exists():
            continue
        target = archive / name
        if source.is_dir():
            shutil.copytree(source, target)
            for path in target.rglob("*"):
                if path.is_file():
                    copied.append(path)
        else:
            shutil.copy2(source, target)
            copied.append(target)
    review_target = archive / "validity_review.json"
    shutil.copy2(evidence, review_target)
    copied.append(review_target)
    manifest = {
        "schema_version": "experiment-attempt-archive-v1",
        "status": disposition,
        "request_id": args.request_id,
        "attempt": attempt_number,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "reason": args.reason,
        "files": [
            {"path": path.relative_to(run).as_posix(), "sha256": sha(path)}
            for path in sorted(copied)
        ],
    }
    manifest_path = archive / "attempt_manifest.json"
    atomic_json(manifest_path, manifest)
    history = request.setdefault("attempt_history", [])
    history.append({
        "attempt": attempt_number,
        "disposition": disposition,
        "reason": args.reason,
        "archive_manifest": {"path": manifest_path.relative_to(run).as_posix(), "sha256": sha(manifest_path)},
    })
    next_status = "HALT_AND_REPLAN" if receipt_status == "PASS" else "AWAITING_SUPERVISOR_REVIEW"
    request["status"] = next_status
    request.pop("execution_receipt", None)
    request.pop("blocking_evidence", None)
    request.pop("supervisor_approval", None)
    request["materialized_experiment"] = {
        "path": str(experiment_dir / "experiment.json"),
        "sha256": sha(experiment_dir / "experiment.json"),
        "status": next_status,
    }
    experiment_path = experiment_dir / "experiment.json"
    experiment = json.loads(experiment_path.read_text())
    experiment["status"] = next_status
    experiment.setdefault("revision_history", []).append({
        "attempt": attempt_number,
        "reason": args.reason,
        "review_evidence": {"path": review_target.relative_to(run).as_posix(), "sha256": sha(review_target)},
    })
    atomic_json(experiment_path, experiment)
    request["materialized_experiment"]["sha256"] = sha(experiment_path)
    atomic_json(queue_path, queue)
    print(json.dumps({
        "status": next_status, "request_id": args.request_id,
        "attempt_archive": str(archive), "manifest_sha256": sha(manifest_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
