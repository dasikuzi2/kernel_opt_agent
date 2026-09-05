#!/usr/bin/env python3
"""Execute a dispatched argv contract and preserve a hash-bound execution receipt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from evidence_utils import path_is_within, read_object, sha256
from experiment_utils import validate_materialized_experiment


PHASES = ("clean_build", "static_audit", "correctness", "warmup", "measure", "analyze")


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def artifact_path(run: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (run / path).resolve()
    if not path_is_within(resolved, run):
        raise ValueError(f"experiment artifact escapes run directory: {resolved}")
    return resolved


def command_input_identities(argv: list[str], cwd: Path) -> list[dict]:
    identities = []
    for value in argv[:2]:
        path = Path(value)
        candidate = path if path.is_absolute() else cwd / path
        if candidate.is_file():
            identities.append({"path": str(candidate.resolve()), "sha256": sha256(candidate)})
    return identities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    queue_path = run / "models/experiment_queue.json"
    queue = read_object(queue_path)
    request = next((item for item in queue.get("requests", []) if item.get("request_id") == args.request_id), None)
    if request is None or request.get("status") != "DISPATCHED":
        raise ValueError("only a DISPATCHED request can execute")
    experiment_ref = request.get("materialized_experiment", {})
    experiment_path = Path(experiment_ref.get("path", ""))
    if not experiment_path.is_absolute():
        experiment_path = run / experiment_path
    if experiment_ref.get("sha256") != sha256(experiment_path):
        raise ValueError("materialized experiment changed after dispatch")
    errors = validate_materialized_experiment(experiment_path, run)
    if errors:
        raise ValueError("invalid materialized experiment: " + "; ".join(errors))
    experiment = read_object(experiment_path)
    hardware_path = run / "hardware.json"
    workload_path = run / "workload.json"
    receipt_path = experiment_path.parent / "execution_receipt.json"
    logs_dir = experiment_path.parent / "execution_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "experiment-execution-receipt-v1",
        "status": "RUNNING",
        "request_id": args.request_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "experiment_identity": {"path": str(experiment_path), "sha256": sha256(experiment_path)},
        "hardware_identity": {"path": str(hardware_path), "sha256": sha256(hardware_path)},
        "workload_identity": {"path": str(workload_path), "sha256": sha256(workload_path)},
        "target": read_object(hardware_path).get("target", {}),
        "environment": {
            "cwd": str(run),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "commands": [],
    }
    pre_artifacts = {}
    for name in ("raw_samples", "result", "static_audit"):
        path = artifact_path(run, str(experiment.get("artifacts", {}).get(name, "")))
        pre_artifacts[name] = {"existed": path.is_file(), "sha256": sha256(path) if path.is_file() else None}
    receipt["pre_execution_artifacts"] = pre_artifacts
    request["status"] = "RUNNING"
    atomic_json(queue_path, queue)
    failure = None
    for phase in PHASES:
        for index, argv in enumerate(experiment.get("commands", {}).get(phase, [])):
            started = datetime.now(timezone.utc).isoformat()
            completed = subprocess.run(argv, cwd=run, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_path = logs_dir / f"{phase}-{index}.stdout.txt"
            stderr_path = logs_dir / f"{phase}-{index}.stderr.txt"
            stdout_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)
            record = {
                "phase": phase,
                "index": index,
                "argv": argv,
                "input_identities": command_input_identities(argv, run),
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": completed.returncode,
                "stdout": {"path": stdout_path.relative_to(run).as_posix(), "sha256": sha256(stdout_path)},
                "stderr": {"path": stderr_path.relative_to(run).as_posix(), "sha256": sha256(stderr_path)},
            }
            receipt["commands"].append(record)
            if completed.returncode:
                failure = f"{phase}[{index}] exited {completed.returncode}"
                break
        if failure:
            break
    missing_artifacts = []
    if not failure:
        for name in ("raw_samples", "result", "static_audit"):
            path = artifact_path(run, str(experiment.get("artifacts", {}).get(name, "")))
            if not path.is_file():
                missing_artifacts.append(name)
            elif pre_artifacts[name]["existed"] and pre_artifacts[name]["sha256"] == sha256(path):
                missing_artifacts.append(f"{name}:stale-preexisting")
        if missing_artifacts:
            failure = f"commands completed but required artifacts are missing: {missing_artifacts}"
    reproduction_path = artifact_path(run, str(experiment.get("artifacts", {}).get("reproduction_log", "")))
    reproduction_path.parent.mkdir(parents=True, exist_ok=True)
    reproduction = {
        "schema_version": "experiment-reproduction-v1",
        "request_id": args.request_id,
        "experiment_identity": receipt["experiment_identity"],
        "hardware_identity": receipt["hardware_identity"],
        "workload_identity": receipt["workload_identity"],
        "commands": [{"phase": item["phase"], "argv": item["argv"], "exit_code": item["exit_code"]} for item in receipt["commands"]],
    }
    atomic_json(reproduction_path, reproduction)
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    receipt["status"] = "FAIL" if failure else "PASS"
    receipt["failure"] = failure
    receipt["artifacts"] = {
        name: {"path": artifact_path(run, value).relative_to(run).as_posix(), "sha256": sha256(artifact_path(run, value))}
        for name, value in experiment.get("artifacts", {}).items()
        if artifact_path(run, value).is_file()
    }
    atomic_json(receipt_path, receipt)
    queue = read_object(queue_path)
    request = next(item for item in queue["requests"] if item.get("request_id") == args.request_id)
    receipt_identity = {"path": receipt_path.relative_to(run).as_posix(), "sha256": sha256(receipt_path), "status": receipt["status"]}
    request["execution_receipt"] = receipt_identity
    if failure:
        request["status"] = "BLOCKED"
        request["blocking_evidence"] = [receipt_identity]
    atomic_json(queue_path, queue)
    print(json.dumps({"status": receipt["status"], "request_id": args.request_id, "receipt": str(receipt_path), "failure": failure}, sort_keys=True))
    return 0 if not failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
