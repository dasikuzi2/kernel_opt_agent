#!/usr/bin/env python3
"""Create a frozen optimization run after the mandatory three-part intake."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from evidence_utils import resolve_evidence_path, validate_hardware_evidence
from schema_utils import validate_instance


REMINDER = """Kernel optimization intake is required:
1. Operator computation: equations/pseudocode, tensors, numerics, ABI and legal rewrites.
2. Target workload: cases and weights, modes/layouts/concurrency, graph/cache semantics and objective.
3. Target hardware: exact device or auto-discovery permission, software stack, clocks/power and allowed features.
4. Hardware evidence: the agent must locate exact vendor-official programming-model, ISA, architecture and device documents. If an exact document cannot be found, the developer must provide its location. Undocumented hardware facts are forbidden.
Historical values are never reused silently; name a prior run and confirm reuse explicitly.
"""


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(data: dict, expected_schema: str, fields: tuple[str, ...], label: str) -> None:
    if data.get("schema_version") != expected_schema:
        raise ValueError(f"{label}: expected schema_version={expected_schema!r}")
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValueError(f"{label}: missing required fields {missing}")


def validate_intake_semantics(operator: dict, workload: dict, hardware: dict) -> None:
    if not operator.get("name") or not operator.get("computation", {}).get("definition"):
        raise ValueError("operator: name and computation.definition must be non-empty")
    if not operator.get("inputs") or not operator.get("outputs"):
        raise ValueError("operator: inputs and outputs must be non-empty")
    for tensor in [*operator.get("inputs", []), *operator.get("outputs", [])]:
        for field in ("name", "dtype", "shape", "layout"):
            if field not in tensor or tensor.get(field) in (None, ""):
                raise ValueError(f"operator tensor contract requires {field}")
    for field in ("reference", "acceptance"):
        if not operator.get("numerics", {}).get(field):
            raise ValueError(f"operator numerics.{field} must be non-empty")
    if not operator.get("abi", {}).get("entrypoint"):
        raise ValueError("operator abi.entrypoint must be non-empty")
    seen = set()
    for case in workload.get("cases", []):
        case_id = str(case.get("id", ""))
        if not case_id or case_id in seen or not isinstance(case.get("parameters"), dict) or float(case.get("weight", 0.0)) < 0:
            raise ValueError("workload cases require unique ids, parameter objects and non-negative weights")
        seen.add(case_id)
    objective = workload.get("objective", {})
    if objective.get("direction") not in {"minimize", "maximize"} or not objective.get("metric") or not objective.get("statistic"):
        raise ValueError("workload objective is incomplete")
    target = hardware.get("target", {})
    if not target.get("vendor") or not target.get("device_name"):
        raise ValueError("hardware target requires exact vendor and device_name")
    if not hardware.get("provenance", {}).get("queries"):
        raise ValueError("hardware snapshot requires non-empty query provenance")


def validate_intake_schemas(project_root: Path, artifacts: tuple[tuple[str, dict, str], ...]) -> None:
    """Reject a malformed intake before creating any run directories or files."""
    errors: list[str] = []
    for label, instance, schema_name in artifacts:
        schema_path = project_root / "schemas" / schema_name
        schema = read_json(schema_path)
        errors.extend(f"{label}: {error}" for error in validate_instance(instance, schema))
    if errors:
        raise ValueError("intake schema validation failed: " + "; ".join(errors))


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return value[:48] or "operator"


def intake_template() -> dict:
    return {
        "operator": {
            "schema_version": "operator-contract-v1",
            "name": "",
            "computation": {"definition": "", "dependencies": []},
            "inputs": [],
            "outputs": [],
            "numerics": {"reference": "", "acceptance": ""},
            "abi": {"entrypoint": "", "mutable_arguments": []},
        },
        "workload": {
            "schema_version": "workload-v1",
            "name": "",
            "cases": [{"id": "", "parameters": {}, "weight": 1.0}],
            "objective": {
                "metric": "gpu_active_us",
                "statistic": "median",
                "direction": "minimize",
            },
        },
        "hardware": "Run scripts/discover_hardware.py or provide a hardware-snapshot-v1 JSON file.",
        "hardware_evidence": "Optional at intake. If omitted, the run is created BLOCKED at planning until exact vendor-official documents are archived and validated.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--hardware-evidence", type=Path)
    parser.add_argument("--run-id")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=project_root, help="output root containing runs/; templates always come from this repository")
    parser.add_argument("--print-intake", action="store_true")
    parser.add_argument("--test-legacy-contract", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    print(REMINDER, file=sys.stderr)
    if args.print_intake:
        print(json.dumps(intake_template(), indent=2, sort_keys=True))
        return 0
    missing = [
        name
        for name, value in (
            ("--operator", args.operator),
            ("--workload", args.workload),
            ("--hardware", args.hardware),
        )
        if value is None
    ]
    if missing:
        print(f"Intake incomplete: missing {', '.join(missing)}", file=sys.stderr)
        print("Use --print-intake for a neutral template.", file=sys.stderr)
        return 2

    operator = read_json(args.operator)
    workload = read_json(args.workload)
    hardware = read_json(args.hardware)
    require(operator, "operator-contract-v1", ("name", "computation", "inputs", "outputs", "numerics", "abi"), "operator")
    require(workload, "workload-v1", ("name", "cases", "objective"), "workload")
    require(hardware, "hardware-snapshot-v1", ("captured_at", "host", "target", "software", "tools", "provenance"), "hardware")
    validate_intake_schemas(project_root, (
        ("operator", operator, "operator_contract.schema.json"),
        ("workload", workload, "workload.schema.json"),
        ("hardware", hardware, "hardware_snapshot.schema.json"),
    ))
    validate_intake_semantics(operator, workload, hardware)
    if args.test_legacy_contract and str(hardware.get("target", {}).get("vendor", "")).upper() != "TEST":
        raise ValueError("--test-legacy-contract is restricted to synthetic repository tests")
    if not workload["cases"]:
        raise ValueError("workload: cases must not be empty")
    if sum(float(case.get("weight", 0.0)) for case in workload["cases"]) <= 0:
        raise ValueError("workload: at least one case weight must be positive")

    now = datetime.now(timezone.utc)
    run_id = args.run_id or f"{now:%Y%m%dT%H%M%SZ}_{slug(operator['name'])}"
    run_dir = args.root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")
    for child in (
        "raw",
        "models",
        "candidates",
        "static",
        "traces",
        "microbench_candidates",
        "experiments",
        "hardware_docs",
    ):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    frozen = {
        "operator.json": (args.operator, operator),
        "workload.json": (args.workload, workload),
        "hardware.json": (args.hardware, hardware),
    }
    identities = {}
    for name, (source, data) in frozen.items():
        (run_dir / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        identities[name] = {"source": str(source.resolve()), "sha256": digest(source)}
    evidence_path = run_dir / "hardware_evidence.json"
    if args.hardware_evidence:
        evidence = read_json(args.hardware_evidence)
        source_errors = validate_hardware_evidence(args.hardware_evidence, args.hardware)
        if source_errors:
            raise ValueError("hardware evidence is not READY and valid: " + "; ".join(source_errors))
        evidence = json.loads(json.dumps(evidence))
        for source in evidence.get("sources", []):
            retrieval = source.get("retrieval", {})
            original = resolve_evidence_path(args.hardware_evidence.parent, str(retrieval.get("artifact_path", "")))
            suffix = original.suffix or ".evidence"
            frozen_doc = run_dir / "hardware_docs" / f"{slug(str(source['source_id']))}{suffix}"
            shutil.copyfile(original, frozen_doc)
            retrieval["artifact_path"] = frozen_doc.relative_to(run_dir).as_posix()
            original_receipt = resolve_evidence_path(args.hardware_evidence.parent, str(retrieval.get("receipt_path", "")))
            frozen_receipt = run_dir / "hardware_docs" / f"{slug(str(source['source_id']))}.retrieval.json"
            receipt = read_json(original_receipt)
            receipt["artifact_path"] = retrieval["artifact_path"]
            frozen_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            retrieval["receipt_path"] = frozen_receipt.relative_to(run_dir).as_posix()
            retrieval["receipt_sha256"] = digest(frozen_receipt)
        for measurement in evidence.get("empirical_measurements", []):
            identity = measurement.get("identity", {})
            original = resolve_evidence_path(args.hardware_evidence.parent, str(identity.get("path", "")))
            frozen_measurement = run_dir / "hardware_docs" / f"measurement-{slug(str(measurement['measurement_id']))}.json"
            shutil.copyfile(original, frozen_measurement)
            identity["path"] = frozen_measurement.relative_to(run_dir).as_posix()
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        copied_errors = validate_hardware_evidence(evidence_path, run_dir / "hardware.json")
        if copied_errors:
            raise ValueError("frozen hardware evidence failed validation: " + "; ".join(copied_errors))
        identities["hardware_evidence.json"] = {"source": str(args.hardware_evidence.resolve()), "sha256": digest(args.hardware_evidence)}
    else:
        policy_path = project_root / "hardware/adapters/nvidia/official_source_policy.json"
        if str(hardware.get("target", {}).get("vendor", "")).upper() == "NVIDIA":
            completed = subprocess.run([
                sys.executable, str(project_root / "scripts/init_hardware_evidence.py"),
                "--hardware", str(run_dir / "hardware.json"),
                "--policy", str(policy_path),
                "--output", str(evidence_path),
            ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)
        else:
            roles = ["programming_model", "instruction_set", "architecture_tuning", "device_product_specification"]
            evidence = {
                "schema_version": "hardware-evidence-manifest-v2",
                "status": "BLOCKED",
                "target_identity": hardware.get("target", {}),
                "trusted_policy_identity": {"path": "", "sha256": ""},
                "official_source_policy": {
                    "vendor": hardware.get("target", {}).get("vendor"),
                    "allowed_official_domains": [],
                    "policy": {"developer_document_required_when_adapter_is_missing": True},
                },
                "required_document_roles": roles,
                "sources": [], "facts": [], "empirical_measurements": [],
                "unknowns": [{"kind": "OFFICIAL_DOCUMENT", "role": role, "status": "MISSING"} for role in roles],
                "developer_input_request": {
                    "missing_roles": roles,
                    "message": "No vendor adapter exists. Developer must provide exact official document locations and official domains before planning; inferred hardware facts are forbidden.",
                },
            }
            evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    intake = {
        "schema_version": "optimization-intake-v1",
        "status": "READY",
        "run_id": run_id,
        "created_at": now.isoformat(),
        "operator": operator["name"],
        "workload": workload["name"],
        "target": hardware["target"],
        "source_identities": identities,
        "user_confirmation_required_for_reuse": True,
    }
    (run_dir / "00_intake.json").write_text(json.dumps(intake, indent=2, sort_keys=True) + "\n")
    (run_dir / "decision_log.jsonl").write_text("")
    templates = project_root / "templates" / "optimization_run"
    shutil.copyfile(templates / "work_ledger.json", run_dir / "models" / "work_ledger.json")
    shutil.copyfile(templates / "dag.json", run_dir / "models" / "dag.json")
    shutil.copyfile(templates / "optimization_plan.json", run_dir / "models" / "optimization_plan.json")
    discovery_pool = read_json(templates / "candidate_pool.json")
    discovery_pool["created_at"] = now.isoformat()
    (run_dir / "models" / "candidate_pool.json").write_text(
        json.dumps(discovery_pool, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    opportunity_map = read_json(templates / "opportunity_map.json")
    opportunity_map["created_at"] = now.isoformat()
    (run_dir / "models" / "opportunity_map.json").write_text(
        json.dumps(opportunity_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(templates / "microarchitecture_model.json", run_dir / "models" / "microarchitecture_model.json")
    shutil.copyfile(templates / "baseline.json", run_dir / "models" / "baseline.json")
    shutil.copyfile(templates / "microbenchmark_plan.json", run_dir / "models" / "microbenchmark_plan.json")
    shutil.copyfile(templates / "schedule_model.json", run_dir / "models" / "schedule_model.json")
    shutil.copyfile(templates / "global_schedule_state.json", run_dir / "models" / "global_schedule_state.json")
    shutil.copyfile(templates / "resource_balance.json", run_dir / "models" / "resource_balance.json")
    shutil.copyfile(templates / "tradeoff_frontier.json", run_dir / "models" / "tradeoff_frontier.json")
    shutil.copyfile(templates / "experiment_queue.json", run_dir / "models" / "experiment_queue.json")
    shutil.copyfile(templates / "model_validation.json", run_dir / "models" / "model_validation.json")
    shutil.copyfile(templates / "production_validation.json", run_dir / "models" / "production_validation.json")
    shutil.copyfile(templates / "resource_discovery.json", run_dir / "models" / "resource_discovery.json")
    shutil.copyfile(templates / "instruction_audit.json", run_dir / "static" / "instruction_audit.json")
    if args.test_legacy_contract:
        legacy_global = read_json(run_dir / "models/global_schedule_state.json")
        legacy_global["schema_version"] = "global-schedule-state-v1"
        legacy_global.pop("supervisor", None)
        (run_dir / "models/global_schedule_state.json").write_text(json.dumps(legacy_global, indent=2, sort_keys=True) + "\n")
        legacy_queue = read_json(run_dir / "models/experiment_queue.json")
        legacy_queue["schema_version"] = "experiment-request-queue-v1"
        (run_dir / "models/experiment_queue.json").write_text(json.dumps(legacy_queue, indent=2, sort_keys=True) + "\n")
        legacy_balance = read_json(run_dir / "models/resource_balance.json")
        legacy_balance["schema_version"] = "resource-balance-ledger-v1"
        (run_dir / "models/resource_balance.json").write_text(json.dumps(legacy_balance, indent=2, sort_keys=True) + "\n")
    state = {
        "schema_version": "optimization-run-state-v2" if args.test_legacy_contract else "optimization-run-state-v4",
        "run_id": run_id,
        "completed_phases": ["INTAKE"],
        "current_phase": "PLANNING",
        "allowed_next_phase": "BASELINE",
        "next_action": "Capture a correct discovery baseline, rank quantified global opportunities, implement a diverse production-candidate portfolio, then qualify survivors with the evidence-closed workflow.",
        "terminal": False,
    }
    if not args.test_legacy_contract:
        state["framework_contract_version"] = "evidence-closed-v2"
    (run_dir / "run_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "READY", "run_dir": str(run_dir), "current_phase": state["current_phase"], "next_action": state["next_action"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
