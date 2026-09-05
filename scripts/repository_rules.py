#!/usr/bin/env python3
"""Shared validation rules for reusable benchmark packages."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path, PurePosixPath


DEFINITION_SCHEMA = "microbenchmark-definition-v2"
PROMOTION_SCHEMA = "microbenchmark-promotion-v2"
CATALOG_SCHEMA = "microbenchmark-catalog-v2"
PUBLISHED_STATUS = "PUBLISHED"
CANDIDATE_STATUS = "CANDIDATE"

REQUIRED_DEFINITION_FIELDS = (
    "schema_version",
    "id",
    "publish_path",
    "status",
    "vendor",
    "family",
    "questions",
    "controlled_variables",
    "source_files",
    "driver_files",
    "analyzer_files",
    "measurement_semantics",
    "correctness_controls",
    "dce_guard",
    "portability",
    "known_pollution",
    "claims_allowed",
    "claims_forbidden",
    "qualification",
    "genericity",
)

REQUIRED_MEASUREMENT_FIELDS = (
    "timing_scope",
    "cache_state",
    "repetitions",
    "launch_geometry",
)

REQUIRED_PROMOTION_CHECKS = (
    "correctness",
    "positive_and_negative_controls",
    "measurement_smoke",
    "clean_build",
    "cold_start_reproduction",
    "genericity_review",
    "static_instruction_validation",
    "mechanism_validation",
    "measurement_system_calibrated",
    "production_prediction_validation",
)

QUALIFICATION_ORDER = (
    "DRAFT",
    "STATIC_VALIDATED",
    "MECHANISM_VALIDATED",
    "DEVICE_CALIBRATED",
    "PRODUCTION_PREDICTIVE",
)

STATIC_PROMOTION_CHECKS = {
    "correctness",
    "positive_and_negative_controls",
    "measurement_smoke",
    "clean_build",
    "cold_start_reproduction",
    "genericity_review",
    "static_instruction_validation",
}

ALLOWED_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}

GENERATED_SUFFIXES = {
    ".a",
    ".bin",
    ".cubin",
    ".csv",
    ".fatbin",
    ".log",
    ".ncu-rep",
    ".nsys-rep",
    ".o",
    ".out",
    ".ptx",
    ".pyc",
    ".qdrep",
    ".qdstrm",
    ".sass",
    ".so",
}

FORBIDDEN_NAMES = {
    ".DS_Store",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "outputs",
    "profiles",
    "raw",
    "results",
    "traces",
}


def read_object(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def require_fields(data: dict, fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValueError(f"{label}: missing required fields {missing}")


def safe_relative(value: str, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: expected a non-empty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"{label}: unsafe path {value!r}")
    if any(not part or part.startswith(".") for part in pure.parts):
        raise ValueError(f"{label}: hidden or empty path component in {value!r}")
    return Path(*pure.parts)


def validate_definition(data: dict, allowed_statuses=(CANDIDATE_STATUS, PUBLISHED_STATUS)) -> None:
    require_fields(data, REQUIRED_DEFINITION_FIELDS, "benchmark definition")
    if data["schema_version"] != DEFINITION_SCHEMA:
        raise ValueError(f"benchmark definition: expected {DEFINITION_SCHEMA}")
    if data["status"] not in allowed_statuses:
        raise ValueError(f"benchmark definition: invalid status {data['status']!r}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*\.v[0-9]+", str(data["id"])):
        raise ValueError("benchmark definition: id must end in .vN and use lowercase stable characters")
    safe_relative(data["publish_path"], "publish_path")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._+-]*", str(data["vendor"])):
        raise ValueError("benchmark definition: invalid vendor")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", str(data["family"])):
        raise ValueError("benchmark definition: family must be lowercase kebab-case")
    for field in (
        "questions",
        "controlled_variables",
        "source_files",
        "driver_files",
        "correctness_controls",
        "claims_allowed",
        "claims_forbidden",
    ):
        if not isinstance(data[field], list) or not data[field]:
            raise ValueError(f"benchmark definition: {field} must be a non-empty list")
    if not isinstance(data["analyzer_files"], list):
        raise ValueError("benchmark definition: analyzer_files must be a list")
    if not isinstance(data["known_pollution"], list):
        raise ValueError("benchmark definition: known_pollution must be a list")
    qualification = data["qualification"]
    require_fields(
        qualification,
        ("highest_status", "levels", "static_validation", "mechanism_validation", "device_calibration", "production_prediction"),
        "qualification",
    )
    allowed_qualification = {
        "DRAFT",
        "STATIC_VALIDATED",
        "MECHANISM_VALIDATED",
        "DEVICE_CALIBRATED",
        "PRODUCTION_PREDICTIVE",
    }
    if qualification["highest_status"] not in allowed_qualification:
        raise ValueError("qualification: invalid highest_status")
    if not isinstance(qualification["levels"], list):
        raise ValueError("qualification: levels must be a list")
    if not set(qualification["levels"]).issubset({"P0", "P1", "P2", "P3", "P4"}):
        raise ValueError("qualification: levels must use P0-P4")
    highest_index = QUALIFICATION_ORDER.index(qualification["highest_status"])
    if highest_index >= 2 and not qualification["levels"]:
        raise ValueError("qualification: mechanism/device/production qualification requires P-level evidence")
    required_passes = (
        (1, "static_validation"),
        (2, "mechanism_validation"),
        (3, "device_calibration"),
        (4, "production_prediction"),
    )
    for threshold, field in required_passes:
        if highest_index >= threshold and qualification[field] != "PASS":
            raise ValueError(f"qualification: {field} must PASS for {qualification['highest_status']}")
    require_fields(data["measurement_semantics"], REQUIRED_MEASUREMENT_FIELDS, "measurement_semantics")
    genericity = data["genericity"]
    require_fields(
        genericity,
        (
            "application_independent",
            "production_dependencies",
            "hardware_parameters_explicit",
            "generated_artifacts_committed",
        ),
        "genericity",
    )
    if genericity["application_independent"] is not True:
        raise ValueError("genericity: application_independent must be true")
    if genericity["production_dependencies"] != []:
        raise ValueError("genericity: production_dependencies must be empty")
    if genericity["hardware_parameters_explicit"] is not True:
        raise ValueError("genericity: hardware_parameters_explicit must be true")
    if genericity["generated_artifacts_committed"] is not False:
        raise ValueError("genericity: generated_artifacts_committed must be false")


def declared_files(data: dict) -> set[Path]:
    files = {Path("benchmark.json")}
    for field in ("source_files", "driver_files", "analyzer_files"):
        for item in data[field]:
            files.add(safe_relative(item, field))
    return files


def text_files(package: Path) -> list[Path]:
    return sorted(path for path in package.rglob("*") if path.is_file())


def validate_package_files(package: Path, data: dict) -> None:
    expected = declared_files(data)
    actual: set[Path] = set()
    for path in package.rglob("*"):
        relative = path.relative_to(package)
        if path.is_symlink():
            raise ValueError(f"benchmark package: symlink forbidden: {relative}")
        if any(part in FORBIDDEN_NAMES for part in relative.parts):
            raise ValueError(f"benchmark package: generated directory/file forbidden: {relative}")
        if path.is_file():
            actual.add(relative)
            if path.suffix.lower() in GENERATED_SUFFIXES:
                raise ValueError(f"benchmark package: generated artifact forbidden: {relative}")
            if path.suffix.lower() not in ALLOWED_SOURCE_SUFFIXES:
                raise ValueError(f"benchmark package: unsupported source suffix: {relative}")
    if actual != expected:
        missing = sorted(str(item) for item in expected - actual)
        undeclared = sorted(str(item) for item in actual - expected)
        raise ValueError(f"benchmark package inventory mismatch: missing={missing}, undeclared={undeclared}")


def term_pattern(term: str) -> re.Pattern:
    escaped = re.escape(term)
    if term.replace("_", "").isalnum():
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def validate_pure_text(package: Path, application_terms: list[str]) -> None:
    historical_terms = ["f" + "la", "q" + "wen", "g" + "dn", "delta" + "_rule"]
    terms = historical_terms + [term for term in application_terms if term]
    forbidden_paths = ["/workspace/" + "dance/", "/Users/" + "a66100/"]
    patterns = [(term, term_pattern(term)) for term in terms]
    for path in text_files(package):
        text = path.read_text(encoding="utf-8", errors="strict")
        for term, pattern in patterns:
            if pattern.search(text):
                raise ValueError(f"benchmark package: application term {term!r} found in {path.relative_to(package)}")
        for prefix in forbidden_paths:
            if prefix in text:
                raise ValueError(f"benchmark package: task-specific absolute path found in {path.relative_to(package)}")


def file_identity(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def resolve_inside(root: Path, identity: dict, label: str) -> Path:
    if not isinstance(identity, dict):
        raise ValueError(f"{label}: identity must be an object")
    value = identity.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: identity path is required")
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not inside(path, root):
        raise ValueError(f"{label}: path escapes the optimization run")
    if not path.is_file():
        raise ValueError(f"{label}: artifact is missing: {path}")
    if identity.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"{label}: sha256 mismatch: {path}")
    return path


def inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_check_result(path: Path, check_id: str, candidate_identity: dict) -> None:
    result = read_object(path)
    if result.get("schema_version") != "microbenchmark-check-result-v1":
        raise ValueError(f"promotion check {check_id}: invalid check-result schema")
    if result.get("check_id") != check_id or result.get("status") != "PASS":
        raise ValueError(f"promotion check {check_id}: result does not record matching PASS")
    if result.get("candidate_identity") != candidate_identity:
        raise ValueError(f"promotion check {check_id}: candidate identity mismatch")
    if not isinstance(result.get("conclusion"), str) or not result["conclusion"].strip():
        raise ValueError(f"promotion check {check_id}: a concrete conclusion is required")
    if not isinstance(result.get("method"), str) or not result["method"].strip():
        raise ValueError(f"promotion check {check_id}: the validation method is required")


def validate_reproduction_receipt(path: Path, run_root: Path, candidate_identity: dict) -> tuple[str, set[tuple[str, str]]]:
    receipt = read_object(path)
    if receipt.get("schema_version") != "microbenchmark-reproduction-receipt-v1" or receipt.get("status") != "PASS":
        raise ValueError("promotion reproduction receipt must use v1 and record PASS")
    if receipt.get("candidate_identity") != candidate_identity:
        raise ValueError("promotion reproduction receipt candidate identity mismatch")
    executor = Path(__file__).with_name("execute_microbench_reproduction.py")
    executor_identity = receipt.get("executor_identity", {})
    if executor_identity.get("path") != "scripts/execute_microbench_reproduction.py" or executor_identity.get("sha256") != hashlib.sha256(executor.read_bytes()).hexdigest():
        raise ValueError("promotion reproduction receipt executor identity mismatch")
    process_id = str(receipt.get("independent_process_id", "")).strip()
    if not process_id:
        raise ValueError("promotion reproduction receipt requires independent_process_id")
    commands = receipt.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("promotion reproduction receipt requires executed commands")
    for index, command in enumerate(commands):
        argv = command.get("argv") if isinstance(command, dict) else None
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError(f"promotion reproduction command {index}: argv is invalid")
        if command.get("exit_code") != 0:
            raise ValueError(f"promotion reproduction command {index}: nonzero exit code")
        for stream in ("stdout", "stderr"):
            resolve_inside(run_root, command.get(stream, {}), f"promotion reproduction {index} {stream}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("promotion reproduction receipt requires immutable output artifacts")
    for index, identity in enumerate(artifacts):
        resolve_inside(run_root, identity, f"promotion reproduction artifact {index}")
    return process_id, {(str(item.get("path")), str(item.get("sha256"))) for item in artifacts}


def validate_promotion_evidence(
    data: dict,
    candidate_id: str,
    *,
    candidate: Path,
    run_root: Path,
    repository_root: Path,
    definition: dict,
) -> list[str]:
    require_fields(
        data,
        (
            "schema_version",
            "candidate_id",
            "checks",
            "application_terms_removed",
            "candidate_identity",
            "reproduction_receipts",
            "registered_measurement_ids",
        ),
        "promotion evidence",
    )
    if data["schema_version"] != PROMOTION_SCHEMA:
        raise ValueError(f"promotion evidence: expected {PROMOTION_SCHEMA}")
    if data["candidate_id"] != candidate_id:
        raise ValueError("promotion evidence: candidate_id mismatch")
    expected_candidate_identity = file_identity(candidate / "benchmark.json")
    observed_candidate_identity = data.get("candidate_identity")
    observed_candidate_path = resolve_inside(run_root, observed_candidate_identity, "promotion candidate identity")
    if observed_candidate_path != (candidate / "benchmark.json").resolve() or observed_candidate_identity.get("sha256") != expected_candidate_identity["sha256"]:
        raise ValueError("promotion evidence: candidate benchmark identity mismatch")
    require_fields(data["checks"], REQUIRED_PROMOTION_CHECKS, "promotion checks")
    level = definition["qualification"]["highest_status"]
    if level == "DRAFT":
        raise ValueError("promotion evidence: DRAFT candidates cannot be published")
    required_checks = set(STATIC_PROMOTION_CHECKS)
    if QUALIFICATION_ORDER.index(level) >= QUALIFICATION_ORDER.index("MECHANISM_VALIDATED"):
        required_checks.add("mechanism_validation")
    if QUALIFICATION_ORDER.index(level) >= QUALIFICATION_ORDER.index("DEVICE_CALIBRATED"):
        required_checks.add("measurement_system_calibrated")
    if level == "PRODUCTION_PREDICTIVE":
        required_checks.add("production_prediction_validation")
    check_evidence_keys: set[tuple[str, str]] = set()
    for name in REQUIRED_PROMOTION_CHECKS:
        check = data["checks"].get(name)
        if not isinstance(check, dict):
            raise ValueError(f"promotion check {name}: expected a structured check object")
        expected_status = "PASS" if name in required_checks else "NOT_APPLICABLE"
        if check.get("status") != expected_status:
            raise ValueError(f"promotion check {name}: expected {expected_status} for {level}")
        if expected_status == "PASS":
            evidence = check.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError(f"promotion check {name}: immutable evidence is required")
            for index, identity in enumerate(evidence):
                result_path = resolve_inside(run_root, identity, f"promotion check {name} evidence {index}")
                validate_check_result(result_path, name, observed_candidate_identity)
                check_evidence_keys.add((str(identity.get("path")), str(identity.get("sha256"))))
            if not isinstance(check.get("conclusion"), str) or not check["conclusion"].strip():
                raise ValueError(f"promotion check {name}: conclusion is required")
        elif check.get("evidence") not in ([], None):
            raise ValueError(f"promotion check {name}: NOT_APPLICABLE cannot carry evidence")
    if not isinstance(data["application_terms_removed"], list):
        raise ValueError("promotion evidence: application_terms_removed must be a list")
    receipts = data.get("reproduction_receipts")
    if not isinstance(receipts, list) or len(receipts) < 2:
        raise ValueError("promotion evidence: at least two independent reproduction receipts are required")
    validated_receipts = [
        validate_reproduction_receipt(
            resolve_inside(run_root, identity, f"promotion reproduction receipt {index}"),
            run_root,
            observed_candidate_identity,
        )
        for index, identity in enumerate(receipts)
    ]
    process_ids = {item[0] for item in validated_receipts}
    if len(process_ids) < 2:
        raise ValueError("promotion evidence: cold-start reproductions must use distinct process ids")
    executed_artifacts = set().union(*(item[1] for item in validated_receipts))
    if not check_evidence_keys <= executed_artifacts:
        raise ValueError("promotion evidence: every PASS check result must be produced by a trusted reproduction execution")
    measurement_ids = data.get("registered_measurement_ids")
    if not isinstance(measurement_ids, list):
        raise ValueError("promotion evidence: registered_measurement_ids must be a list")
    if QUALIFICATION_ORDER.index(level) >= QUALIFICATION_ORDER.index("DEVICE_CALIBRATED"):
        index_path = repository_root / "hardware/measurements/index.json"
        index = read_object(index_path)
        qualified = {
            record.get("id") for record in index.get("records", [])
            if record.get("qualification") == "EVIDENCE_CLOSED_V2"
        }
        if not measurement_ids or not set(measurement_ids) <= qualified:
            raise ValueError("promotion evidence: device qualification requires registered EVIDENCE_CLOSED_V2 measurements")
    elif measurement_ids:
        raise ValueError("promotion evidence: static/mechanism qualification cannot claim registered device calibration")
    return [str(term) for term in data["application_terms_removed"]]


def catalog_entry(data: dict) -> dict:
    return {
        "id": data["id"],
        "path": data["publish_path"],
        "vendor": data["vendor"],
        "family": data["family"],
        "status": PUBLISHED_STATUS,
        "qualification": data["qualification"],
        "questions": data["questions"],
        "capabilities": data.get("capabilities", {}),
        "requires": data.get("requires", []),
        "optional_evidence": data.get("optional_evidence", []),
    }


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
