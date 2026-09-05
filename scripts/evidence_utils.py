#!/usr/bin/env python3
"""Shared immutable-evidence and official-source validation helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from urllib.parse import urlparse


PERFORMANCE_TOKENS = (
    "bandwidth", "throughput", "latency", "peak", "flop", "tops", "rate",
    "issue_per", "bytes_per", "cycles",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_HARDWARE_EVIDENCE_SCHEMA = "hardware-evidence-manifest-v2"


def read_object(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_numeric(value, *, positive: bool = False) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and (number > 0 if positive else True)


def resolve_evidence_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def normalized_text(value: str) -> str:
    return " ".join(value.split())


def archived_text(path: Path) -> str:
    return normalized_text(path.read_text(encoding="utf-8", errors="ignore"))


def trusted_policy_path(vendor: str) -> Path:
    return PROJECT_ROOT / "hardware" / "adapters" / vendor.strip().lower() / "official_source_policy.json"


def target_fields_match(expected: dict, observed: dict) -> list[str]:
    errors: list[str] = []
    aliases = {
        "device_name": ("device_name", "name"),
        "compute_capability": ("compute_capability", "compute_cap"),
        "uuid": ("uuid",),
    }
    for canonical, names in aliases.items():
        left = next((expected.get(name) for name in names if expected.get(name) not in (None, "")), None)
        right = next((observed.get(name) for name in names if observed.get(name) not in (None, "")), None)
        if left is not None and right is not None and str(left).strip().lower() != str(right).strip().lower():
            errors.append(f"target identity mismatch for {canonical}: manifest={left!r}, hardware={right!r}")
    left_vendor = str(expected.get("vendor", "")).strip().upper()
    right_vendor = str(observed.get("vendor", "")).strip().upper()
    if not left_vendor or not right_vendor or left_vendor != right_vendor:
        errors.append(f"target identity mismatch for vendor: manifest={left_vendor!r}, hardware={right_vendor!r}")
    return errors


def domain_allowed(url: str, allowed_domains: list[str]) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(
        host == domain.lower() or host.endswith("." + domain.lower())
        for domain in allowed_domains
    )


def validate_identity(
    base: Path,
    record: dict,
    label: str,
    errors: list[str],
    *,
    containment_root: Path | None = None,
) -> None:
    path_value = record.get("path") or record.get("artifact_path")
    expected = record.get("sha256")
    if not path_value or not expected:
        errors.append(f"{label}: path/artifact_path and sha256 are required")
        return
    path = resolve_evidence_path(base, str(path_value))
    if containment_root is not None and not path_is_within(path, containment_root):
        errors.append(f"{label}: artifact escapes the allowed evidence root: {path}")
        return
    if not path.is_file():
        errors.append(f"{label}: immutable artifact does not exist: {path}")
        return
    actual = sha256(path)
    if actual != expected:
        errors.append(f"{label}: sha256 mismatch: expected {expected}, got {actual}")


def validate_locator(artifact: Path, locator: dict, label: str, errors: list[str]) -> None:
    if not isinstance(locator, dict):
        errors.append(f"{label}: locator must be a structured object")
        return
    for field in ("kind", "value", "support_text", "support_text_sha256"):
        if not locator.get(field):
            errors.append(f"{label}: locator.{field} is required")
    support = str(locator.get("support_text", ""))
    support_hash = hashlib.sha256(support.encode("utf-8")).hexdigest()
    if locator.get("support_text_sha256") != support_hash:
        errors.append(f"{label}: locator support-text hash mismatch")
    if artifact.is_file() and support and normalized_text(support) not in archived_text(artifact):
        errors.append(f"{label}: exact support text is not present in the archived official document")


def validate_retrieval_receipt(
    base: Path,
    source: dict,
    artifact: Path,
    allowed_domains: list[str],
    label: str,
    errors: list[str],
) -> None:
    retrieval = source.get("retrieval", {})
    if source.get("url") != retrieval.get("requested_url"):
        errors.append(f"{label}: source URL must equal the retrieval requested URL")
    if retrieval.get("method") != "AGENT_HTTPS_DOWNLOAD":
        errors.append(f"{label}: official documents require method=AGENT_HTTPS_DOWNLOAD")
    if retrieval.get("http_status") != 200:
        errors.append(f"{label}: official document retrieval must record HTTP 200")
    for field in ("requested_url", "final_url", "retriever", "retrieved_at"):
        if not retrieval.get(field):
            errors.append(f"{label}: retrieval.{field} is required")
    for field in ("requested_url", "final_url"):
        if retrieval.get(field) and not domain_allowed(str(retrieval[field]), allowed_domains):
            errors.append(f"{label}: retrieval.{field} is not on the trusted vendor domain allowlist")
    receipt_record = {
        "path": retrieval.get("receipt_path"),
        "sha256": retrieval.get("receipt_sha256"),
    }
    validate_identity(base, receipt_record, f"{label} retrieval receipt", errors, containment_root=base)
    receipt_path = resolve_evidence_path(base, str(retrieval.get("receipt_path", "")))
    if not receipt_path.is_file():
        return
    try:
        receipt = read_object(receipt_path)
    except Exception as error:
        errors.append(f"{label}: invalid retrieval receipt: {error}")
        return
    if receipt.get("schema_version") != "official-document-retrieval-v1":
        errors.append(f"{label}: invalid retrieval receipt schema")
    comparisons = {
        "requested_url": retrieval.get("requested_url"),
        "final_url": retrieval.get("final_url"),
        "http_status": retrieval.get("http_status"),
        "artifact_sha256": retrieval.get("sha256"),
        "artifact_path": retrieval.get("artifact_path"),
    }
    for field, expected in comparisons.items():
        if receipt.get(field) != expected:
            errors.append(f"{label}: retrieval receipt {field} does not match manifest")
    if artifact.is_file() and receipt.get("artifact_sha256") != sha256(artifact):
        errors.append(f"{label}: retrieval receipt is not bound to the archived artifact")


def source_applies_to_target(source: dict, target: dict, label: str, errors: list[str]) -> None:
    applicability = source.get("applicability", {})
    vendor = str(target.get("vendor", "")).upper()
    if str(applicability.get("vendor", "")).upper() != vendor:
        errors.append(f"{label}: source applicability vendor does not match target")
    device_name = str(target.get("device_name") or target.get("name") or "").strip().lower()
    compute = str(target.get("compute_capability") or target.get("compute_cap") or "").strip().lower()
    models = {str(item).strip().lower() for item in applicability.get("device_models", [])}
    computes = {str(item).strip().lower() for item in applicability.get("compute_capabilities", [])}
    role = source.get("role")
    if role == "device_product_specification":
        if not device_name or device_name not in models:
            errors.append(f"{label}: product specification is not explicitly applicable to target device {device_name!r}")
    elif compute:
        if compute not in computes:
            errors.append(f"{label}: document is not explicitly applicable to compute capability {compute!r}")
    elif device_name not in models:
        errors.append(f"{label}: document has no exact target architecture/device applicability")


def validate_hardware_evidence(manifest_path: Path, hardware_path: Path | None = None) -> list[str]:
    manifest_path = manifest_path.resolve()
    base = manifest_path.parent
    errors: list[str] = []
    try:
        data = read_object(manifest_path)
    except Exception as error:
        return [str(error)]
    if data.get("schema_version") != SUPPORTED_HARDWARE_EVIDENCE_SCHEMA:
        errors.append("hardware evidence: invalid schema_version")
    target = data.get("target_identity", {})
    vendor = str(target.get("vendor", ""))
    policy_path = trusted_policy_path(vendor)
    if not policy_path.is_file():
        errors.append(f"hardware evidence: no repository-trusted adapter policy for vendor {vendor!r}")
        trusted_policy = {}
    else:
        trusted_policy = read_object(policy_path)
    policy_identity = data.get("trusted_policy_identity", {})
    expected_policy_path = policy_path.resolve().relative_to(PROJECT_ROOT).as_posix() if policy_path.exists() else ""
    if policy_identity.get("path") != expected_policy_path or policy_identity.get("sha256") != (sha256(policy_path) if policy_path.exists() else None):
        errors.append("hardware evidence: trusted policy identity is missing or does not match the repository adapter")
    policy = data.get("official_source_policy", {})
    allowed_domains = list(trusted_policy.get("allowed_official_domains", []))
    if policy.get("vendor") != trusted_policy.get("vendor") or policy.get("allowed_official_domains") != allowed_domains:
        errors.append("hardware evidence: policy snapshot does not exactly match the repository-trusted adapter")
    required_roles = set(trusted_policy.get("required_document_roles", []))
    if set(data.get("required_document_roles", [])) != required_roles:
        errors.append("hardware evidence: required document roles do not match the trusted adapter policy")
    if hardware_path is None and (base / "hardware.json").is_file():
        hardware_path = base / "hardware.json"
    if hardware_path is not None and hardware_path.is_file():
        errors.extend(target_fields_match(target, read_object(hardware_path).get("target", {})))
    sources = data.get("sources", [])
    source_ids: dict[str, dict] = {}
    covered_roles: set[str] = set()
    for index, source in enumerate(sources if isinstance(sources, list) else []):
        label = f"hardware source {index}"
        source_id = source.get("source_id")
        if not source_id:
            errors.append(f"{label}: source_id is required")
            continue
        if source_id in source_ids:
            errors.append(f"{label}: duplicate source_id {source_id}")
        source_ids[str(source_id)] = source
        authority = source.get("authority")
        if authority != "VENDOR_OFFICIAL_DOCUMENT":
            errors.append(f"{label}: hardware document sources must use VENDOR_OFFICIAL_DOCUMENT")
        for field in ("role", "title", "version", "locator", "applicability"):
            if not source.get(field):
                errors.append(f"{label}: {field} is required")
        if source.get("role") in required_roles and authority == "VENDOR_OFFICIAL_DOCUMENT":
            covered_roles.add(str(source["role"]))
        elif source.get("role") not in required_roles:
            errors.append(f"{label}: role is not declared by the trusted adapter policy")
        if authority == "VENDOR_OFFICIAL_DOCUMENT":
            url = source.get("url")
            if not url or not domain_allowed(str(url), allowed_domains):
                errors.append(f"{label}: URL is not on the declared vendor-official domains")
        retrieval = source.get("retrieval", {})
        validate_identity(base, retrieval, f"{label} retrieval", errors, containment_root=base)
        artifact = resolve_evidence_path(base, str(retrieval.get("artifact_path", "")))
        validate_retrieval_receipt(base, source, artifact, allowed_domains, label, errors)
        validate_locator(artifact, source.get("locator", {}), label, errors)
        source_applies_to_target(source, target, label, errors)
        role_review = source.get("role_review", {})
        if role_review.get("status") != "APPROVED" or not role_review.get("reviewer") or not role_review.get("reviewed_at"):
            errors.append(f"{label}: document-role applicability requires an explicit semantic review")
    missing_roles = sorted(required_roles - covered_roles)
    if missing_roles:
        errors.append(f"hardware evidence: missing official document roles {missing_roles}")

    fact_ids: set[str] = set()
    for index, fact in enumerate(data.get("facts", [])):
        label = f"hardware fact {index}"
        fact_id = fact.get("fact_id")
        if not fact_id:
            errors.append(f"{label}: fact_id is required")
        elif str(fact_id) in fact_ids:
            errors.append(f"{label}: duplicate fact_id {fact_id}")
        else:
            fact_ids.add(str(fact_id))
        for field in ("scope", "field", "value", "evidence_class", "source_id", "locator"):
            if fact.get(field) is None or fact.get(field) == "":
                errors.append(f"{label}: {field} is required")
        source = source_ids.get(str(fact.get("source_id")))
        if source is None:
            errors.append(f"{label}: source_id is not present in sources")
            continue
        evidence_class = fact.get("evidence_class")
        if evidence_class == "DOCUMENTED_FACT" and source.get("authority") != "VENDOR_OFFICIAL_DOCUMENT":
            errors.append(f"{label}: DOCUMENTED_FACT requires a vendor-official document")
        if evidence_class == "OFFICIAL_DEVICE_QUERY" and source.get("authority") != "VENDOR_OFFICIAL_TOOL":
            errors.append(f"{label}: OFFICIAL_DEVICE_QUERY requires a vendor-official tool output")
        field_name = str(fact.get("field", "")).lower()
        if evidence_class == "OFFICIAL_DEVICE_QUERY" and any(token in field_name for token in PERFORMANCE_TOKENS):
            errors.append(f"{label}: performance parameters cannot be promoted from a device query")
        artifact = resolve_evidence_path(base, str(source.get("retrieval", {}).get("artifact_path", "")))
        validate_locator(artifact, fact.get("locator", {}), label, errors)
        review = fact.get("semantic_review", {})
        if review.get("status") != "APPROVED" or not review.get("reviewer") or not review.get("reviewed_at"):
            errors.append(f"{label}: documented fact requires an explicit semantic-review receipt")

    for index, measurement in enumerate(data.get("empirical_measurements", [])):
        label = f"empirical measurement {index}"
        if measurement.get("evidence_class") != "TARGET_MEASUREMENT":
            errors.append(f"{label}: evidence_class must be TARGET_MEASUREMENT")
        if not measurement.get("measurement_id"):
            errors.append(f"{label}: measurement_id is required")
        identity = measurement.get("identity", {})
        validate_identity(base, identity, f"{label} identity", errors, containment_root=base)
        measurement_index = PROJECT_ROOT / "hardware/measurements/index.json"
        registered = {}
        if measurement_index.is_file():
            registered = {
                str(record.get("id")): record
                for record in read_object(measurement_index).get("records", [])
                if record.get("qualification") == "EVIDENCE_CLOSED_V2"
            }
        record = registered.get(str(measurement.get("measurement_id")))
        if record is None:
            errors.append(f"{label}: measurement_id is not registered as EVIDENCE_CLOSED_V2")
        elif identity.get("sha256") != record.get("summary", {}).get("sha256"):
            errors.append(f"{label}: identity must match the registered immutable measurement summary")

    if data.get("status") == "READY":
        if errors:
            errors.append("hardware evidence: READY is forbidden while validation errors remain")
        if data.get("developer_input_request") not in (None, {}):
            errors.append("hardware evidence: READY requires developer_input_request=null")
        if data.get("unknowns"):
            errors.append("hardware evidence: READY is forbidden while target hardware unknowns remain")
    elif data.get("status") == "BLOCKED":
        request = data.get("developer_input_request")
        if not isinstance(request, dict) or not request.get("missing_roles") or not request.get("message"):
            errors.append("hardware evidence: BLOCKED requires a concrete developer_input_request")
    else:
        errors.append("hardware evidence: status must be READY or BLOCKED")
    return errors


def validate_evidence_references(base: Path, records, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(records, list) or not records:
        return [f"{label}: immutable evidence records are required"]
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"{label} {index}: evidence must be an object with path and sha256")
            continue
        validate_identity(base, record, f"{label} {index}", errors, containment_root=base)
    return errors


def validate_p0_receipt(path: Path, run: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"P0 receipt is missing: {path}"]
    data = read_object(path)
    if data.get("schema_version") != "p0-calibration-receipt-v2":
        errors.append("P0 receipt: invalid schema_version")
    if data.get("status") != "PASS":
        errors.append("P0 receipt: status must PASS")
    controls = data.get("controls", {})
    required = (
        "timer_bracket", "zero_work", "live_sink", "graph_direct_equivalence",
        "clock_stability", "cold_warm_separation", "competing_load",
        "independent_process_replication",
    )
    for name in required:
        if controls.get(name, {}).get("status") != "PASS":
            errors.append(f"P0 receipt: control {name} must PASS")
    errors.extend(validate_evidence_references(run, data.get("raw_evidence"), "P0 raw evidence"))
    validate_identity(run, data.get("environment_identity", {}), "P0 environment identity", errors, containment_root=run)
    validate_identity(run, data.get("live_sink_identity", {}), "P0 live sink identity", errors, containment_root=run)
    validate_identity(run, data.get("input_identity", {}), "P0 calibration input", errors, containment_root=run)
    input_path = resolve_evidence_path(run, str(data.get("input_identity", {}).get("path", "")))
    if input_path.is_file():
        try:
            from p0_utils import evaluate_p0
            expected_controls = evaluate_p0(read_object(input_path))
            if controls != expected_controls:
                errors.append("P0 receipt: controls do not match deterministic recomputation from raw input")
        except Exception as error:
            errors.append(f"P0 receipt: cannot recompute controls: {error}")
    return errors


def validate_hardware_spec(path: Path, allowed_domains: list[str] | None = None) -> list[str]:
    errors: list[str] = []
    data = read_object(path)
    if data.get("schema_version") != "hardware-spec-v3":
        errors.append(f"{path}: hardware specification must use hardware-spec-v3")
    policy_path = trusted_policy_path(str(data.get("vendor", "")))
    if not policy_path.is_file():
        errors.append(f"{path}: repository-trusted vendor policy is missing")
        trusted_domains = []
    else:
        trusted_domains = read_object(policy_path).get("allowed_official_domains", [])
        policy_identity = data.get("trusted_policy_identity", {})
        if policy_identity.get("path") != policy_path.relative_to(PROJECT_ROOT).as_posix() or policy_identity.get("sha256") != sha256(policy_path):
            errors.append(f"{path}: trusted policy identity mismatch")
    allowed_domains = trusted_domains
    source_ids = set()
    trusted_roles = set(read_object(policy_path).get("required_document_roles", [])) if policy_path.is_file() else set()
    sources = data.get("sources", [])
    if not sources:
        errors.append(f"{path}: at least one vendor-official source is required")
    for index, source in enumerate(sources):
        label = f"{path} source {index}"
        source_id = source.get("source_id")
        if not source_id or source_id in source_ids:
            errors.append(f"{label}: unique source_id is required")
        source_ids.add(source_id)
        if source.get("authority") != "VENDOR_OFFICIAL_DOCUMENT":
            errors.append(f"{label}: authority must be VENDOR_OFFICIAL_DOCUMENT")
        if source.get("role") not in trusted_roles:
            errors.append(f"{label}: role is not admitted by the trusted adapter")
        if allowed_domains and not domain_allowed(str(source.get("url", "")), allowed_domains):
            errors.append(f"{label}: URL is not on the vendor-official domain allowlist")
        for field in ("url", "title", "version", "role", "locator", "applicability", "role_review", "retrieval"):
            if not source.get(field):
                errors.append(f"{label}: {field} is required")
        retrieval = source.get("retrieval", {})
        digest = str(retrieval.get("sha256", ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            errors.append(f"{label}: retrieval sha256 must be an exact lowercase SHA-256")
        artifact = resolve_evidence_path(path.parent, str(retrieval.get("artifact_path", "")))
        if not path_is_within(artifact, path.parent) or not artifact.is_file() or (artifact.is_file() and sha256(artifact) != digest):
            errors.append(f"{label}: archived official artifact is missing, outside the specification directory, or hash-mismatched")
        validate_locator(artifact, source.get("locator", {}), label, errors)
        validate_retrieval_receipt(path.parent, source, artifact, trusted_domains, label, errors)
        applicability = source.get("applicability", {})
        if str(applicability.get("vendor", "")).upper() != str(data.get("vendor", "")).upper():
            errors.append(f"{label}: applicability vendor mismatch")
        identity = data.get("identity", {})
        if source.get("role") == "device_product_specification":
            required_devices = {str(item).lower() for item in identity.get("device_models", [])}
            supported_devices = {str(item).lower() for item in applicability.get("device_models", [])}
            if not required_devices or not required_devices <= supported_devices:
                errors.append(f"{label}: product document does not cover every specification device model")
        elif str(identity.get("architecture", "")).lower() not in {
            str(item).lower() for item in applicability.get("architectures", [])
        }:
            errors.append(f"{label}: document does not explicitly cover the specification architecture")
        review = source.get("role_review", {})
        if review.get("status") != "APPROVED" or not all(review.get(field) for field in ("reviewer", "reviewed_at", "claim")):
            errors.append(f"{label}: approved document-role semantic review is required")
    spec_fact_ids = set()
    for index, fact in enumerate(data.get("facts", [])):
        if fact.get("fact_id"):
            spec_fact_ids.add(str(fact["fact_id"]))
        if fact.get("source_id") not in source_ids or not fact.get("locator"):
            errors.append(f"{path} fact {index}: exact official source and locator are required")
        if fact.get("evidence_class") != "DOCUMENTED_FACT":
            errors.append(f"{path} fact {index}: evidence_class must be DOCUMENTED_FACT")
        source = next((item for item in sources if item.get("source_id") == fact.get("source_id")), {})
        artifact = resolve_evidence_path(path.parent, str(source.get("retrieval", {}).get("artifact_path", "")))
        validate_locator(artifact, fact.get("locator", {}), f"{path} fact {index}", errors)
        if fact.get("semantic_review", {}).get("status") != "APPROVED" or not all(
            fact.get("semantic_review", {}).get(field) for field in ("reviewer", "reviewed_at", "claim")
        ):
            errors.append(f"{path} fact {index}: semantic review is required")
    for index, rate in enumerate(data.get("theoretical_rates", [])):
        if not set(rate.get("source_ids", [])) <= source_ids or not rate.get("source_ids"):
            errors.append(f"{path} theoretical rate {index}: official source_ids are required")
        for field in ("conditions", "derivation"):
            if not rate.get(field):
                errors.append(f"{path} theoretical rate {index}: {field} is required")
        input_facts = set(map(str, rate.get("input_fact_ids", [])))
        if not input_facts or not input_facts <= spec_fact_ids:
            errors.append(f"{path} theoretical rate {index}: reviewed documented input_fact_ids are required")
        if rate.get("semantic_review", {}).get("status") != "APPROVED" or not all(
            rate.get("semantic_review", {}).get(field) for field in ("reviewer", "reviewed_at", "claim")
        ):
            errors.append(f"{path} theoretical rate {index}: semantic review is required")
        if not finite_numeric(rate.get("value"), positive=True) or not rate.get("unit"):
            errors.append(f"{path} theoretical rate {index}: positive value and unit are required")
    return errors
