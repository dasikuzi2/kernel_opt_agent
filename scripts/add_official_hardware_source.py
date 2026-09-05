#!/usr/bin/env python3
"""Download, archive and bind one exact vendor-official hardware document."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from evidence_utils import (
    domain_allowed,
    normalized_text,
    read_object,
    sha256,
    trusted_policy_path,
    validate_hardware_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--locator-kind", choices=("SECTION", "TABLE", "HTML_ANCHOR", "TEXT"), required=True)
    parser.add_argument("--locator", required=True)
    parser.add_argument("--support-text", required=True, help="Exact short text present in the archived document")
    parser.add_argument("--url", required=True)
    parser.add_argument("--compute-capability", action="append", default=[])
    parser.add_argument("--device-model", action="append", default=[])
    parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    data = read_object(manifest_path)
    vendor = str(data.get("target_identity", {}).get("vendor", ""))
    policy_path = trusted_policy_path(vendor)
    policy = read_object(policy_path)
    allowed = policy.get("allowed_official_domains", [])
    if not domain_allowed(args.url, allowed):
        raise ValueError("URL is not on the declared vendor-official domain allowlist")
    if args.role not in policy.get("required_document_roles", []):
        raise ValueError("role is not required by this target's official-source policy")
    if any(source.get("source_id") == args.source_id for source in data.get("sources", [])):
        raise ValueError(f"duplicate source_id: {args.source_id}")
    archive_dir = manifest_path.parent / "hardware_docs"
    archive_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(args.url, headers={"User-Agent": "kernel-opt-agent-official-evidence/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        artifact_bytes = response.read()
        final_url = response.geturl()
        http_status = int(getattr(response, "status", 200))
        headers = dict(response.headers.items())
    if http_status != 200 or not domain_allowed(final_url, allowed):
        raise ValueError(f"official document retrieval failed or redirected outside trusted domains: {http_status} {final_url}")
    decoded = normalized_text(artifact_bytes.decode("utf-8", errors="ignore"))
    if normalized_text(args.support_text) not in decoded:
        raise ValueError("--support-text is not present in the bytes downloaded from the official URL")
    suffix = Path(urlparse(final_url).path).suffix or ".html"
    destination = archive_dir / f"{args.source_id}{suffix}"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_bytes(artifact_bytes)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    receipt_path = archive_dir / f"{args.source_id}.retrieval.json"
    receipt = {
        "schema_version": "official-document-retrieval-v1",
        "retrieved_at": retrieved_at,
        "retriever": "python urllib.request",
        "requested_url": args.url,
        "final_url": final_url,
        "http_status": http_status,
        "content_type": headers.get("Content-Type"),
        "etag": headers.get("ETag"),
        "last_modified": headers.get("Last-Modified"),
        "artifact_path": destination.relative_to(manifest_path.parent).as_posix(),
        "artifact_sha256": sha256(destination),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    source = {
        "source_id": args.source_id,
        "role": args.role,
        "authority": "VENDOR_OFFICIAL_DOCUMENT",
        "title": args.title,
        "version": args.version,
        "locator": {
            "kind": args.locator_kind,
            "value": args.locator,
            "support_text": args.support_text,
            "support_text_sha256": hashlib.sha256(args.support_text.encode("utf-8")).hexdigest(),
        },
        "applicability": {
            "vendor": vendor,
            "compute_capabilities": sorted(set(args.compute_capability)),
            "device_models": sorted(set(args.device_model)),
        },
        "role_review": {
            "status": "APPROVED",
            "reviewer": args.reviewer,
            "reviewed_at": retrieved_at,
            "claim": f"archived document supports role={args.role} for the declared applicability",
        },
        "url": args.url,
        "retrieval": {
            "method": "AGENT_HTTPS_DOWNLOAD",
            "retrieved_at": retrieved_at,
            "retriever": "python urllib.request",
            "requested_url": args.url,
            "final_url": final_url,
            "http_status": http_status,
            "artifact_path": destination.relative_to(manifest_path.parent).as_posix(),
            "sha256": sha256(destination),
            "receipt_path": receipt_path.relative_to(manifest_path.parent).as_posix(),
            "receipt_sha256": sha256(receipt_path),
        },
    }
    data.setdefault("sources", []).append(source)
    covered = {item.get("role") for item in data["sources"] if item.get("authority") == "VENDOR_OFFICIAL_DOCUMENT"}
    missing = sorted(set(data.get("required_document_roles", [])) - covered)
    data["unknowns"] = [item for item in data.get("unknowns", []) if item.get("role") not in covered]
    if missing:
        data["status"] = "BLOCKED"
        data["developer_input_request"] = {
            "missing_roles": missing,
            "message": "Search vendor-official sources for the missing roles; if exact documents cannot be located, request their locations from the developer.",
        }
    else:
        data["status"] = "READY"
        data["developer_input_request"] = None
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    errors = validate_hardware_evidence(manifest_path)
    if errors and data["status"] == "READY":
        raise ValueError("updated READY manifest failed validation: " + "; ".join(errors))
    print(json.dumps({"status": data["status"], "source": source, "missing_roles": missing}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
