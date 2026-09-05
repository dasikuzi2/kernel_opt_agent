#!/usr/bin/env python3
"""Register an immutable hardware measurement in the repository index."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from evidence_utils import path_is_within, validate_hardware_evidence, validate_p0_receipt


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--hardware-evidence", type=Path, required=True)
    parser.add_argument("--p0-receipt", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--raw-samples", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    if index.get("schema_version") != "hardware-measurement-index-v2":
        raise ValueError("measurement index must use v2")
    repository_root = args.index.resolve().parents[2]
    measurement_root = args.index.resolve().parent
    paths = {
        "hardware": args.hardware, "manifest": args.manifest, "summary": args.summary,
        "hardware_evidence": args.hardware_evidence, "p0_receipt": args.p0_receipt,
        "source": args.source, "binary": args.binary, "sass": args.sass,
        "raw_samples": args.raw_samples,
    }
    for name, path in paths.items():
        if not path.resolve().is_file() or not path_is_within(path.resolve(), measurement_root):
            raise ValueError(f"{name} must be an archived file inside hardware/measurements")
    if any(record["id"] == args.id for record in index["records"]):
        raise ValueError(f"measurement id already registered: {args.id}")
    summary = json.loads(args.summary.read_text())
    if summary.get("status") != "VALID":
        raise ValueError("only VALID measurement suites may be registered")
    hardware_errors = validate_hardware_evidence(args.hardware_evidence, args.hardware)
    if hardware_errors:
        raise ValueError("hardware evidence is invalid: " + "; ".join(hardware_errors))
    p0_errors = validate_p0_receipt(args.p0_receipt, measurement_root)
    if p0_errors:
        raise ValueError("P0 evidence is invalid: " + "; ".join(p0_errors))
    def relative_identity(path: Path):
        return {"path": path.resolve().relative_to(repository_root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    index["records"].append({
        "id": args.id,
        "qualification": "EVIDENCE_CLOSED_V2",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        **{name: relative_identity(path) for name, path in paths.items()},
    })
    temporary = args.index.with_suffix(args.index.suffix + ".tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.index)
    print(json.dumps({"registered": args.id, "index": str(args.index)}, sort_keys=True))


if __name__ == "__main__":
    main()
