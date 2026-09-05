#!/usr/bin/env python3
"""Materialize a small FlashInfer Trace view without copying tensor blobs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def filter_workloads(lines: list[str], case_ids: set[str]) -> list[str]:
    selected: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        identifier = record.get("workload", {}).get("uuid")
        if identifier in case_ids:
            selected[identifier] = json.dumps(record, separators=(",", ":"))
    missing = sorted(case_ids - selected.keys())
    if missing:
        raise ValueError(f"screening cases absent from source workload: {missing}")
    return [selected[identifier] for identifier in sorted(selected)]


def _single_match(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern!r} below {root}, found {len(matches)}")
    return matches[0]


def materialize(source: Path, destination: Path, definition: str, manifest: dict) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if destination == source or source in destination.parents:
        raise ValueError("destination must not be the source or a child of it")

    case_ids = {case["id"] for case in manifest.get("cases", [])}
    if not case_ids:
        raise ValueError("screening manifest has no cases")
    definition_source = _single_match(source / "definitions", f"{definition}.json")
    workload_source = _single_match(source / "workloads", f"{definition}.jsonl")
    filtered = filter_workloads(workload_source.read_text(encoding="utf-8").splitlines(), case_ids)

    relative_definition = definition_source.relative_to(source / "definitions")
    relative_workload = workload_source.relative_to(source / "workloads")
    definition_target = destination / "definitions" / relative_definition
    workload_target = destination / "workloads" / relative_workload
    definition_target.parent.mkdir(parents=True, exist_ok=True)
    workload_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(definition_source, definition_target)
    workload_target.write_text("\n".join(filtered) + "\n", encoding="utf-8")

    for name in ("blob", "solutions"):
        source_path = source / name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        os.symlink(source_path, destination / name, target_is_directory=True)

    return {
        "schema_version": "flashinfer-trace-subset-receipt-v1",
        "claim_scope": manifest.get("claim_scope"),
        "source": str(source),
        "destination": str(destination),
        "definition": definition,
        "case_count": len(filtered),
        "case_ids": sorted(case_ids),
        "storage_policy": "definition/workload copied; blob/solutions symlinked read-only by convention",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--definition", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt = materialize(args.source, args.destination, args.definition, manifest)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
