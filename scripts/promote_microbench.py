#!/usr/bin/env python3
"""Validate and append a run-local microbenchmark to the reusable catalog."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

from repository_rules import (
    CANDIDATE_STATUS,
    CATALOG_SCHEMA,
    PUBLISHED_STATUS,
    atomic_json,
    catalog_entry,
    read_object,
    validate_definition,
    validate_package_files,
    validate_promotion_evidence,
    validate_pure_text,
)


def inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    candidate = args.candidate.resolve()
    evidence_path = args.evidence.resolve()
    runs_root = root / "runs"
    if not inside(candidate, runs_root) or "microbench_candidates" not in candidate.parts:
        raise ValueError("candidate must live under runs/<run-id>/microbench_candidates/")
    if not inside(evidence_path, runs_root):
        raise ValueError("promotion evidence must live under the same repository runs/")
    definition = read_object(candidate / "benchmark.json")
    validate_definition(definition, allowed_statuses=(CANDIDATE_STATUS,))
    validate_package_files(candidate, definition)
    evidence = read_object(evidence_path)
    run_root = next(parent for parent in candidate.parents if parent.parent == runs_root)
    application_terms = validate_promotion_evidence(
        evidence,
        definition["id"],
        candidate=candidate,
        run_root=run_root,
        repository_root=root,
        definition=definition,
    )
    validate_pure_text(candidate, application_terms)

    catalog_path = root / "microbench" / "catalog.json"
    catalog = read_object(catalog_path)
    if catalog.get("schema_version") != CATALOG_SCHEMA:
        raise ValueError(f"catalog: expected {CATALOG_SCHEMA}")
    if any(entry["id"] == definition["id"] for entry in catalog["benchmarks"]):
        raise ValueError(f"published benchmark id already exists: {definition['id']}")
    destination = (root / "microbench" / definition["publish_path"]).resolve()
    if not inside(destination, (root / "microbench").resolve()):
        raise ValueError("publish_path escapes microbench/")
    if destination.exists():
        raise FileExistsError(f"published destination already exists: {destination}")

    if args.check_only:
        print(json.dumps({"status": "VALID", "candidate": str(candidate), "destination": str(destination)}, sort_keys=True))
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.promotion-", dir=destination.parent))
    try:
        for source in candidate.rglob("*"):
            relative = source.relative_to(candidate)
            target = staging / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        published = read_object(staging / "benchmark.json")
        published["status"] = PUBLISHED_STATUS
        atomic_json(staging / "benchmark.json", published)
        validate_definition(published, allowed_statuses=(PUBLISHED_STATUS,))
        validate_package_files(staging, published)
        validate_pure_text(staging, application_terms)

        updated_catalog = dict(catalog)
        updated_catalog["benchmarks"] = sorted(
            [*catalog["benchmarks"], catalog_entry(published)], key=lambda item: item["id"]
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
        atomic_json(catalog_path, updated_catalog)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    receipt = evidence_path.with_name(evidence_path.stem.replace(".promotion", "") + ".receipt.json")
    atomic_json(
        receipt,
        {
            "schema_version": "microbenchmark-promotion-receipt-v1",
            "candidate_id": definition["id"],
            "published_at": datetime.now(timezone.utc).isoformat(),
            "destination": destination.relative_to(root).as_posix(),
        },
    )
    print(json.dumps({"status": "PUBLISHED", "id": definition["id"], "destination": str(destination)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)
