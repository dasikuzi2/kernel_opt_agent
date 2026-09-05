#!/usr/bin/env python3
"""Fail closed when reusable repository zones contain generated or task-specific material."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from repository_rules import (
    CATALOG_SCHEMA,
    GENERATED_SUFFIXES,
    PUBLISHED_STATUS,
    catalog_entry,
    read_object,
    validate_definition,
    validate_package_files,
    validate_pure_text,
)
from evidence_utils import validate_hardware_spec, validate_identity


ALLOWED_TOP_LEVEL = {
    ".git",
    ".gitattributes",
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "REVIEW.md",
    "hardware",
    "knowledge",
    "microbench",
    "runs",
    "schemas",
    "scripts",
    "skill",
    "templates",
    "tests",
}

REUSABLE_ZONES = (
    "microbench",
    "hardware/adapters",
    "hardware/specs",
    "schemas",
    "scripts",
    "skill",
    "templates",
    "tests",
)

GLOBAL_CACHE_NAMES = {".DS_Store", ".mypy_cache", ".pytest_cache", "__pycache__"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []

    unexpected = sorted(path.name for path in root.iterdir() if path.name not in ALLOWED_TOP_LEVEL)
    if unexpected:
        errors.append(f"unexpected top-level entries: {unexpected}")

    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.name in GLOBAL_CACHE_NAMES or path.name.endswith(".tmp"):
            errors.append(f"forbidden cache/temp name: {path.relative_to(root)}")

    historical_terms = ["f" + "la", "q" + "wen", "g" + "dn", "delta" + "_rule"]
    for zone_name in (*REUSABLE_ZONES, "knowledge"):
        zone = root / zone_name
        if not zone.exists():
            if zone_name in REUSABLE_ZONES:
                errors.append(f"missing reusable zone: {zone_name}")
            continue
        for path in zone.rglob("*"):
            if path.is_file() and path.suffix.lower() in GENERATED_SUFFIXES:
                errors.append(f"generated artifact in reusable zone: {path.relative_to(root)}")
        try:
            validate_pure_text(zone, historical_terms)
        except Exception as error:
            errors.append(str(error))

    catalog_path = root / "microbench" / "catalog.json"
    try:
        catalog = read_object(catalog_path)
        if catalog.get("schema_version") != CATALOG_SCHEMA:
            errors.append(f"catalog schema is not {CATALOG_SCHEMA}")
        entries = catalog.get("benchmarks", [])
        if len({entry.get("id") for entry in entries}) != len(entries):
            errors.append("catalog contains duplicate benchmark ids")
        catalog_by_path = {entry.get("path"): entry for entry in entries}
        manifests = sorted((root / "microbench").rglob("benchmark.json"))
        manifest_paths = set()
        for manifest_path in manifests:
            package = manifest_path.parent
            relative_package = package.relative_to(root / "microbench").as_posix()
            manifest_paths.add(relative_package)
            try:
                definition = read_object(manifest_path)
                validate_definition(definition, allowed_statuses=(PUBLISHED_STATUS,))
                validate_package_files(package, definition)
                validate_pure_text(package, historical_terms)
                if definition["publish_path"] != relative_package:
                    errors.append(f"publish_path mismatch in {manifest_path.relative_to(root)}")
                if catalog_by_path.get(relative_package) != catalog_entry(definition):
                    errors.append(f"catalog entry mismatch for {relative_package}")
            except Exception as error:
                errors.append(str(error))
        if manifest_paths != set(catalog_by_path):
            errors.append(
                f"catalog/package path mismatch: manifests={sorted(manifest_paths)}, catalog={sorted(catalog_by_path)}"
            )
    except Exception as error:
        errors.append(f"catalog audit failed: {error}")

    for spec in sorted((root / "hardware/specs").glob("*.json")):
        if spec.name == "hardware_spec.schema.json":
            continue
        try:
            errors.extend(validate_hardware_spec(spec))
        except Exception as error:
            errors.append(f"hardware specification audit failed for {spec}: {error}")

    measurement_index = root / "hardware/measurements/index.json"
    try:
        index = read_object(measurement_index)
        if index.get("schema_version") != "hardware-measurement-index-v2":
            errors.append("hardware measurement index must use v2")
        seen_measurements = set()
        for record in index.get("records", []):
            measurement_id = record.get("id")
            if not measurement_id or measurement_id in seen_measurements:
                errors.append("hardware measurement index requires unique non-empty ids")
            seen_measurements.add(measurement_id)
            qualification = record.get("qualification")
            if qualification not in {"LEGACY_UNQUALIFIED", "EVIDENCE_CLOSED_V2"}:
                errors.append(f"hardware measurement {measurement_id}: invalid qualification")
            for field in ("hardware", "manifest", "summary"):
                validate_identity(root, record.get(field, {}), f"hardware measurement {measurement_id} {field}", errors, containment_root=root / "hardware/measurements")
            if qualification == "EVIDENCE_CLOSED_V2":
                for field in ("hardware_evidence", "p0_receipt", "source", "binary", "sass", "raw_samples"):
                    validate_identity(root, record.get(field, {}), f"hardware measurement {measurement_id} {field}", errors, containment_root=root / "hardware/measurements")
    except Exception as error:
        errors.append(f"hardware measurement index audit failed: {error}")

    result = {"status": "PASS" if not errors else "FAIL", "root": str(root), "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
