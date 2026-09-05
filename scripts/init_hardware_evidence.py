#!/usr/bin/env python3
"""Create a fail-closed hardware evidence manifest for an optimization run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_utils import read_object, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hardware = read_object(args.hardware)
    policy = read_object(args.policy)
    required = list(policy.get("required_document_roles", []))
    result = {
        "schema_version": "hardware-evidence-manifest-v2",
        "status": "BLOCKED",
        "target_identity": hardware.get("target", {}),
        "trusted_policy_identity": {
            "path": args.policy.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix(),
            "sha256": sha256(args.policy),
        },
        "official_source_policy": {
            "vendor": policy.get("vendor"),
            "allowed_official_domains": policy.get("allowed_official_domains", []),
            "policy": policy.get("policy", {}),
        },
        "required_document_roles": required,
        "sources": [],
        "facts": [],
        "empirical_measurements": [],
        "unknowns": [
            {"kind": "OFFICIAL_DOCUMENT", "role": role, "status": "MISSING"}
            for role in required
        ],
        "developer_input_request": {
            "missing_roles": required,
            "search_topics": policy.get("search_topics", {}),
            "message": "Agent must search vendor-official sources first. If an exact document for the target architecture/device cannot be found, ask the developer for its official document location. Do not record inferred hardware facts.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "BLOCKED", "output": str(args.output), "missing_roles": required}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
