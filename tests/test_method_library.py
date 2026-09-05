#!/usr/bin/env python3
"""Exercise transfer-aware method matching and stale-receipt routing."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from optimizer_step import discovery_action


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cli(command: str, *args: str, expected: int = 0) -> dict:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/kernel_opt.py"), command, *args],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert completed.returncode == expected, (completed.stdout, completed.stderr)
    return json.loads(completed.stdout) if completed.stdout.strip() else {}


def opportunity(identifier: str, families: list[str], evidence_sha256: str) -> dict:
    return {
        "opportunity_id": identifier,
        "name": identifier,
        "model_scope": "CURRENT_SCHEDULE",
        "source_model_term": identifier,
        "affected_stages": [identifier],
        "current_contribution_us": 10.0,
        "optimistic_gain_ceiling_us": 5.0,
        "likely_gain_interval_us": {"lower": 1.0, "upper": 3.0},
        "confidence": "MEDIUM",
        "rewrite_families": families,
        "implementation_budget_minutes": 10.0,
        "hypothesis": "remove globally visible work with a different architecture",
        "derivation": "baseline contribution multiplied by a bounded removable fraction",
        "evidence": [{"path": "models/baseline.json", "sha256": evidence_sha256, "claim": "current objective contribution"}],
    }


def main() -> None:
    validated = cli("method", "validate")
    assert validated["status"] == "PASS" and validated["card_count"] >= 5, validated

    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary) / "run"
        (run / "models").mkdir(parents=True)
        for name in ("operator.json", "workload.json", "hardware.json"):
            shutil.copyfile(ROOT / "tests" / "fixtures" / name, run / name)
        hardware = json.loads((run / "hardware.json").read_text(encoding="utf-8"))
        hardware["target"] = {"vendor": "NVIDIA", "device_name": "NVIDIA GeForce RTX 4060 Laptop GPU", "device_index": 0}
        write(run / "hardware.json", hardware)
        write(run / "models" / "baseline.json", {"status": "VALID", "correctness": {"status": "PASS"}})
        evidence_sha256 = hashlib.sha256((run / "models" / "baseline.json").read_bytes()).hexdigest()
        cli("candidate", "init", "--run", str(run), "--min-candidates", "2", "--max-candidates", "4", "--min-families", "2")
        cli("opportunity", "init", "--run", str(run), "--min-opportunities", "2", "--max-opportunities", "4", "--min-rewrite-families", "2", "--min-candidate-opportunities", "2")
        for spec in (
            opportunity("attention-overlap", ["async-pipeline"], evidence_sha256),
            opportunity("remove-intermediate", ["cross-stage-fusion"], evidence_sha256),
        ):
            path = run / "models" / f"{spec['opportunity_id']}.json"
            write(path, spec)
            cli("opportunity", "add", "--run", str(run), "--spec", str(path))
        cli("opportunity", "rank", "--run", str(run))

        retrieve = discovery_action(run, ROOT / "scripts")
        assert retrieve and retrieve["action"] == "RETRIEVE_OPTIMIZATION_METHODS", retrieve
        receipt = cli("method", "recommend", "--run", str(run))
        by_opportunity = {row["opportunity_id"]: row["matches"] for row in receipt["recommendations"]}
        attention = {row["method_id"]: row for row in by_opportunity["attention-overlap"]}
        assert attention["flashattention3-async-pipeline"]["transfer_status"] == "BLOCKED_UNVERIFIED_CAPABILITY", attention
        fusion = {row["method_id"]: row for row in by_opportunity["remove-intermediate"]}
        assert fusion["korch-fission-orchestration"]["transfer_status"] == "DIRECT", fusion
        assert receipt["evaluation_guards"][0]["claim_boundary"] == "DISCOVERY_PRIOR_ONLY"

        expand = discovery_action(run, ROOT / "scripts")
        assert expand and expand["action"] == "EXPAND_DISCOVERY_PORTFOLIO", expand
        assert expand["blocking_inputs"][0]["transfer_aware_method_matches"], expand

        receipt_path = run / "models" / "method_matches.json"
        edited = json.loads(receipt_path.read_text(encoding="utf-8"))
        edited["recommendations"][0]["matches"][0]["score"] += 1000
        write(receipt_path, edited)
        tampered = discovery_action(run, ROOT / "scripts")
        assert tampered and tampered["action"] == "RETRIEVE_OPTIMIZATION_METHODS", tampered
        assert "edited" in tampered["blocking_inputs"][0]["stale_or_missing_receipt"], tampered
        cli("method", "recommend", "--run", str(run))

        operator = json.loads((run / "operator.json").read_text(encoding="utf-8"))
        operator["name"] = "changed-after-match"
        write(run / "operator.json", operator)
        stale = discovery_action(run, ROOT / "scripts")
        assert stale and stale["action"] == "RETRIEVE_OPTIMIZATION_METHODS", stale
        assert "stale" in stale["blocking_inputs"][0]["stale_or_missing_receipt"], stale

    print("method library test: PASS")


if __name__ == "__main__":
    main()
