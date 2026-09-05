#!/usr/bin/env python3
"""Exercise repairable discovery, screening and qualification promotion."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from optimizer_step import discovery_action


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_cli(run: Path, *args: str, expected: int = 0) -> dict:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/kernel_opt.py"), "candidate", *args, "--run", str(run)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == expected, (completed.stdout, completed.stderr)
    if expected == 0:
        assert completed.stdout.strip(), (args, completed.returncode, completed.stdout, completed.stderr)
    else:
        assert completed.stderr.strip(), (args, completed.returncode, completed.stdout, completed.stderr)
    return json.loads(completed.stdout) if completed.stdout.strip() else {}


def opportunity_cli(run: Path, *args: str, expected: int = 0) -> dict:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/kernel_opt.py"), "opportunity", *args, "--run", str(run)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert completed.returncode == expected, (completed.stdout, completed.stderr)
    return json.loads(completed.stdout) if completed.stdout.strip() else {}


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary) / "runs" / "discovery"
        (run / "models").mkdir(parents=True)
        workspace = run / "candidates" / "c1" / "workspace"
        workspace.mkdir(parents=True)
        write(run / "models/baseline.json", {
            "schema_version": "production-baseline-v1",
            "status": "VALID",
            "correctness": {"status": "PASS", "evidence": []},
        })
        write(run / "models/experiment_queue.json", {"requests": []})
        write(workspace / "kernel.py", "VALUE = 1\n")
        kernel_hash = hashlib.sha256((workspace / "kernel.py").read_bytes()).hexdigest()
        write(workspace / "build.py", "from pathlib import Path\nraise SystemExit(0 if Path('ready.flag').exists() else 3)\n")
        write(workspace / "correctness.py", "raise SystemExit(0)\n")
        write(workspace / "smoke.py", """
import json
import hashlib
from pathlib import Path
result = {
    "schema_version": "candidate-smoke-result-v3",
    "status": "PASS",
    "candidate_id": "c1",
    "objective": {"direction": "minimize", "baseline": 10.0, "candidate": 8.0, "unit": "us_weighted"},
    "cases": [{"case_id": "anchor", "role": "ANCHOR"}, {"case_id": "edge", "role": "EDGE"}],
    "reachability": {
        "status": "PASS",
        "expected_path": "test-candidate-kernel",
        "observed_path": "test-candidate-kernel",
        "compile_cache_policy": "NOT_COMPILED",
        "execution_proof": {
            "kind": "DIRECT_SENTINEL",
            "scope": "test-candidate-kernel direct smoke invocation",
            "observed_count": 1,
            "minimum_count": 1,
            "evidence_index": 0,
        },
        "evidence": [{
            "path": "candidates/c1/workspace/kernel.py",
            "sha256": hashlib.sha256(Path("kernel.py").read_bytes()).hexdigest(),
        }],
    },
}
Path('../smoke.json').write_text(json.dumps(result))
""".lstrip())
        spec_path = run / "candidates" / "c1" / "spec.json"
        candidate_spec = {
            "candidate_id": "c1",
            "opportunity_id": "fuse-transfer",
            "name": "repairable candidate",
            "family": "layout-redesign",
            "change_axes": ["layout", "warp-ownership"],
            "hypothesis": "remove a materialized transfer",
            "expected_global_effect": "reduce weighted production latency",
            "predicted_global_gain_us": {"lower": 1.0, "upper": 3.0},
            "dependency_contract": {
                "status": "PROVEN_LEGAL",
                "preserved_dependencies": ["the output still depends on the same input values"],
                "changed_boundaries": ["the candidate changes only the implementation boundary"],
                "prohibited_rewrites": ["cached or constant outputs are forbidden"],
                "numerical_ordering": "The fixture preserves its scalar operation order.",
                "evidence": [{
                    "path": "candidates/c1/workspace/kernel.py",
                    "sha256": kernel_hash,
                    "claim": "candidate implementation used by the dependency audit",
                }],
            },
            "source_paths": ["candidates/c1/workspace/kernel.py"],
            "commands": {
                "build": {"argv": ["{python}", "build.py"], "cwd": "candidates/c1/workspace", "timeout_seconds": 30},
                "correctness": {"argv": ["{python}", "correctness.py"], "cwd": "candidates/c1/workspace", "timeout_seconds": 30},
                "smoke": {"argv": ["{python}", "smoke.py"], "cwd": "candidates/c1/workspace", "timeout_seconds": 30}
            },
            "smoke_result_path": "candidates/c1/smoke.json",
            "development_budget": {"max_technical_attempts": 3}
        }
        missing_dependency = dict(candidate_spec)
        missing_dependency.pop("dependency_contract")
        missing_dependency_path = run / "candidates" / "c1" / "missing-dependency.json"
        write(missing_dependency_path, missing_dependency)
        run_cli(run, "add", "--spec", str(missing_dependency_path), expected=1)
        write(spec_path, candidate_spec)
        run_cli(
            run, "init", "--min-candidates", "1", "--max-candidates", "2",
            "--min-families", "1", "--max-technical-attempts", "3",
            "--max-candidate-wall-clock-minutes", "5", "--max-total-wall-clock-minutes", "10",
            "--promotion-threshold-percent", "1.0",
        )
        opportunity_cli(
            run, "init", "--min-opportunities", "1", "--max-opportunities", "2",
            "--min-rewrite-families", "1", "--min-candidate-opportunities", "1",
        )
        opportunity_spec = run / "models" / "opportunity-spec.json"
        baseline_hash = hashlib.sha256((run / "models/baseline.json").read_bytes()).hexdigest()
        write(opportunity_spec, {
            "opportunity_id": "fuse-transfer",
            "name": "fuse materialized transfer",
            "model_scope": "DECOMPOSITION_CONDITIONAL",
            "source_model_term": "raw_o write plus read",
            "affected_stages": ["S3", "post"],
            "current_contribution_us": 4.0,
            "optimistic_gain_ceiling_us": 3.0,
            "likely_gain_interval_us": {"lower": 1.0, "upper": 2.5},
            "confidence": "HIGH",
            "rewrite_families": ["layout-redesign", "persistent-grid"],
            "implementation_budget_minutes": 10,
            "hypothesis": "fusion removes a materialized global-memory boundary",
            "derivation": "stage timing minus the mandatory semantic output store",
            "evidence": [{"path": "models/baseline.json", "sha256": baseline_hash, "claim": "current objective contribution"}],
        })
        opportunity_cli(run, "add", "--spec", str(opportunity_spec))
        opportunity_cli(run, "rank")
        run_cli(run, "add", "--spec", str(spec_path))

        # Candidate execution must fail closed if ranked evidence is edited
        # after registration. Reranking repairs the derived score and digest.
        opportunity_map_path = run / "models/opportunity_map.json"
        opportunity_map = json.loads(opportunity_map_path.read_text(encoding="utf-8"))
        opportunity_map["opportunities"][0]["priority_score"] = 999.0
        write(opportunity_map_path, opportunity_map)
        run_cli(run, "run", "--candidate-id", "c1", expected=1)
        pool = json.loads((run / "models/candidate_pool.json").read_text(encoding="utf-8"))
        assert pool["candidates"][0]["attempts"] == []
        opportunity_cli(run, "rank")

        failed = run_cli(run, "run", "--candidate-id", "c1")
        assert failed.get("status") == "DEVELOPING", failed
        pool = json.loads((run / "models/candidate_pool.json").read_text(encoding="utf-8"))
        item = pool["candidates"][0]
        assert item["attempts"][0]["status"] == "TECHNICAL_FAILURE"
        assert item["status"] != "REJECTED"
        repair = discovery_action(run, ROOT / "scripts")
        assert repair and repair["action"] == "REPAIR_DISCOVERY_CANDIDATE"

        write(workspace / "ready.flag", "ready\n")
        screened = run_cli(run, "run", "--candidate-id", "c1")
        assert screened["status"] == "QUALIFICATION_READY"
        assert abs(screened["improvement_percent"] - 20.0) < 1e-9
        opportunity_map = json.loads((run / "models/opportunity_map.json").read_text(encoding="utf-8"))
        observation = opportunity_map["opportunities"][0]["observations"][0]
        assert observation["observed_global_gain_us"] == 2.0
        assert observation["residual_us"] == 0.0
        pool = json.loads((run / "models/candidate_pool.json").read_text(encoding="utf-8"))
        pool["candidates"].append({
            "candidate_id": "c2",
            "opportunity_id": "fuse-transfer",
            "family": "persistent-grid",
            "status": "QUALIFICATION_READY",
            "screening": {"improvement_percent": 30.0},
        })
        write(run / "models/candidate_pool.json", pool)
        strongest = discovery_action(run, ROOT / "scripts")
        assert strongest and strongest["action"] == "PROMOTE_DISCOVERY_CANDIDATE"
        assert strongest["commands"][0][-1] == "c2"
        pool["candidates"].pop()
        write(run / "models/candidate_pool.json", pool)

        # Direct promotion is also evidence-gated, even if the candidate was
        # screened while the opportunity map was valid.
        opportunity_map = json.loads(opportunity_map_path.read_text(encoding="utf-8"))
        opportunity_map["opportunities"][0]["priority_score"] = 999.0
        write(opportunity_map_path, opportunity_map)
        run_cli(run, "promote", "--candidate-id", "c1", expected=1)
        opportunity_cli(run, "rank")

        promoted = run_cli(run, "promote", "--candidate-id", "c1")
        assert promoted["status"] == "PROMOTED_TO_QUALIFICATION"
        qualification = discovery_action(run, ROOT / "scripts")
        assert qualification and qualification["action"] == "BUILD_QUALIFICATION_CONTRACT"
        promotion_path = run / promoted["promotion"]["path"]
        promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
        assert "production acceptance" in promotion["claims_forbidden"]
        pool = json.loads((run / "models/candidate_pool.json").read_text(encoding="utf-8"))
        pool["status"] = "PAUSED"
        write(run / "models/candidate_pool.json", pool)
        paused = discovery_action(run, ROOT / "scripts")
        assert paused and paused["action"] == "DISCOVERY_BUDGET_REVIEW"
    print("candidate discovery loop test: PASS")


if __name__ == "__main__":
    main()
