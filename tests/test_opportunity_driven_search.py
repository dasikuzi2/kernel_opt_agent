#!/usr/bin/env python3
"""Exercise opportunity ranking, claim bounds, and implementation-first routing."""

from __future__ import annotations

import json
import hashlib
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


def cli(command: str, run: Path, *args: str, expected: int = 0) -> dict:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/kernel_opt.py"), command, *args, "--run", str(run)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert completed.returncode == expected, (completed.stdout, completed.stderr)
    return json.loads(completed.stdout) if completed.stdout.strip() else {}


def opportunity_spec(identifier: str, family: str, gain: tuple[float, float], ceiling: float, cost: float, evidence_sha256: str) -> dict:
    return {
        "opportunity_id": identifier,
        "name": identifier,
        "model_scope": "CURRENT_SCHEDULE",
        "source_model_term": f"term-{identifier}",
        "affected_stages": [identifier],
        "current_contribution_us": ceiling + 1,
        "optimistic_gain_ceiling_us": ceiling,
        "likely_gain_interval_us": {"lower": gain[0], "upper": gain[1]},
        "confidence": "HIGH",
        "rewrite_families": [family],
        "implementation_budget_minutes": cost,
        "hypothesis": "remove globally visible scheduled work",
        "derivation": "measured stage contribution times removable work fraction",
        "evidence": [{"path": "models/baseline.json", "sha256": evidence_sha256, "claim": "current objective contribution"}],
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary) / "run"
        (run / "models").mkdir(parents=True)
        for name in ("operator.json", "workload.json", "hardware.json"):
            shutil.copyfile(ROOT / "tests" / "fixtures" / name, run / name)
        write(run / "models" / "baseline.json", {"status": "VALID", "correctness": {"status": "PASS"}})
        evidence_sha256 = hashlib.sha256((run / "models/baseline.json").read_bytes()).hexdigest()
        cli("candidate", run, "init", "--min-candidates", "2", "--max-candidates", "4", "--min-families", "2")
        first = discovery_action(run, ROOT / "scripts")
        assert first and first["action"] == "BUILD_OPPORTUNITY_MAP", first
        bound_path = run / "models" / "optimistic-bound.json"
        write(bound_path, {"maximum_speedup": 1.1, "target_speedup": 2.0})
        write(run / "models" / "feasibility_gate.json", {
            "schema_version": "optimization-feasibility-gate-v1",
            "status": "VALID",
            "decision": "TARGET_INFEASIBLE",
            "target": {"metric": "gpu_active_us", "speedup": 2.0},
            "bound": {"maximum_speedup": 1.1, "optimistic_latency_floor_ms": 1.0},
            "evidence": [{"path": "models/optimistic-bound.json", "sha256": hashlib.sha256(bound_path.read_bytes()).hexdigest()}],
            "required_reframe_options": ["relax the target or contract"],
        })
        infeasible = discovery_action(run, ROOT / "scripts")
        assert infeasible and infeasible["action"] == "STOP_OR_REFRAME_INFEASIBLE_TARGET", infeasible
        authorized_gate = json.loads((run / "models" / "feasibility_gate.json").read_text())
        authorized_gate["residual_search_policy"] = {
            "status": "AUTHORIZED",
            "objective_mode": "BEST_FEASIBLE_WITHIN_FROZEN_CONTRACT",
            "minimum_likely_gain_us": 1.0,
            "max_total_wall_clock_minutes": 60.0,
            "stop_conditions": ["no ranked opportunity exceeds the material-gain floor"],
            "authorization": "test fixture explicitly authorizes bounded residual search",
        }
        write(run / "models" / "feasibility_gate.json", authorized_gate)
        residual = discovery_action(run, ROOT / "scripts")
        assert residual and residual["action"] == "BUILD_OPPORTUNITY_MAP", residual
        malformed_gate = json.loads((run / "models" / "feasibility_gate.json").read_text())
        malformed_gate["residual_search_policy"]["minimum_likely_gain_us"] = 0
        write(run / "models" / "feasibility_gate.json", malformed_gate)
        malformed = discovery_action(run, ROOT / "scripts")
        assert malformed and malformed["action"] == "BLOCK_INVALID_FEASIBILITY_GATE", malformed
        invalid_gate = json.loads((run / "models" / "feasibility_gate.json").read_text())
        invalid_gate.pop("residual_search_policy")
        invalid_gate["bound"]["maximum_speedup"] = 2.1
        write(run / "models" / "feasibility_gate.json", invalid_gate)
        blocked_gate = discovery_action(run, ROOT / "scripts")
        assert blocked_gate and blocked_gate["action"] == "BLOCK_INVALID_FEASIBILITY_GATE", blocked_gate
        (run / "models" / "feasibility_gate.json").unlink()
        bound_path.unlink()
        cli(
            "opportunity", run, "init", "--min-opportunities", "2", "--max-opportunities", "4",
            "--min-rewrite-families", "2", "--min-candidate-opportunities", "2",
        )
        specs = [
            opportunity_spec("large-cheap", "cross-stage-fusion", (3.0, 5.0), 6.0, 10.0, evidence_sha256),
            opportunity_spec("small-expensive", "tile-retune", (1.0, 2.0), 3.0, 30.0, evidence_sha256),
        ]
        for spec in specs:
            path = run / "models" / f"{spec['opportunity_id']}.json"
            write(path, spec)
            cli("opportunity", run, "add", "--spec", str(path))
        before_rank = discovery_action(run, ROOT / "scripts")
        assert before_rank and before_rank["action"] == "RANK_OPPORTUNITIES", before_rank
        ranked = cli("opportunity", run, "rank")
        assert ranked["opportunities"][0]["opportunity_id"] == "large-cheap", ranked
        closure_path = run / "models" / "large-cheap-closure.json"
        closure_payload = {
            "result": "candidate family is below the global materiality floor",
            "measured_speedup": 0.999,
        }
        write(closure_path, closure_payload)
        closed = cli(
            "opportunity",
            run,
            "close",
            "--opportunity-id",
            "large-cheap",
            "--disposition",
            "BELOW_MATERIALITY_FLOOR",
            "--reason",
            "Measured whole-workload gain cannot clear the frozen promotion threshold.",
            "--evidence",
            str(closure_path),
            "--evidence-claim",
            "paired whole-workload screen and materiality decision",
            "--reopen-condition",
            "the workload or measured promotion threshold changes",
        )
        assert closed["status"] == "CLOSED" and closed["priority_score"] == 0
        closed_routing = discovery_action(run, ROOT / "scripts")
        assert closed_routing and closed_routing["action"] == "RETRIEVE_OPTIMIZATION_METHODS"
        assert (
            closed_routing["blocking_inputs"][0]["opportunity_id"]
            == "small-expensive"
        ), closed_routing
        closed_methods = cli("method", run, "recommend")
        assert [
            item["opportunity_id"] for item in closed_methods["recommendations"]
        ] == ["small-expensive"], closed_methods
        cli(
            "opportunity",
            run,
            "close",
            "--opportunity-id",
            "small-expensive",
            "--disposition",
            "MEASURED_REJECT",
            "--reason",
            "The paired measurement rejects the remaining active implementation family.",
            "--evidence",
            str(closure_path),
            "--evidence-claim",
            "paired rejection evidence",
            "--reopen-condition",
            "an independent architecture family changes the measured bound",
        )
        all_closed = discovery_action(run, ROOT / "scripts")
        assert all_closed and all_closed["action"] == "OPPORTUNITY_PORTFOLIO_CLOSED"
        cli(
            "opportunity",
            run,
            "reopen",
            "--opportunity-id",
            "small-expensive",
            "--reason",
            "The test introduces an independent architecture family.",
        )
        write(closure_path, {**closure_payload, "measured_speedup": 1.5})
        invalid_closure = discovery_action(run, ROOT / "scripts")
        assert invalid_closure and invalid_closure["action"] == "BLOCK_INVALID_OPPORTUNITY_MAP"
        write(closure_path, closure_payload)
        reopened = cli(
            "opportunity",
            run,
            "reopen",
            "--opportunity-id",
            "large-cheap",
            "--reason",
            "The test changes the workload contract and intentionally resumes search.",
        )
        assert reopened["status"] == "UNIMPLEMENTED" and "closure" not in reopened
        residual_bound_path = run / "models" / "residual-bound.json"
        write(residual_bound_path, {"maximum_speedup": 1.1, "target_speedup": 2.0})
        write(run / "models" / "feasibility_gate.json", {
            "schema_version": "optimization-feasibility-gate-v1",
            "status": "VALID",
            "decision": "TARGET_INFEASIBLE",
            "target": {"metric": "gpu_active_us", "speedup": 2.0},
            "bound": {"maximum_speedup": 1.1, "optimistic_latency_floor_ms": 1.0},
            "evidence": [{
                "path": "models/residual-bound.json",
                "sha256": hashlib.sha256(residual_bound_path.read_bytes()).hexdigest(),
            }],
            "required_reframe_options": ["relax the target or contract"],
            "residual_search_policy": {
                "status": "AUTHORIZED",
                "objective_mode": "BEST_FEASIBLE_WITHIN_FROZEN_CONTRACT",
                "minimum_likely_gain_us": 10.0,
                "max_total_wall_clock_minutes": 60.0,
                "stop_conditions": ["no opportunity clears the gain floor"],
                "authorization": "test fixture authorizes bounded residual search",
            },
        })
        materiality_stop = discovery_action(run, ROOT / "scripts")
        assert materiality_stop and materiality_stop["action"] == "STOP_RESIDUAL_SEARCH_AT_MATERIALITY_FLOOR", materiality_stop
        timed_gate = json.loads((run / "models" / "feasibility_gate.json").read_text())
        timed_gate["residual_search_policy"]["minimum_likely_gain_us"] = 1.0
        write(run / "models" / "feasibility_gate.json", timed_gate)
        timed_pool = json.loads((run / "models" / "candidate_pool.json").read_text())
        timed_pool["discovery_started_at"] = "2000-01-01T00:00:00+00:00"
        write(run / "models" / "candidate_pool.json", timed_pool)
        time_stop = discovery_action(run, ROOT / "scripts")
        assert time_stop and time_stop["action"] == "STOP_RESIDUAL_SEARCH_AT_TIME_BUDGET", time_stop
        timed_pool["discovery_started_at"] = None
        write(run / "models" / "candidate_pool.json", timed_pool)
        (run / "models" / "feasibility_gate.json").unlink()
        residual_bound_path.unlink()
        map_path = run / "models" / "opportunity_map.json"
        tampered = json.loads(map_path.read_text(encoding="utf-8"))
        tampered["opportunities"][0]["priority_score"] = 999.0
        write(map_path, tampered)
        blocked = discovery_action(run, ROOT / "scripts")
        assert blocked and blocked["action"] == "BLOCK_INVALID_OPPORTUNITY_MAP", blocked
        cli("opportunity", run, "rank")
        retrieve = discovery_action(run, ROOT / "scripts")
        assert retrieve and retrieve["action"] == "RETRIEVE_OPTIMIZATION_METHODS", retrieve
        cli("method", run, "recommend")
        implement = discovery_action(run, ROOT / "scripts")
        assert implement and implement["action"] == "EXPAND_DISCOVERY_PORTFOLIO", implement
        target = implement["blocking_inputs"][0]["next_ranked_uncovered_opportunity"]
        assert target["opportunity_id"] == "large-cheap", target
        pool_path = run / "models" / "candidate_pool.json"
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        pool["candidates"] = [
            {"candidate_id": "low-first", "opportunity_id": "small-expensive", "family": "tile-retune", "status": "PROPOSED"},
            {"candidate_id": "high-second", "opportunity_id": "large-cheap", "family": "cross-stage-fusion", "status": "PROPOSED"},
        ]
        write(pool_path, pool)
        ranked_implementation = discovery_action(run, ROOT / "scripts")
        assert ranked_implementation and ranked_implementation["action"] == "IMPLEMENT_DISCOVERY_CANDIDATE"
        assert ranked_implementation["blocking_inputs"][0]["candidate_id"] == "high-second", ranked_implementation

        invalid = opportunity_spec("false-proof", "bad-family", (1.0, 2.0), 2.0, 10.0, evidence_sha256)
        invalid["model_scope"] = "ABSOLUTE_GLOBAL_OPTIMUM"
        invalid_path = run / "models" / "invalid.json"
        write(invalid_path, invalid)
        failed = cli("opportunity", run, "add", "--spec", str(invalid_path), expected=1)
        assert not failed

        candidate_root = run / "candidates" / "bad"
        candidate_root.mkdir(parents=True)
        source = candidate_root / "kernel.py"
        source.write_text("pass\n", encoding="utf-8")
        candidate_spec = {
            "candidate_id": "bad", "opportunity_id": "missing", "name": "bad",
            "family": "cross-stage-fusion", "change_axes": ["fusion"], "hypothesis": "test",
            "expected_global_effect": "test", "predicted_global_gain_us": {"lower": 1, "upper": 2},
            "source_paths": ["candidates/bad/kernel.py"],
            "commands": {stage: {"argv": [sys.executable, "-c", "pass"], "cwd": "candidates/bad", "timeout_seconds": 2} for stage in ("build", "correctness", "smoke")},
            "smoke_result_path": "candidates/bad/smoke.json",
        }
        candidate_path = candidate_root / "spec.json"
        write(candidate_path, candidate_spec)
        failed_candidate = cli("candidate", run, "add", "--spec", str(candidate_path), expected=1)
        assert not failed_candidate
    print("opportunity-driven search test: PASS")


if __name__ == "__main__":
    main()
