#!/usr/bin/env python3
"""Dependency-free repository behavior tests."""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from archive_final_binary_sass import compatible_architecture_codes
from count_sass import classify_mnemonic
from advance_run import validate_resource_balance


def run(command, expected=0):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == expected, (command, result.stdout, result.stderr)
    return result


def resource_row(resource_id, resource_kind, state="UNKNOWN", request_ids=None):
    request_ids = list(request_ids or [])
    measured = state != "UNKNOWN"
    row = {
        "resource_id": resource_id,
        "resource_kind": resource_kind,
        "material": True,
        "mandatory_work": {"value": 1024, "unit": "bytes"},
        "actual_work": {"value": 1024, "unit": "bytes"},
        "production_point": {"status": state, "value": 1.0, "unit": "work/us"},
        "matched_saturation": {"status": state, "value": 2.0 if measured else None, "unit": "work/us", "conditions": ["matched fixture"]},
        "utilization": {
            "status": state,
            "value_percent": 50.0 if measured else None,
            "numerator": {"value": 1.0, "unit": "work/us"} if measured else None,
            "denominator": {"value": 2.0, "unit": "work/us"} if measured else None,
            "time_window": "kernel active",
            "boundary": resource_kind,
        },
        "critical_path": {"status": state, "contribution_us": 1.0 if measured else None, "coupling_model": "A/B/AB", "probability": 0.8},
        "non_saturation_causes": ["RESIDENCY_LIMIT"],
        "evidence": ["fixture measurement"] if measured else [],
        "unresolved_request_ids": request_ids,
    }
    if resource_kind == "TENSOR_CORE":
        row["compute_efficiency"] = {
            "device_coverage": 0.75 if measured else None,
            "eligible_time_fraction": 0.5 if measured else None,
            "eligible_window_issue_efficiency": 0.8 if measured else None,
            "composition_status": "MEASURED" if measured else "UNKNOWN",
        }
    return row


def schedule_point(schedule_id, decision):
    return {
        "schedule_id": schedule_id,
        "correctness": "PASS" if decision != "PENDING" else "PENDING",
        "valid_compute": {"operations": 1024},
        "padded_compute": {"operations": 0},
        "bytes_by_boundary": {"request": 1024, "L2": 1024, "device_memory": 1024},
        "allocation": {"registers_per_thread": 8, "shared_bytes": 0},
        "device_coverage": {"fraction": 1.0},
        "synchronization": {"barriers": 0},
        "predicted_dag_us": 9.5 if decision != "PENDING" else None,
        "measured_us": 10.0 if decision != "PENDING" else None,
        "uncertainty": {"us": 0.5},
        "decision": decision,
    }


def experiment_request(status="PROPOSED"):
    return {
        "request_id": "req-matched-service",
        "status": status,
        "issued_by_role": "GLOBAL_SCHEDULER",
        "workload_cases": ["n262144", "n1048576"],
        "model_field": "resource_balance.*.utilization",
        "candidate_decision": "choose current or candidate schedule",
        "causal_question": "What is the matched request-service saturation point?",
        "resource_ids": ["lsu", "memory"],
        "affected_stage_ids": ["kernel"],
        "priority": 0,
        "sensitivity": {
            "max_weighted_benefit_us": 2.0,
            "critical_path_probability": 0.8,
            "uncertainty": "high",
            "experiment_cost": "low",
        },
        "controls": ["zero work", "dependent", "independent"],
        "measurement_contract": {"timer": "GPU events", "cache": "declared", "geometry": "matched"},
        "expected_sass": ["LDG", "STG"],
        "catalog_resolution": {
            "catalog_queried": True,
            "query": {"resource": "request service"},
            "decision": "CREATE_RUN_LOCAL",
            "package_id": None,
            "reason": "no production-predictive package",
        },
        "result_binding": {"status": "BOUND" if status == "RESOLVED" else "PENDING", "evidence": ["fixture"] if status == "RESOLVED" else []},
        "promotion_disposition": {"status": "RUN_LOCAL", "reason": "fixture remains application-shaped"},
    }


def main():
    assert compatible_architecture_codes("12.0") == {"sm_120", "sm_120a"}
    assert "sm_121" not in compatible_architecture_codes("12.0")
    unknown_tensor = resource_row("tensor", "TENSOR_CORE", request_ids=["req-matched-service"])
    tensor_balance = {
        "schema_version": "resource-balance-ledger-v1",
        "cases": [{
            "case_id": "tensor-case",
            "resource_rows": [unknown_tensor],
            "device_coverage": {"status": "UNKNOWN"},
            "critical_path": {"status": "UNKNOWN"},
            "model_residual": {"status": "UNKNOWN"},
        }],
    }
    tensor_workload = {"cases": [{"id": "tensor-case"}]}
    tensor_queue = {"requests": [experiment_request()]}
    tensor_errors = []
    validate_resource_balance(tensor_balance, tensor_workload, {"tensor"}, tensor_queue, tensor_errors)
    assert not tensor_errors, tensor_errors
    unknown_tensor["compute_efficiency"]["device_coverage"] = 0.0
    tensor_errors = []
    validate_resource_balance(tensor_balance, tensor_workload, {"tensor"}, tensor_queue, tensor_errors)
    assert any("must keep compute_efficiency.device_coverage null" in error for error in tensor_errors)
    expected_sass_classes = {
        "ELECT": "warp_collective",
        "ENDCOLLECTIVE": "warp_collective",
        "F2FP.BF16.F32.PACK_AB": "conversion",
        "FCHK": "predicate_compare",
        "FENCE.VIEW.ASYNC.S": "memory_wait",
        "FHADD.BF16": "simt_fp",
        "FHFMA.BF16": "simt_fp",
        "HSETP2.BF16_V2.GE.AND": "predicate_compare",
        "IABS": "integer_address",
        "IADD3": "integer_address",
        "IMNMX.S64": "integer_address",
        "NANOSLEEP.SYNCS": "memory_wait",
        "P2R": "move_select",
        "PLOP3.LUT": "predicate_compare",
        "R2UR.BROADCAST": "move_select",
        "SYNCS.ARRIVE.TRANS64": "barrier",
        "SYNCS.EXCH.64": "barrier",
        "SYNCS.PHASECHK.TRANS64.TRYWAIT": "memory_wait",
        "UIADD3.64": "integer_address",
        "UTMACCTL.PF": "async_copy",
        "UTMALDG.5D": "async_copy",
        "VIMNMX.U32": "integer_address",
        "UVIMNMX.U32": "integer_address",
    }
    for mnemonic, expected_class in expected_sass_classes.items():
        assert classify_mnemonic(mnemonic) == [expected_class]

    # The ranking receipt must persist the resource-ledger probability used by
    # its own score; otherwise the evidence-closed gate cannot reproduce it.
    with tempfile.TemporaryDirectory() as temporary:
        ranking_run = Path(temporary)
        (ranking_run / "models").mkdir()
        (ranking_run / "traces").mkdir()
        (ranking_run / "workload.json").write_text(json.dumps({
            "cases": [{"id": "case", "weight": 1.0}],
        }))
        (ranking_run / "models/resource_balance.json").write_text(json.dumps({
            "cases": [{
                "case_id": "case",
                "critical_path": {"total_us": 10.0, "stage_gpu_active_us": {"kernel": 10.0}},
                "resource_rows": [{
                    "resource_id": "tensor",
                    "matched_saturation": {"status": "UNKNOWN"},
                    "utilization": {"status": "UNKNOWN"},
                    "critical_path": {"status": "UNKNOWN", "probability": 0.8},
                }],
            }],
        }))
        (ranking_run / "models/experiment_queue.json").write_text(json.dumps({
            "requests": [{
                "request_id": "req",
                "resource_ids": ["tensor"],
                "affected_stage_ids": ["kernel"],
                "workload_cases": ["case"],
                "sensitivity": {"experiment_cost": "LOW"},
            }],
        }))
        run([sys.executable, str(ROOT / "scripts/rank_experiments.py"), "--run", str(ranking_run)])
        ranked = json.loads((ranking_run / "models/experiment_queue.json").read_text())["requests"][0]["sensitivity"]
        assert ranked["critical_path_probability"] == 0.8
        assert ranked["ranking_score"] == 8.0

    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text())

    cli_help = run([sys.executable, str(ROOT / "scripts/kernel_opt.py"), "--help"])
    assert "new-run" in cli_help.stdout and "experiment-execute" in cli_help.stdout
    with tempfile.TemporaryDirectory() as temporary:
        failed_cli = run([
            sys.executable, str(ROOT / "scripts/kernel_opt.py"), "candidate", "status",
            "--run", str(Path(temporary) / "missing-run"),
        ], expected=1)
        assert "candidate pool is missing" in failed_cli.stderr

    report_fixture = ROOT / "tests/fixtures/human_review_report.json"
    run([
        sys.executable,
        str(ROOT / "scripts/validate_human_review_report.py"),
        str(report_fixture),
    ])
    with tempfile.TemporaryDirectory() as temporary:
        rendered = Path(temporary) / "review.html"
        run([
            sys.executable,
            str(ROOT / "scripts/render_human_review_report.py"),
            str(report_fixture),
            str(rendered),
        ])
        assert rendered.exists()
        text = rendered.read_text()
        assert "示例算子人工审核报告" in text
        assert "__REPORT_DATA__" not in text

    missing = run([sys.executable, str(ROOT / "scripts/new_run.py")], expected=2)
    for phrase in ("Operator computation", "Target workload", "Target hardware"):
        assert phrase in missing.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        invalid_operator = json.loads((ROOT / "tests/fixtures/operator.json").read_text())
        invalid_operator["inputs"][0]["shape"] = "[invalid-string-shape]"
        invalid_operator_path = root / "invalid-operator.json"
        invalid_operator_path.write_text(json.dumps(invalid_operator))
        invalid = run([
            sys.executable, str(ROOT / "scripts/new_run.py"),
            "--root", str(root), "--run-id", "invalid-intake",
            "--operator", str(invalid_operator_path),
            "--workload", str(ROOT / "tests/fixtures/workload.json"),
            "--hardware", str(ROOT / "tests/fixtures/hardware.json"),
            "--test-legacy-contract",
        ], expected=1)
        assert "intake schema validation failed" in invalid.stderr
        assert not (root / "runs/invalid-intake").exists()
        invalid_operator_path.unlink()

        complete = run([
            sys.executable, str(ROOT / "scripts/new_run.py"),
            "--root", str(root), "--run-id", "self-test",
            "--operator", str(ROOT / "tests/fixtures/operator.json"),
            "--workload", str(ROOT / "tests/fixtures/workload.json"),
            "--hardware", str(ROOT / "tests/fixtures/hardware.json"),
            "--test-legacy-contract",
        ])
        result = json.loads(complete.stdout)
        assert result["status"] == "READY"
        assert result["current_phase"] == "PLANNING"
        assert (root / "runs/self-test/00_intake.json").exists()
        assert (root / "runs/self-test/run_state.json").exists()
        assert (root / "runs/self-test/models/work_ledger.json").exists()
        assert (root / "runs/self-test/models/dag.json").exists()
        assert (root / "runs/self-test/models/optimization_plan.json").exists()
        assert (root / "runs/self-test/models/microarchitecture_model.json").exists()
        assert (root / "runs/self-test/models/baseline.json").exists()
        assert (root / "runs/self-test/models/microbenchmark_plan.json").exists()
        assert (root / "runs/self-test/models/schedule_model.json").exists()
        assert (root / "runs/self-test/models/global_schedule_state.json").exists()
        assert (root / "runs/self-test/models/resource_balance.json").exists()
        assert (root / "runs/self-test/models/tradeoff_frontier.json").exists()
        assert (root / "runs/self-test/models/experiment_queue.json").exists()
        assert (root / "runs/self-test/models/model_validation.json").exists()
        assert (root / "runs/self-test/models/production_validation.json").exists()
        assert (root / "runs/self-test/static/instruction_audit.json").exists()
        assert (root / "runs/self-test/microbench_candidates").is_dir()
        assert not (root / "runs/self-test/models/hypothesis.json").exists()
        assert not (root / "runs/self-test/evidence_index.json").exists()

        skipped = run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(root / "runs/self-test"), "--to", "MODELING", "--check-only",
        ], expected=1)
        assert "sequential" in skipped.stdout
        unready = run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(root / "runs/self-test"), "--to", "BASELINE", "--check-only",
        ], expected=1)
        assert "optimization_plan.status" in unready.stdout

        plan_path = root / "runs/self-test/models/optimization_plan.json"
        plan = json.loads(plan_path.read_text())
        plan.update({
            "status": "EXECUTABLE",
            "objective": "minimize weighted GPU-active latency",
            "global_scheduler_owner": "fixture-global-scheduler",
            "workload_priorities": [
                {"case_id": "n262144", "weight": 0.25},
                {"case_id": "n1048576", "weight": 0.75},
            ],
            "experiment_queue": [{"id": "measurement-calibration", "depends_on": []}],
            "correctness_gates": ["reference agreement"],
            "evidence_gates": ["final binary and raw samples"],
            "acceptance_rule": "paired confidence interval improves the weighted objective",
            "stop_criteria": ["bound gap below tolerance"],
            "revision_history": [{"revision": 1, "reason": "initial plan"}],
            "model_error_tolerances_percent": {
                "p1_p2_to_p3": 10.0,
                "schedule_to_p4": 10.0,
                "achieved_to_feasible_bound": 5.0,
            },
        })
        plan_path.write_text(json.dumps(plan))
        architecture_path = root / "runs/self-test/models/microarchitecture_model.json"
        architecture = json.loads(architecture_path.read_text())
        architecture["target_identity"] = {"device": "synthetic-device"}
        architecture["scope"] = ["SIMT", "LSU", "memory hierarchy"]
        architecture_path.write_text(json.dumps(architecture))
        microbench_path = root / "runs/self-test/models/microbenchmark_plan.json"
        microbench = json.loads(microbench_path.read_text())
        microbench["target_questions"] = ["matched load/store service"]
        microbench_path.write_text(json.dumps(microbench))
        missing_global = run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(root / "runs/self-test"), "--to", "BASELINE", "--check-only",
        ], expected=1)
        assert "global_schedule_state.status must be PLANNED" in missing_global.stdout

        global_path = root / "runs/self-test/models/global_schedule_state.json"
        global_state = json.loads(global_path.read_text())
        global_state.update({
            "status": "PLANNED",
            "owner": {
                "role": "GLOBAL_SCHEDULER",
                "owner_id": "fixture-global-scheduler",
                "exclusive_authority": [
                    "RANK_EXPERIMENTS", "CLOSE_RESOURCE_MODEL",
                    "ACCEPT_GLOBAL_CANDIDATE", "AUTHORIZE_LIMIT_REPORT",
                ],
            },
            "material_resources": ["lsu", "memory"],
            "stage_assignments": [{"scope": "fixture-stage", "owner": "fixture-worker"}],
            "decision_policy": {
                "objective": "minimize weighted GPU-active latency",
                "ranking": "benefit and uncertainty",
                "local_speedup_is_not_global_acceptance": True,
            },
            "revision_history": [{"revision": 1, "reason": "initial global plan"}],
        })
        global_path.write_text(json.dumps(global_state))
        queue_path = root / "runs/self-test/models/experiment_queue.json"
        queue = json.loads(queue_path.read_text())
        queue.update({
            "status": "EXECUTABLE",
            "catalog_snapshot": {"sha256": "4" * 64},
            "requests": [experiment_request()],
        })
        queue_path.write_text(json.dumps(queue))
        balance_path = root / "runs/self-test/models/resource_balance.json"
        balance = json.loads(balance_path.read_text())
        balance.update({
            "status": "INITIALIZED",
            "cases": [
                {
                    "case_id": case_id,
                    "resource_rows": [
                        resource_row("lsu", "REQUEST_SERVICE", request_ids=["req-matched-service"]),
                        resource_row("memory", "DEVICE_MEMORY", request_ids=["req-matched-service"]),
                    ],
                    "device_coverage": {"status": "UNKNOWN"},
                    "critical_path": {"status": "UNKNOWN"},
                    "model_residual": {"status": "UNKNOWN"},
                }
                for case_id in ("n262144", "n1048576")
            ],
        })
        balance_path.write_text(json.dumps(balance))
        frontier_path = root / "runs/self-test/models/tradeoff_frontier.json"
        frontier = json.loads(frontier_path.read_text())
        frontier.update({
            "status": "INITIALIZED",
            "objective": {"metric": "weighted_gpu_active_us"},
            "cases": [
                {
                    "case_id": case_id,
                    "legal_minimum": {"bytes_by_boundary": {"request": 1024}},
                    "current_schedule": schedule_point("current", "PENDING"),
                    "candidates": [],
                    "pareto_frontier": [],
                }
                for case_id in ("n262144", "n1048576")
            ],
        })
        frontier_path.write_text(json.dumps(frontier))
        ready = run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(root / "runs/self-test"), "--to", "BASELINE", "--check-only",
        ])
        assert json.loads(ready.stdout)["status"] == "PASS"
        advanced = run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(root / "runs/self-test"), "--to", "BASELINE",
        ])
        assert json.loads(advanced.stdout)["advanced"] is True
        assert json.loads((root / "runs/self-test/run_state.json").read_text())["current_phase"] == "BASELINE"

        run_dir = root / "runs/self-test"
        baseline_path = run_dir / "models/baseline.json"
        baseline = json.loads(baseline_path.read_text())
        baseline.update({
            "status": "VALID",
            "correctness": {"status": "PASS", "evidence": ["raw/reference.json"]},
            "source_identities": [{"path": "source.cu", "sha256": "1" * 64}],
            "measurement_methods": {
                "cpu_dispatch": "host monotonic clock",
                "gpu_active": "CUDA events",
                "end_to_end": "host monotonic clock with synchronization",
            },
            "environment_controls": {
                "competing_load": "idle gate",
                "clock_power_policy": "recorded",
                "thermal_policy": "recorded",
                "warmup_and_cold_start": "separated",
            },
            "cases": [
                {
                    "case_id": case_id,
                    "correctness": "PASS",
                    "source_identity": {"sha256": "1" * 64},
                    "raw_samples": {"path": f"raw/{case_id}.json", "sha256": "2" * 64},
                    "cpu_dispatch": {"median_us": 1.0},
                    "gpu_active": {"median_us": 10.0},
                    "end_to_end": {"median_us": 11.0},
                }
                for case_id in ("n262144", "n1048576")
            ],
        })
        baseline_path.write_text(json.dumps(baseline))
        run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(run_dir), "--to", "MODELING",
        ])

        architecture = json.loads(architecture_path.read_text())
        architecture.update({
            "status": "INITIALIZED",
            "resource_nodes": [{"id": "lsu"}, {"id": "memory"}],
            "workload_mappings": [{"case_id": "n262144"}, {"case_id": "n1048576"}],
            "evidence": [{"kind": "device-query"}],
        })
        architecture_path.write_text(json.dumps(architecture))
        ledger_path = run_dir / "models/work_ledger.json"
        ledger = json.loads(ledger_path.read_text())
        ledger["cases"] = [
            {
                "case_id": case_id,
                "valid_work": {"bytes": 1024},
                "padded_or_redundant_work": {"bytes": 0},
                "assumptions": ["contiguous"],
                "evidence": ["operator contract"],
            }
            for case_id in ("n262144", "n1048576")
        ]
        ledger_path.write_text(json.dumps(ledger))
        dag_path = run_dir / "models/dag.json"
        dag = json.loads(dag_path.read_text())
        dag["nodes"] = [{"id": "load"}, {"id": "compute"}, {"id": "store"}]
        dag["mathematical_edges"] = [["load", "compute"], ["compute", "store"]]
        dag["critical_paths"] = [["load", "compute", "store"]]
        dag_path.write_text(json.dumps(dag))
        schedule_path = run_dir / "models/schedule_model.json"
        schedule = json.loads(schedule_path.read_text())
        schedule.update({
            "status": "INITIALIZED",
            "binary_identity": {"sha256": "3" * 64},
            "sass_control_flow": [{"block": 0}],
            "dynamic_instruction_method": "exact loop trip counts",
            "resource_mapping": [{"instruction": "LDG", "resource": "lsu"}],
            "workload_cases": [{"case_id": "n262144"}, {"case_id": "n1048576"}],
            "evidence": [{"kind": "SASS"}],
        })
        schedule_path.write_text(json.dumps(schedule))
        microbench = json.loads(microbench_path.read_text())
        microbench.update({
            "status": "EXECUTABLE",
            "coupling_tests": [{"a": "lsu", "b": "simt", "controls": ["A", "B", "AB"]}],
            "cross_layer_prediction_gates": [{"from": "P1/P2", "to": "P3", "threshold_percent": 10.0}],
        })
        microbench["levels"]["P0"].update({"status": "PASS", "experiments": ["timer control"]})
        microbench_path.write_text(json.dumps(microbench))
        global_state = json.loads(global_path.read_text())
        global_state["status"] = "MODEL_READY"
        global_state["revision_history"].append({"revision": 2, "reason": "resource rows and requests mapped"})
        global_path.write_text(json.dumps(global_state))
        queue = json.loads(queue_path.read_text())
        queue["status"] = "ACTIVE"
        queue["requests"] = [experiment_request("DISPATCHED")]
        queue_path.write_text(json.dumps(queue))
        balance = json.loads(balance_path.read_text())
        balance["cases"][0]["resource_rows"][0]["unresolved_request_ids"] = []
        balance_path.write_text(json.dumps(balance))
        unbound_unknown = run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(run_dir), "--to", "EXPERIMENT", "--check-only",
        ], expected=1)
        assert "UNKNOWN requires a dispatched experiment request" in unbound_unknown.stdout
        balance["cases"][0]["resource_rows"][0]["unresolved_request_ids"] = ["req-matched-service"]
        balance_path.write_text(json.dumps(balance))
        run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(run_dir), "--to", "EXPERIMENT",
        ])

        architecture = json.loads(architecture_path.read_text())
        architecture.update({
            "status": "CALIBRATED",
            "allocation_constraints": [{"resource": "registers"}],
            "service_curves": [{"resource": "lsu", "rate": 1.0}],
            "latency_constraints": [{"resource": "lsu", "cycles": 1}],
            "overlap_constraints": [{"a": "lsu", "b": "simt"}],
        })
        architecture_path.write_text(json.dumps(architecture))
        microbench = json.loads(microbench_path.read_text())
        for level in ("P0", "P1", "P2", "P3"):
            microbench["levels"][level].update({"status": "PASS", "experiments": [f"{level} evidence"]})
        microbench_path.write_text(json.dumps(microbench))
        validation_path = run_dir / "models/model_validation.json"
        validation = json.loads(validation_path.read_text())
        validation.update({
            "measurement_system": {"status": "PASS", "evidence": ["P0"]},
            "component_predictions": [{
                "predicted_us": 9.5,
                "measured_us": 10.0,
                "error_percent": 5.0,
                "threshold_percent": 10.0,
                "status": "PASS",
                "evidence": ["P3 samples"],
            }],
        })
        validation_path.write_text(json.dumps(validation))
        schedule = json.loads(schedule_path.read_text())
        schedule["status"] = "CALIBRATED"
        schedule["workload_cases"] = [
            {
                "case_id": case_id,
                "dynamic_instruction_work": {"LDG": 1, "STG": 1},
                "bounds": {
                    "silicon_lower": {"us": 7.0},
                    "resource_service": {"us": 8.0},
                    "dependency": {"us": 7.5},
                    "feasible_schedule": {"us": 9.0},
                },
                "predicted_production_us": 9.5,
                "uncertainty": {"us": 0.5},
            }
            for case_id in ("n262144", "n1048576")
        ]
        schedule_path.write_text(json.dumps(schedule))
        instruction_path = run_dir / "static/instruction_audit.json"
        instruction = json.loads(instruction_path.read_text())
        instruction.update({
            "candidate_id": "candidate-1",
            "source_identity": {"sha256": "1" * 64},
            "toolchain_identity": {"compiler": "fixture"},
            "binary_identity": {"sha256": "3" * 64},
            "resource_usage": {"registers_per_thread": 8},
            "expected_signatures": ["LDG", "STG"],
            "observed_sass_signatures": ["LDG", "STG"],
            "sass_control_flow": [{"block": 0}],
            "dynamic_instruction_counts_by_case": [{"case_id": "n262144", "LDG": 1}],
            "dependency_chains": [["LDG", "STG"]],
            "issue_and_latency_analysis": [{"resource": "lsu"}],
            "occupancy_and_cta_waves": [{"waves": 1}],
            "resource_mapping": [{"instruction": "LDG", "resource": "lsu"}],
            "verdict": "MATCH",
        })
        instruction_path.write_text(json.dumps(instruction))
        balance = json.loads(balance_path.read_text())
        balance["status"] = "CALIBRATED"
        for case in balance["cases"]:
            case["resource_rows"] = [
                resource_row("lsu", "REQUEST_SERVICE", "MEASURED"),
                resource_row("memory", "DEVICE_MEMORY", "BOUNDED"),
            ]
            case["device_coverage"] = {"status": "MEASURED", "fraction": 1.0}
            case["critical_path"] = {"status": "MEASURED", "predicted_us": 9.5}
            case["model_residual"] = {"status": "BOUNDED", "us": 0.5}
        balance["evidence"] = ["fixture measurements"]
        balance_path.write_text(json.dumps(balance))
        queue = json.loads(queue_path.read_text())
        queue["requests"] = [experiment_request("RESOLVED")]
        queue_path.write_text(json.dumps(queue))
        frontier = json.loads(frontier_path.read_text())
        frontier["status"] = "CALIBRATED"
        for case in frontier["cases"]:
            case["current_schedule"] = schedule_point("current", "REJECT")
            case["candidates"] = [schedule_point("candidate-1", "ACCEPT")]
            case["pareto_frontier"] = ["candidate-1"]
        frontier["global_decision"] = {
            "status": "ACCEPT",
            "selected_schedule": "candidate-1",
            "reason": "weighted objective improves within the calibrated model",
            "issued_by_role": "GLOBAL_SCHEDULER",
        }
        frontier["evidence"] = ["fixture A/B"]
        frontier_path.write_text(json.dumps(frontier))
        global_state = json.loads(global_path.read_text())
        global_state["status"] = "CANDIDATE_SELECTED"
        global_state["revision_history"].append({"revision": 3, "reason": "candidate selected"})
        global_path.write_text(json.dumps(global_state))
        decision_dir = run_dir / "candidates/candidate-1"
        decision_dir.mkdir()
        decision_path = decision_dir / "candidate_decision.json"
        decision_path.write_text(json.dumps({
            "schema_version": "candidate-decision-v1",
            "decision": "ACCEPT",
            "correctness": {"status": "PASS"},
            "instruction_audit": {"status": "MATCH", "path": "static/instruction_audit.json"},
        }))
        unsigned_candidate = run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(run_dir), "--to", "PRODUCTION_VALIDATION", "--check-only",
        ], expected=1)
        assert "requires global scheduler ACCEPT" in unsigned_candidate.stdout
        decision_path.write_text(json.dumps({
            "schema_version": "candidate-decision-v1",
            "decision": "ACCEPT",
            "correctness": {"status": "PASS"},
            "instruction_audit": {"status": "MATCH", "path": "static/instruction_audit.json"},
            "global_schedule_decision": {
                "status": "ACCEPT",
                "issued_by_role": "GLOBAL_SCHEDULER",
                "resource_balance_revision": 1,
                "tradeoff_frontier_revision": 1,
                "weighted_objective_effect": {"delta_us": -0.5}
            },
        }))
        run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(run_dir), "--to", "PRODUCTION_VALIDATION",
        ])

        architecture = json.loads(architecture_path.read_text())
        architecture["status"] = "VALIDATED"
        architecture_path.write_text(json.dumps(architecture))
        plan = json.loads(plan_path.read_text())
        plan["status"] = "COMPLETE"
        plan_path.write_text(json.dumps(plan))
        microbench = json.loads(microbench_path.read_text())
        microbench["status"] = "COMPLETE"
        microbench["levels"]["P4"].update({"status": "PASS", "experiments": ["production exact"]})
        microbench_path.write_text(json.dumps(microbench))
        schedule = json.loads(schedule_path.read_text())
        schedule["status"] = "VALIDATED"
        schedule_path.write_text(json.dumps(schedule))
        validation = json.loads(validation_path.read_text())
        validation.update({
            "status": "PASS",
            "production_predictions": [{
                "predicted_us": 9.5,
                "measured_us": 10.0,
                "error_percent": 5.0,
                "threshold_percent": 10.0,
                "status": "PASS",
                "evidence": ["P4 samples"],
            }],
        })
        validation_path.write_text(json.dumps(validation))
        production_path = run_dir / "models/production_validation.json"
        production = json.loads(production_path.read_text())
        production.update({
            "status": "PASS",
            "correctness": {"status": "PASS", "evidence": ["reference"]},
            "source_and_binary_identities": [{"sha256": "3" * 64}],
            "p4_status": "PASS",
            "end_to_end_evidence": ["raw/end_to_end.json"],
            "cases": [
                {
                    "case_id": case_id,
                    "correctness": "PASS",
                    "raw_samples": {"path": f"raw/{case_id}-p4.json"},
                    "gpu_active": {"median_us": 10.0},
                    "end_to_end": {"median_us": 11.0},
                    "predicted_us": 9.5,
                    "measured_us": 10.0,
                    "error_percent": 5.0,
                    "threshold_percent": 10.0,
                    "status": "PASS",
                    "evidence": ["P4 samples"],
                }
                for case_id in ("n262144", "n1048576")
            ],
        })
        production_path.write_text(json.dumps(production))
        global_state = json.loads(global_path.read_text())
        global_state["status"] = "VALIDATED"
        global_state["human_report_gate"]["status"] = "READY"
        global_state["revision_history"].append({"revision": 4, "reason": "production validation"})
        global_path.write_text(json.dumps(global_state))
        balance = json.loads(balance_path.read_text())
        balance["status"] = "VALIDATED"
        balance_path.write_text(json.dumps(balance))
        frontier = json.loads(frontier_path.read_text())
        frontier["status"] = "VALIDATED"
        frontier_path.write_text(json.dumps(frontier))
        queue = json.loads(queue_path.read_text())
        queue["status"] = "CLOSED"
        queue_path.write_text(json.dumps(queue))
        run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(run_dir), "--to", "CERTIFICATION",
        ])
        assert json.loads((run_dir / "run_state.json").read_text())["current_phase"] == "CERTIFICATION"

        (root / "microbench").mkdir()
        for reusable_zone in (
            "hardware/adapters",
            "hardware/measurements",
            "hardware/specs",
            "schemas",
            "scripts",
            "skill",
            "templates",
            "tests",
        ):
            (root / reusable_zone).mkdir(parents=True, exist_ok=True)
        (root / "hardware/measurements/index.json").write_text(json.dumps({
            "schema_version": "hardware-measurement-index-v2", "records": [],
        }))
        (root / "microbench/catalog.json").write_text(json.dumps({
            "schema_version": "microbenchmark-catalog-v2",
            "benchmarks": [],
            "unpublished_families": [],
        }))
        limit_raw = run_dir / "raw/limit_samples.json"
        limit_raw.write_text(json.dumps({"samples": [10.0] * 9}))
        limit_evidence = {"path": str(limit_raw), "sha256": hashlib.sha256(limit_raw.read_bytes()).hexdigest()}
        workload_identity = {"path": str(run_dir / "workload.json"), "sha256": hashlib.sha256((run_dir / "workload.json").read_bytes()).hexdigest()}
        achieved_path = run_dir / "raw/achieved.json"
        achieved_path.write_text(json.dumps({
            "schema_version": "achieved-performance-v1",
            "workload_identity": workload_identity,
            "correctness": {"status": "PASS", "evidence": [limit_evidence]},
            "cases": [
                {"case_id": case_id, "measured_us": 9.9, "confidence": {"level": 0.95, "lower_us": 9.8, "upper_us": 10.0}, "evidence": [limit_evidence]}
                for case_id in ("n262144", "n1048576")
            ],
        }))
        explanation_path = run_dir / "models/architecture_explanation.json"
        explanation_path.write_text(json.dumps({
            "schema_version": "architecture-explanation-v1",
            "status": "VALID",
            "sass_findings": ["final binary instruction mechanism verified"],
            "microarchitecture_findings": ["remaining service is LSU constrained"],
            "bounded_residuals": [{"upper_us": 1.0}],
            "falsification_tests": ["matched LSU dependency control"],
            "missing_evidence": ["proprietary scheduler detail"],
        }))
        explained_output = run_dir / "architecture_limit_certificate.json"
        explained = run([
            sys.executable, str(ROOT / "scripts/emit_certificate.py"),
            "--run", str(run_dir),
            "--achieved", str(achieved_path),
            "--architecture-explanation", str(explanation_path),
            "--output", str(explained_output),
        ])
        assert json.loads(explained.stdout)["status"] == "ARCHITECTURALLY_EXPLAINED"
        fake_bound = run_dir / "models/fake_bound.json"
        fake_bound.write_text(json.dumps({"status": "VALID", "weighted_us": 9.0}))
        fake_proof = run([
            sys.executable, str(ROOT / "scripts/emit_certificate.py"),
            "--run", str(run_dir), "--achieved", str(achieved_path),
            "--silicon-bound", str(fake_bound), "--resource-bound", str(fake_bound),
            "--dag-bound", str(fake_bound), "--schedule-bound", str(fake_bound),
        ], expected=1)
        assert "invalid SILICON_LOWER bound" in fake_proof.stderr
        bound_paths = {}
        bound_kinds = {
            "silicon": "SILICON_LOWER",
            "resource": "RESOURCE_SERVICE_LOWER",
            "dag": "DEPENDENCY_DAG_LOWER",
            "schedule": "FEASIBLE_SCHEDULE_UPPER",
        }
        for name, kind in bound_kinds.items():
            path = run_dir / f"models/{name}_bound.json"
            lower = name != "schedule"
            path.write_text(json.dumps({
                "schema_version": "limit-bound-v1", "kind": kind, "status": "VALID",
                "workload_identity": workload_identity,
                "objective": {"metric": "gpu_active_us", "unit": "us", "direction": "minimize", "statistic": "median"},
                "cases": [
                    {
                        "case_id": case_id,
                        "bound_us": 9.7 if lower else 9.9,
                        "confidence": {"level": 0.95, "lower_us": 9.6 if lower else 9.8, "upper_us": 9.8 if lower else 10.0},
                        "derivation": {"formula": "fixture bound with immutable evidence"},
                        "evidence": [limit_evidence],
                    }
                    for case_id in ("n262144", "n1048576")
                ],
            }))
            bound_paths[name] = path
        certificate = run([
            sys.executable, str(ROOT / "scripts/emit_certificate.py"),
            "--run", str(run_dir),
            "--achieved", str(achieved_path),
            "--silicon-bound", str(bound_paths["silicon"]),
            "--resource-bound", str(bound_paths["resource"]),
            "--dag-bound", str(bound_paths["dag"]),
            "--schedule-bound", str(bound_paths["schedule"]),
        ])
        assert json.loads(certificate.stdout)["status"] == "PROVEN_WITHIN_MODEL"
        run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(run_dir), "--to", "COMPLETE",
        ])
        assert json.loads((run_dir / "run_state.json").read_text())["terminal"] is True

        csv_path = root / "curve.csv"
        csv_path.write_text("repeat,gpu_us\n0,1\n0,1.1\n1,3\n1,3.1\n2,5\n2,5.1\n")
        fit_path = root / "fit.json"
        run([sys.executable, str(ROOT / "scripts/fit_service_curve.py"), "--input", str(csv_path), "--output", str(fit_path), "--bootstrap", "200"])
        assert abs(json.loads(fit_path.read_text())["fit"]["beta"] - 2.0) < 1e-9

        paired = root / "paired.csv"
        paired.write_text("pair,candidate,duration_us\n1,a,10\n1,b,9\n2,b,8\n2,a,10\n3,a,11\n3,b,9\n")
        paired_out = root / "paired.json"
        run([sys.executable, str(ROOT / "scripts/compare_paired.py"), "--input", str(paired), "--baseline", "a", "--candidate", "b", "--correctness", "pass", "--bootstrap", "200", "--output", str(paired_out)])
        assert json.loads(paired_out.read_text())["decision"] == "ACCEPT"

    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "runs/scaffold"
        (run_dir / "microbench_candidates").mkdir(parents=True)
        (run_dir / "run_state.json").write_text(json.dumps({"current_phase": "EXPERIMENT"}))
        scaffold = run([
            sys.executable, str(ROOT / "scripts/new_microbench_candidate.py"),
            "--run", str(run_dir),
            "--id", "nvidia.scaffold-check.v1",
            "--publish-path", "nvidia/scaffold_check",
            "--vendor", "NVIDIA",
            "--family", "scaffold-check",
            "--question", "Does candidate creation remain run-local?",
        ])
        scaffold_result = json.loads(scaffold.stdout)
        assert Path(scaffold_result["candidate"]).is_dir()
        assert Path(scaffold_result["promotion_evidence"]).exists()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        candidate = root / "runs/demo/microbench_candidates/demo-probe"
        candidate.mkdir(parents=True)
        (root / "microbench").mkdir()
        for reusable_zone in (
            "hardware/adapters",
            "hardware/measurements",
            "hardware/specs",
            "schemas",
            "scripts",
            "skill",
            "templates",
            "tests",
        ):
            (root / reusable_zone).mkdir(parents=True)
        (root / "hardware/measurements/index.json").write_text(json.dumps({
            "schema_version": "hardware-measurement-index-v2", "records": [],
        }))
        (root / "microbench/catalog.json").write_text(json.dumps({
            "schema_version": "microbenchmark-catalog-v2",
            "benchmarks": [],
            "unpublished_families": [],
        }))
        definition = {
            "schema_version": "microbenchmark-definition-v2",
            "id": "nvidia.demo-probe.v1",
            "publish_path": "nvidia/demo_probe",
            "status": "CANDIDATE",
            "vendor": "NVIDIA",
            "family": "demo-probe",
            "questions": ["What is the controlled instruction service time?"],
            "controlled_variables": ["repeat count"],
            "source_files": ["probe.cu"],
            "driver_files": ["run.py"],
            "analyzer_files": [],
            "measurement_semantics": {
                "timing_scope": "one native kernel",
                "cache_state": "explicit",
                "repetitions": "explicit",
                "launch_geometry": "explicit",
            },
            "correctness_controls": ["live checksum"],
            "dce_guard": "live checksum",
            "portability": {"hardware_parameters_explicit": ["grid"], "constraints": []},
            "known_pollution": [],
            "claims_allowed": ["matched local service curve"],
            "claims_forbidden": ["cross-device rate"],
            "qualification": {
                "highest_status": "STATIC_VALIDATED",
                "levels": [],
                "static_validation": "PASS",
                "mechanism_validation": "UNASSESSED",
                "device_calibration": "UNASSESSED",
                "production_prediction": "UNASSESSED",
            },
            "genericity": {
                "application_independent": True,
                "production_dependencies": [],
                "hardware_parameters_explicit": True,
                "generated_artifacts_committed": False,
            },
        }
        (candidate / "benchmark.json").write_text(json.dumps(definition))
        (candidate / "probe.cu").write_text("__global__ void probe(unsigned *x) { x[threadIdx.x] += 1; }\n")
        (candidate / "run.py").write_text("print('smoke')\n")
        run_root = root / "runs/demo"
        validation_root = run_root / "microbench_validation/demo-probe"
        validation_root.mkdir(parents=True)
        def immutable(path):
            return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        candidate_identity = immutable(candidate / "benchmark.json")
        static_checks = {
            "correctness", "positive_and_negative_controls", "measurement_smoke",
            "clean_build", "cold_start_reproduction", "genericity_review",
            "static_instruction_validation",
        }
        check_order = sorted(static_checks)
        reproduction_receipts = []
        produced_check_paths = {}
        for process_id in ("cold-process-a", "cold-process-b"):
            output_dir = validation_root / process_id
            artifact_paths = [output_dir / f"{check_id}.json" for check_id in check_order]
            artifact_paths.append(output_dir / "run-marker.json")
            writer_argv = [
                sys.executable, str(ROOT / "tests/fixtures/write_microbench_check_results.py"),
                "--candidate", str(candidate / "benchmark.json"),
                "--output-dir", str(output_dir),
            ]
            for check_id in check_order:
                writer_argv.extend(["--check", check_id])
            receipt = validation_root / f"{process_id}.receipt.json"
            execution_argv = [
                sys.executable, str(ROOT / "scripts/execute_microbench_reproduction.py"),
                "--run", str(run_root), "--candidate", str(candidate),
                "--process-id", process_id, "--argv-json", json.dumps(writer_argv),
                "--output-receipt", str(receipt),
            ]
            for artifact_path in artifact_paths:
                execution_argv.extend(["--artifact", str(artifact_path)])
            run(execution_argv)
            reproduction_receipts.append(immutable(receipt))
            if not produced_check_paths:
                produced_check_paths = {check_id: output_dir / f"{check_id}.json" for check_id in check_order}
        checks = {}
        for check_id in (
            "correctness", "positive_and_negative_controls", "measurement_smoke",
            "clean_build", "cold_start_reproduction", "genericity_review",
            "static_instruction_validation", "mechanism_validation",
            "measurement_system_calibrated", "production_prediction_validation",
        ):
            if check_id not in static_checks:
                checks[check_id] = {"status": "NOT_APPLICABLE", "conclusion": None, "evidence": []}
                continue
            result_path = produced_check_paths[check_id]
            checks[check_id] = {
                "status": "PASS",
                "conclusion": f"{check_id} passed",
                "evidence": [immutable(result_path)],
            }
        evidence = candidate.parent / "demo-probe.promotion.json"
        legacy_evidence = candidate.parent / "legacy.promotion.json"
        legacy_evidence.write_text(json.dumps({
            "schema_version": "microbenchmark-promotion-v1",
            "candidate_id": definition["id"],
            "checks": {name: "PASS" for name in checks},
            "application_terms_removed": [], "commands": ["python run.py"],
            "evidence_paths": ["claimed-only.json"],
        }))
        legacy_rejected = run([
            sys.executable, str(ROOT / "scripts/promote_microbench.py"),
            "--root", str(root), "--candidate", str(candidate),
            "--evidence", str(legacy_evidence), "--check-only",
        ], expected=1)
        assert "promotion evidence" in legacy_rejected.stdout
        evidence.write_text(json.dumps({
            "schema_version": "microbenchmark-promotion-v2",
            "candidate_id": definition["id"],
            "candidate_identity": candidate_identity,
            "checks": checks,
            "application_terms_removed": [],
            "reproduction_receipts": reproduction_receipts,
            "registered_measurement_ids": [],
        }))
        promoted = run([
            sys.executable, str(ROOT / "scripts/harvest_microbenches.py"),
            "--root", str(root), "--run", str(root / "runs/demo"), "--promote",
        ])
        harvest = json.loads(promoted.stdout)
        assert any(item.get("status") == "PUBLISHED" for item in harvest["results"])
        assert harvest["results"][-1]["status"] == "REPOSITORY_AUDIT_PASS"
        published = root / "microbench/nvidia/demo_probe/benchmark.json"
        assert json.loads(published.read_text())["status"] == "PUBLISHED"
        assert json.loads((root / "microbench/catalog.json").read_text())["benchmarks"][0]["id"] == definition["id"]

        device_definition = json.loads((candidate / "benchmark.json").read_text())
        device_definition["qualification"] = {
            "highest_status": "DEVICE_CALIBRATED", "levels": ["P0", "P1", "P2"],
            "static_validation": "PASS", "mechanism_validation": "PASS",
            "device_calibration": "PASS", "production_prediction": "UNASSESSED",
        }
        (candidate / "benchmark.json").write_text(json.dumps(device_definition))
        device_candidate_identity = immutable(candidate / "benchmark.json")
        device_required = sorted(static_checks | {"mechanism_validation", "measurement_system_calibrated"})
        device_receipts = []
        device_outputs = {}
        for process_id in ("device-cold-a", "device-cold-b"):
            output_dir = validation_root / process_id
            artifact_paths = [output_dir / f"{check_id}.json" for check_id in device_required]
            artifact_paths.append(output_dir / "run-marker.json")
            writer_argv = [
                sys.executable, str(ROOT / "tests/fixtures/write_microbench_check_results.py"),
                "--candidate", str(candidate / "benchmark.json"), "--output-dir", str(output_dir),
            ]
            for check_id in device_required:
                writer_argv.extend(["--check", check_id])
            receipt = validation_root / f"{process_id}.receipt.json"
            execution_argv = [
                sys.executable, str(ROOT / "scripts/execute_microbench_reproduction.py"),
                "--run", str(run_root), "--candidate", str(candidate), "--process-id", process_id,
                "--argv-json", json.dumps(writer_argv), "--output-receipt", str(receipt),
            ]
            for artifact_path in artifact_paths:
                execution_argv.extend(["--artifact", str(artifact_path)])
            run(execution_argv)
            device_receipts.append(immutable(receipt))
            if not device_outputs:
                device_outputs = {check_id: output_dir / f"{check_id}.json" for check_id in device_required}
        device_checks = {}
        for check_id in checks:
            if check_id in device_required:
                device_checks[check_id] = {
                    "status": "PASS", "conclusion": f"{check_id} passed",
                    "evidence": [immutable(device_outputs[check_id])],
                }
            else:
                device_checks[check_id] = {"status": "NOT_APPLICABLE", "conclusion": None, "evidence": []}
        evidence.write_text(json.dumps({
            "schema_version": "microbenchmark-promotion-v2", "candidate_id": definition["id"],
            "candidate_identity": device_candidate_identity, "checks": device_checks,
            "application_terms_removed": [], "reproduction_receipts": device_receipts,
            "registered_measurement_ids": ["fabricated-unregistered-measurement"],
        }))
        unregistered_device = run([
            sys.executable, str(ROOT / "scripts/promote_microbench.py"),
            "--root", str(root), "--candidate", str(candidate), "--evidence", str(evidence), "--check-only",
        ], expected=1)
        assert "EVIDENCE_CLOSED_V2" in unregistered_device.stdout
        pollution = published.parent / "build"
        pollution.mkdir()
        (pollution / "probe.o").write_bytes(b"generated")
        rejected_audit = run(
            [sys.executable, str(ROOT / "scripts/audit_repository.py"), "--root", str(root)],
            expected=1,
        )
        assert json.loads(rejected_audit.stdout)["status"] == "FAIL"
        (pollution / "probe.o").unlink()
        pollution.rmdir()

    audit = run([sys.executable, str(ROOT / "scripts/audit_repository.py"), "--root", str(ROOT)])
    assert json.loads(audit.stdout)["status"] == "PASS"
    cli_audit = run([sys.executable, str(ROOT / "scripts/kernel_opt.py"), "audit", "--root", str(ROOT)])
    assert json.loads(cli_audit.stdout)["status"] == "PASS"

    words = ("f" + "la", "q" + "wen", "g" + "dn")
    compound = "delta" + "_rule"
    banned = re.compile(
        r"\b(?:" + "|".join(re.escape(token) for token in words) + r")\b|" + compound,
        re.IGNORECASE,
    )
    reusable_zones = (
        "microbench",
        "hardware/adapters",
        "hardware/specs",
        "schemas",
        "scripts",
        "skill",
        "templates",
        "tests",
    )
    for zone_name in reusable_zones:
        for path in (ROOT / zone_name).rglob("*"):
            if path.is_file() and path.suffix in {".md", ".py", ".json", ".yaml", ".cu", ".sh"}:
                assert not banned.search(path.read_text(errors="ignore")), f"application-specific token in {path}"

    # Evidence-closed workflow: official documents are mandatory, final SASS
    # drives resource discovery, and an empty planned experiment cannot claim
    # to be dispatched.
    official_fixture = ROOT / "tests/fixtures/hardware_evidence.json"
    validated_hardware = run([
        sys.executable, str(ROOT / "scripts/validate_hardware_evidence.py"),
        str(official_fixture),
    ])
    assert json.loads(validated_hardware.stdout)["status"] == "PASS"
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        invalid_manifest = json.loads(official_fixture.read_text())
        invalid_manifest["sources"][0]["url"] = "https://untrusted.example/programming-model"
        invalid_path = temporary / "invalid-hardware-evidence.json"
        invalid_path.write_text(json.dumps(invalid_manifest))
        invalid = run([
            sys.executable, str(ROOT / "scripts/validate_hardware_evidence.py"),
            str(invalid_path),
        ], expected=1)
        assert "not on the declared vendor-official domains" in invalid.stdout
        self_declared = json.loads(official_fixture.read_text())
        self_declared["official_source_policy"]["allowed_official_domains"] = ["untrusted.example"]
        for source in self_declared["sources"]:
            source["url"] = "https://untrusted.example/document"
            source["retrieval"]["requested_url"] = "https://untrusted.example/document"
            source["retrieval"]["final_url"] = "https://untrusted.example/document"
        self_declared_path = temporary / "self-declared-policy.json"
        self_declared_path.write_text(json.dumps(self_declared))
        self_declared_result = run([
            sys.executable, str(ROOT / "scripts/validate_hardware_evidence.py"), str(self_declared_path),
        ], expected=1)
        assert "does not exactly match the repository-trusted adapter" in self_declared_result.stdout

        binary_path = temporary / "kernel.cubin"
        binary_path.write_bytes(b"synthetic final binary")
        sass_input = temporary / "final.sass"
        sass_input.write_text("""Function : synthetic
/*0000*/ MMA;
/*0010*/ MMA;
/*0020*/ LDG.E;
/*0030*/ LDG.E;
/*0040*/ STG.E;
/*0050*/ BAR.SYNC;
/*0060*/ EXIT;
/*0070*/ EXIT;
""")
        disassembly_receipt = temporary / "disassembly.json"
        disassembly_receipt.write_text(json.dumps({
            "schema_version": "final-binary-disassembly-receipt-v1", "status": "PASS", "exit_code": 0,
            "command": ["synthetic-disassembler", str(binary_path)],
            "tool": {"path": "synthetic-disassembler", "version": "1"},
            "target": {"vendor": "TEST", "device_name": "synthetic-device"},
            "binary_identity": {"path": str(binary_path), "sha256": hashlib.sha256(binary_path.read_bytes()).hexdigest()},
            "sass_identity": {"path": str(sass_input), "sha256": hashlib.sha256(sass_input.read_bytes()).hexdigest()},
        }))
        sass_path = temporary / "sass-summary.json"
        run([
            sys.executable, str(ROOT / "scripts/count_sass.py"),
            "--input", str(sass_input), "--binary", str(binary_path),
            "--disassembly-receipt", str(disassembly_receipt), "--output", str(sass_path),
        ])
        unknown_sass = temporary / "unknown.sass"
        unknown_sass.write_text("Function : unknown\n/*0000*/ UNDOCUMENTEDOP;\n")
        unknown_receipt = temporary / "unknown-disassembly.json"
        unknown_receipt.write_text(json.dumps({
            "schema_version": "final-binary-disassembly-receipt-v1", "status": "PASS", "exit_code": 0,
            "command": ["synthetic-disassembler", str(binary_path)], "tool": {"path": "synthetic", "version": "1"},
            "target": {"vendor": "TEST", "device_name": "synthetic-device"},
            "binary_identity": {"path": str(binary_path), "sha256": hashlib.sha256(binary_path.read_bytes()).hexdigest()},
            "sass_identity": {"path": str(unknown_sass), "sha256": hashlib.sha256(unknown_sass.read_bytes()).hexdigest()},
        }))
        unknown_summary = temporary / "unknown-summary.json"
        unknown_result = run([
            sys.executable, str(ROOT / "scripts/count_sass.py"), "--input", str(unknown_sass),
            "--binary", str(binary_path), "--disassembly-receipt", str(unknown_receipt),
            "--output", str(unknown_summary),
        ], expected=1)
        assert json.loads(unknown_result.stdout)["status"] == "BLOCKED"
        assert "UNDOCUMENTEDOP" in json.loads(unknown_summary.read_text())["unclassified_mnemonics"]
        discovery_path = temporary / "resource-discovery.json"
        discovered = run([
            sys.executable, str(ROOT / "scripts/discover_resources.py"),
            "--sass-summary", str(sass_path),
            "--hardware-evidence", str(official_fixture),
            "--output", str(discovery_path),
        ])
        assert json.loads(discovered.stdout)["status"] == "READY"
        resource_ids = set(json.loads(discovery_path.read_text())["required_resource_ids"])
        assert {"tensor_compute", "load_store_request", "l2_boundary", "device_memory_boundary", "synchronization"} <= resource_ids

        receipt_path = temporary / "catalog-receipt.json"
        query = run([
            sys.executable, str(ROOT / "scripts/query_microbench_catalog.py"),
            "--catalog", str(ROOT / "microbench/catalog.json"),
            "--resource", "kernel_dispatch",
            "--mechanism", "kernel_launch",
            "--boundary", "launch",
            "--qualification", "STATIC_VALIDATED",
            "--output", str(receipt_path),
        ])
        assert json.loads(query.stdout)["selected_package_id"] == "nvidia.cuda-core-service-curves.v1"

        mismatched_hardware = json.loads((ROOT / "tests/fixtures/hardware.json").read_text())
        mismatched_hardware["target"]["device_name"] = "different-device"
        mismatched_hardware_path = temporary / "mismatched-hardware.json"
        mismatched_hardware_path.write_text(json.dumps(mismatched_hardware))
        target_mismatch = run([
            sys.executable, str(ROOT / "scripts/new_run.py"),
            "--root", str(temporary / "mismatch-root"), "--run-id", "target-mismatch",
            "--operator", str(ROOT / "tests/fixtures/operator.json"),
            "--workload", str(ROOT / "tests/fixtures/workload.json"),
            "--hardware", str(mismatched_hardware_path),
            "--hardware-evidence", str(official_fixture),
        ], expected=1)
        assert "target identity mismatch for device_name" in target_mismatch.stderr

        strict_root = temporary / "strict-root"
        strict = run([
            sys.executable, str(ROOT / "scripts/new_run.py"),
            "--root", str(strict_root), "--run-id", "strict-self-test",
            "--operator", str(ROOT / "tests/fixtures/operator.json"),
            "--workload", str(ROOT / "tests/fixtures/workload.json"),
            "--hardware", str(ROOT / "tests/fixtures/hardware.json"),
            "--hardware-evidence", str(official_fixture),
        ])
        strict_run = Path(json.loads(strict.stdout)["run_dir"])
        strict_state = json.loads((strict_run / "run_state.json").read_text())
        assert strict_state["framework_contract_version"] == "evidence-closed-v2"
        strict_state_path = strict_run / "run_state.json"
        malformed_state = dict(strict_state)
        malformed_state["undeclared_field"] = True
        strict_state_path.write_text(json.dumps(malformed_state))
        malformed_gate = run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(strict_run), "--to", "BASELINE", "--check-only",
        ], expected=1)
        assert "additional property is forbidden" in malformed_gate.stdout, malformed_gate.stdout
        strict_state_path.write_text(json.dumps(strict_state))
        next_action = run([
            sys.executable, str(ROOT / "scripts/optimizer_step.py"),
            "--run", str(strict_run),
        ])
        assert (strict_run / "models/candidate_pool.json").is_file()
        assert (strict_run / "models/opportunity_map.json").is_file()
        assert json.loads(next_action.stdout)["action"] == "CAPTURE_DISCOVERY_BASELINE"
        strict_gate = run([
            sys.executable, str(ROOT / "scripts/advance_run.py"),
            "--run", str(strict_run), "--to", "BASELINE", "--check-only",
        ], expected=1)
        assert "resource discovery must be READY" in strict_gate.stdout
        broken_contract = json.loads(strict_state_path.read_text())
        broken_contract.pop("framework_contract_version")
        strict_state_path.write_text(json.dumps(broken_contract))
        blocked_contract = run([
            sys.executable, str(ROOT / "scripts/optimizer_step.py"), "--run", str(strict_run),
        ], expected=1)
        assert json.loads(blocked_contract.stdout)["action"] == "BLOCK_INVALID_FRAMEWORK_CONTRACT"
    closed_loop = run([sys.executable, str(ROOT / "tests/test_evidence_closed_workflow.py")])
    assert "evidence-closed workflow test: PASS" in closed_loop.stdout
    print("repository self-test: PASS")


if __name__ == "__main__":
    main()
