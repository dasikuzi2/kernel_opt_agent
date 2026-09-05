#!/usr/bin/env python3
"""Derive experiment benefit bounds from the production DAG window and rank them."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evidence_utils import read_object, sha256
from supervision_utils import artifact_path, validate_admissibility_contract, validate_decision_contract


UNCERTAINTY_WEIGHT = {"LOW": 0.25, "MEDIUM": 0.6, "HIGH": 1.0}
COST_WEIGHT = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 4.0}


def request_stages(request: dict, stage_times: dict) -> set[str]:
    stages = set(map(str, request.get("affected_stage_ids", [])))
    if not stages:
        raise ValueError(f"{request.get('request_id')}: affected_stage_ids must be explicit; resource names are not stage names")
    missing = stages - set(stage_times)
    if missing:
        raise ValueError(f"{request.get('request_id')}: affected stages are absent from production DAG timing: {sorted(missing)}")
    return stages


def resource_probability(case: dict, resource_ids: set[str]) -> tuple[float, str]:
    rows = [row for row in case.get("resource_rows", []) if str(row.get("resource_id")) in resource_ids]
    if not rows:
        raise ValueError(f"case {case.get('case_id')}: requested resources are absent from the balance ledger")
    probabilities = []
    statuses = []
    for row in rows:
        critical = row.get("critical_path", {})
        probability = float(critical.get("probability", -1.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"case {case.get('case_id')}/{row.get('resource_id')}: critical-path probability must be in [0,1]")
        probabilities.append(probability)
        statuses.extend([
            row.get("matched_saturation", {}).get("status"),
            row.get("utilization", {}).get("status"),
            critical.get("status"),
        ])
    uncertainty = "HIGH" if "UNKNOWN" in statuses else ("MEDIUM" if "BOUNDED" in statuses else "LOW")
    return max(probabilities), uncertainty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    workload = read_object(run / "workload.json")
    queue_path = run / "models/experiment_queue.json"
    queue = read_object(queue_path)

    queue_version = queue.get("schema_version")
    if queue_version == "experiment-request-queue-v3":
        active = [request for request in queue.get("requests", []) if request.get("status") in {"PROPOSED", "PLANNED"}]
        if not active:
            resolved = [
                request for request in queue.get("requests", [])
                if request.get("status") == "RESOLVED"
                and request.get("result_binding", {}).get("status") == "BOUND"
                and request.get("result_binding", {}).get("model_reconciliation", {}).get("status") == "APPLIED"
            ]
            if not resolved or len(resolved) != len(queue.get("requests", [])):
                raise ValueError(
                    "static-admissibility queue may have zero active gates only "
                    "after every gate is RESOLVED with APPLIED reconciliation"
                )
            for request in resolved:
                contract_path = artifact_path(run, request.get("admissibility_contract", {}))
                errors = validate_admissibility_contract(contract_path, run) if contract_path.is_file() else ["admissibility contract is missing"]
                if errors:
                    raise ValueError(f"{request.get('request_id')}: " + "; ".join(errors))
                request["priority"] = 0
                request.pop("sensitivity", None)
            ranked_at = datetime.now(timezone.utc).isoformat()
            queue["ranking_policy"] = {
                "issued_by_role": "GLOBAL_SCHEDULER",
                "formula": "NO_PERFORMANCE_RANKING_RESOLVED_STATIC_GATE",
                "benefit_bound": "NOT_APPLICABLE_STATIC_ADMISSIBILITY",
                "ranked_at": ranked_at,
            }
            queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
            receipt = {
                "schema_version": "experiment-ranking-receipt-v3",
                "ranked_at": ranked_at,
                "queue_identity": {"path": str(queue_path), "sha256": sha256(queue_path)},
                "ranking": [
                    {"request_id": request["request_id"], "priority": 0, "score": None, "status": "RESOLVED"}
                    for request in resolved
                ],
                "active_gate_count": 0,
                "performance_ranking_performed": False,
            }
            output = run / "traces/experiment_ranking_receipt.json"
            output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            print(json.dumps({
                "status": "PASS", "requests": len(resolved), "active_gates": 0,
                "receipt": str(output), "performance_ranking": False,
            }, sort_keys=True))
            return 0
        if len(active) != 1:
            raise ValueError("static-admissibility queue must contain exactly one active gate; performance ranking is forbidden")
        request = active[0]
        contract_path = artifact_path(run, request.get("admissibility_contract", {}))
        errors = validate_admissibility_contract(contract_path, run) if contract_path.is_file() else ["admissibility contract is missing"]
        if errors:
            raise ValueError(f"{request.get('request_id')}: " + "; ".join(errors))
        request["priority"] = 0
        request.pop("sensitivity", None)
        ranked_at = datetime.now(timezone.utc).isoformat()
        queue["ranking_policy"] = {
            "issued_by_role": "GLOBAL_SCHEDULER",
            "formula": "NO_PERFORMANCE_RANKING_SINGLE_STATIC_GATE",
            "benefit_bound": "NOT_APPLICABLE_STATIC_ADMISSIBILITY",
            "ranked_at": ranked_at,
        }
        queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
        receipt = {
            "schema_version": "experiment-ranking-receipt-v3",
            "ranked_at": ranked_at,
            "queue_identity": {"path": str(queue_path), "sha256": sha256(queue_path)},
            "ranking": [{"request_id": request["request_id"], "priority": 0, "score": None}],
            "performance_ranking_performed": False,
        }
        output = run / "traces/experiment_ranking_receipt.json"
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "PASS", "requests": 1, "receipt": str(output), "performance_ranking": False}, sort_keys=True))
        return 0

    if queue_version == "experiment-request-queue-v2":
        value_field = "candidate_specific_decision_value_us"
        ranking = []
        terminal = []
        for request in queue.get("requests", []):
            if request.get("status") not in {"PROPOSED", "PLANNED"}:
                terminal.append(request)
                continue
            decision_path = artifact_path(run, request.get("decision_contract", {}))
            errors = validate_decision_contract(decision_path, run) if decision_path.is_file() else ["decision contract is missing"]
            if errors:
                raise ValueError(f"{request.get('request_id')}: " + "; ".join(errors))
            decision = read_object(decision_path)
            need = decision["measurement_need"]
            value = float(need["maximum_decision_value"]["value"])
            probability = float(need["decision_flip_probability"])
            reduction = float(need["expected_uncertainty_reduction"])
            cost = str(request.setdefault("sensitivity", {}).get("experiment_cost", "HIGH")).upper()
            cost_weight = COST_WEIGHT.get(cost)
            if cost_weight is None:
                raise ValueError(f"{request.get('request_id')}: invalid experiment cost")
            score = value * probability * reduction / cost_weight
            request["sensitivity"].update({
                value_field: value,
                "decision_flip_probability": probability,
                "expected_uncertainty_reduction": reduction,
                "experiment_cost": cost,
                "experiment_cost_weight": cost_weight,
                "ranking_score": score,
                "benefit_derivation": {
                    "method": "candidate decision sensitivity from the frozen decision contract",
                    "decision_contract_identity": {"path": decision_path.relative_to(run).as_posix(), "sha256": sha256(decision_path)},
                    "top_two_candidate_ids": decision["top_two_candidate_ids"],
                    "quantity_id": need["quantity_id"],
                    "decision_boundary": need["decision_boundary"],
                    "maximum_decision_value": need["maximum_decision_value"],
                },
            })
            ranking.append(request)
        ranking.sort(key=lambda item: (-float(item["sensitivity"]["ranking_score"]), item["request_id"]))
        queue["requests"] = [*ranking, *terminal]
        for priority, request in enumerate(queue["requests"]):
            request["priority"] = priority
        ranked_at = datetime.now(timezone.utc).isoformat()
        queue["ranking_policy"] = {
            "issued_by_role": "GLOBAL_SCHEDULER",
            "formula": f"{value_field} * decision_flip_probability * expected_uncertainty_reduction / experiment_cost_weight",
            "benefit_bound": "frozen candidate decision contract",
            "ranked_at": ranked_at,
        }
        queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
        receipt = {
            "schema_version": "experiment-ranking-receipt-v2",
            "ranked_at": ranked_at,
            "queue_identity": {"path": str(queue_path), "sha256": sha256(queue_path)},
            "ranking": [{"request_id": item["request_id"], "priority": item["priority"], "score": item["sensitivity"]["ranking_score"]} for item in queue["requests"]],
        }
        output = run / "traces/experiment_ranking_receipt.json"
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "PASS", "requests": len(queue["requests"]), "receipt": str(output)}, sort_keys=True))
        return 0

    balance_path = run / "models/resource_balance.json"
    balance = read_object(balance_path)
    weights = {str(case["id"]): float(case.get("weight", 0.0)) for case in workload.get("cases", [])}
    cases = {str(case["case_id"]): case for case in balance.get("cases", [])}
    for request in queue.get("requests", []):
        contributions = []
        upper = 0.0
        probability_numerator = 0.0
        uncertainty_levels = []
        requested_resources = set(map(str, request.get("resource_ids", [])))
        for case_id in request.get("workload_cases", []):
            case = cases[str(case_id)]
            critical = case.get("critical_path", {})
            stage_times = critical.get("stage_gpu_active_us", {})
            total = float(critical.get("total_us", sum(stage_times.values())))
            stages = request_stages(request, stage_times)
            removable = min(total, sum(float(stage_times[stage]) for stage in stages))
            weight = weights.get(str(case_id), 0.0)
            weighted = removable * weight
            probability, case_uncertainty = resource_probability(case, requested_resources)
            upper += weighted
            probability_numerator += weighted * probability
            uncertainty_levels.append(case_uncertainty)
            contributions.append({
                "case_id": str(case_id), "stages": sorted(stages),
                "case_weight": weight, "max_removable_us": removable,
                "weighted_upper_bound_us": weighted,
                "critical_path_probability": probability,
                "uncertainty": case_uncertainty,
            })
        sensitivity = request.setdefault("sensitivity", {})
        uncertainty = "HIGH" if "HIGH" in uncertainty_levels else ("MEDIUM" if "MEDIUM" in uncertainty_levels else "LOW")
        cost = str(sensitivity.get("experiment_cost", "HIGH")).upper()
        probability = probability_numerator / upper if upper > 0 else 0.0
        uncertainty_weight = UNCERTAINTY_WEIGHT.get(uncertainty)
        cost_weight = COST_WEIGHT.get(cost)
        if uncertainty_weight is None or cost_weight is None or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid sensitivity policy for {request.get('request_id')}")
        score = upper * probability * uncertainty_weight / cost_weight
        sensitivity.update({
            "max_weighted_benefit_us": upper,
            "critical_path_probability": probability,
            "uncertainty": uncertainty,
            "experiment_cost": cost,
            "uncertainty_weight": uncertainty_weight,
            "experiment_cost_weight": cost_weight,
            "ranking_score": score,
            "benefit_derivation": {
                "method": "weighted sum of the conservative production stage windows touched by the requested resources; capped by case total",
                "case_contributions": contributions,
                "resource_balance_identity": {"path": str(balance_path), "sha256": sha256(balance_path)},
            },
        })
    queue["requests"].sort(key=lambda request: (-float(request["sensitivity"]["ranking_score"]), request["request_id"]))
    for priority, request in enumerate(queue["requests"]):
        request["priority"] = priority
    ranked_at = datetime.now(timezone.utc).isoformat()
    queue["ranking_policy"] = {
        "issued_by_role": "GLOBAL_SCHEDULER",
        "formula": "max_weighted_benefit_us * critical_path_probability * uncertainty_weight / experiment_cost_weight",
        "benefit_bound": "resource-constrained production stage windows",
        "ranked_at": ranked_at,
    }
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "experiment-ranking-receipt-v1",
        "ranked_at": ranked_at,
        "resource_balance_identity": {"path": str(balance_path), "sha256": sha256(balance_path)},
        "queue_identity": {"path": str(queue_path), "sha256": sha256(queue_path)},
        "ranking": [{"request_id": item["request_id"], "priority": item["priority"], "score": item["sensitivity"]["ranking_score"]} for item in queue["requests"]],
    }
    output = run / "traces/experiment_ranking_receipt.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "requests": len(queue["requests"]), "receipt": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
