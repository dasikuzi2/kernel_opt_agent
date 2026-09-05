#!/usr/bin/env python3
"""Tests for evidence-derived, fail-closed serving policy selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from derive_serving_policy import derive


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def trace(cache_root: str, switches: dict[str, str], throughputs: dict[int, float], drift: bool = False) -> dict:
    controls = {
        "quantization": "none",
        "new_tokens": 2,
        "warmups": 1,
        "trials": 1,
        "ignore_eos": True,
        "sampling": "greedy",
        "case_order": "forward",
        "expected_vllm_cache_root": cache_root,
    }
    environment = {
        "gpu": "fixture GPU",
        "vllm": "fixture vLLM",
        "torch": "fixture torch",
        "cuda": "fixture CUDA",
        "vllm_cache_root": cache_root,
        "guarded_source_sha256": {"utils.py": "abc"},
    }
    for name, value in switches.items():
        controls[f"expected_{name}"] = value
        controls[f"actual_{name}"] = value
    raw_samples = []
    for batch_size in throughputs:
        generated = [[1, 2] for _ in range(batch_size)]
        if drift and batch_size == 1:
            generated[0] = [1, 3]
        for phase in ("warmup", "measure"):
            raw_samples.append({
                "batch_size": batch_size,
                "phase": phase,
                "iteration": 0,
                "generated_token_ids": generated,
                "aggregate_output_tokens_per_second": throughputs[batch_size],
            })
    return {
        "schema_version": "qwen35-vllm-batch-service-v1",
        "status": "PASS",
        "model": "fixture model",
        "workload_identity": {
            "harness_sha256": "a" * 64,
            "prompt_token_ids_sha256": ["b" * 64],
            "prompt_count": 1,
            "request_selection": "fixture rotation",
        },
        "environment": environment,
        "controls": controls,
        "service_curve": [
            {
                "batch_size": batch_size,
                "median_aggregate_output_tokens_per_second": throughput,
                "all_requests_exact_length": True,
            }
            for batch_size, throughput in throughputs.items()
        ],
        "raw_samples": raw_samples,
    }


def make_spec(tmp_path: Path, *, candidate_cache: str = "candidate-cache", jit_ready: bool = True) -> Path:
    write_json(tmp_path / "lifecycle.json", {"status": "fixture evidence"})
    switches = {"optimized": "off"}
    write_json(tmp_path / "baseline.json", trace("baseline-cache", switches, {1: 100.0, 4: 400.0}))
    write_json(tmp_path / "baseline2.json", trace("baseline-cache-2", switches, {1: 102.0, 4: 395.0}))
    write_json(
        tmp_path / "candidate.json",
        trace(candidate_cache, {"optimized": "on"}, {1: 110.0, 4: 390.0}, drift=True),
    )
    write_json(
        tmp_path / "candidate2.json",
        trace(f"{candidate_cache}-2", {"optimized": "on"}, {1: 112.0, 4: 385.0}, drift=True),
    )
    spec = {
        "schema_version": "serving-policy-spec-v1",
        "deployment_point": {"name": "fixture"},
        "identity_controls": ["quantization", "new_tokens", "warmups", "trials", "ignore_eos", "sampling", "case_order"],
        "expected_harness_sha256": "a" * 64,
        "expected_prompt_token_ids_sha256": ["b" * 64],
        "minimum_independent_traces_per_variant": 2,
        "require_distinct_vllm_cache_roots": True,
        "variants": [
            {
                "id": "stock",
                "role": "baseline",
                "supported_contracts": ["stock_token_identity", "drift_allowed"],
                "traces": ["baseline.json", "baseline2.json"],
                "required_switches": {"optimized": "off"},
                "lifecycle": {},
            },
            {
                "id": "candidate",
                "role": "candidate",
                "supported_contracts": ["stock_token_identity", "drift_allowed"],
                "traces": ["candidate.json", "candidate2.json"],
                "required_switches": {"optimized": "on"},
                "lifecycle": {
                    "requires_prepack": True,
                    "prepack_ready": True,
                    "requires_jit_warm": True,
                    "jit_warm_ready": jit_ready,
                    "evidence": ["lifecycle.json"],
                },
            },
        ],
        "contracts": [
            {
                "id": "stock_token_identity",
                "baseline": "stock",
                "candidates": ["candidate"],
                "requires_stock_token_identity": True,
                "minimum_speedup": 1.03,
            },
            {
                "id": "drift_allowed",
                "baseline": "stock",
                "candidates": ["candidate"],
                "requires_stock_token_identity": False,
                "minimum_speedup": 1.03,
            },
        ],
        "limitations": ["fixture is bounded"],
    }
    path = tmp_path / "spec.json"
    write_json(path, spec)
    return path


def test_policy_routes_by_contract_and_batch(tmp_path: Path) -> None:
    result = derive(make_spec(tmp_path))
    routes = {
        (row["numerical_contract"], row["active_decode_batch"]): row
        for row in result["routes"]
    }
    assert routes[("stock_token_identity", 1)]["route"] == "stock"
    assert routes[("drift_allowed", 1)]["route"] == "candidate"
    assert routes[("drift_allowed", 1)]["measured_conservative_speedup_over_baseline"] == pytest.approx(110 / 102)
    assert routes[("drift_allowed", 4)]["route"] == "stock"
    assert result["global_optimum_proven"] is False


def test_baseline_only_contract_binds_evidence_and_never_promotes(tmp_path: Path) -> None:
    spec_path = make_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["contracts"][0]["candidates"] = []
    spec["contracts"][0]["evidence"] = ["lifecycle.json"]
    spec["contracts"][0]["baseline_only_reason"] = "candidate failed quality authority"
    write_json(spec_path, spec)

    result = derive(spec_path)
    exact_routes = [
        row for row in result["routes"]
        if row["numerical_contract"] == "stock_token_identity"
    ]
    assert all(row["route"] == "stock" for row in exact_routes)
    assert all(
        row["reason"] == "candidate failed quality authority"
        for row in exact_routes
    )
    contract_inputs = result["evidence"]["contract:stock_token_identity"]["contract_inputs"]
    assert contract_inputs[0]["path"] == "lifecycle.json"
    assert len(contract_inputs[0]["sha256"]) == 64


def test_policy_rejects_unisolated_cache(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not isolated"):
        derive(make_spec(tmp_path, candidate_cache="baseline-cache"))


def test_policy_rejects_insufficient_repetition(tmp_path: Path) -> None:
    spec_path = make_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["variants"][1]["traces"] = ["candidate.json"]
    write_json(spec_path, spec)
    with pytest.raises(ValueError, match="insufficient independent traces"):
        derive(spec_path)


def test_less_qualified_baseline_can_veto_but_not_promote(tmp_path: Path) -> None:
    spec_path = make_spec(tmp_path)
    challenger = trace("unused", {"optimized": "off"}, {1: 111.0, 4: 410.0})
    challenger["environment"].pop("vllm_cache_root")
    challenger["environment"].pop("guarded_source_sha256")
    challenger["controls"].pop("expected_vllm_cache_root")
    write_json(tmp_path / "challenger.json", challenger)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["variants"][0]["challenger_traces"] = ["challenger.json"]
    write_json(spec_path, spec)
    result = derive(spec_path)
    routes = {
        (row["numerical_contract"], row["active_decode_batch"]): row
        for row in result["routes"]
    }
    assert routes[("drift_allowed", 1)]["route"] == "stock"
    evidence = result["evidence"]["drift_allowed:candidate"]
    assert evidence["baseline_challenger_inputs"][0]["use"] == "BASELINE_ENVELOPE_VETO_ONLY"


def test_policy_rejects_unready_lifecycle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JIT warmup"):
        derive(make_spec(tmp_path, jit_ready=False))


def test_policy_rejects_mismatched_workload(tmp_path: Path) -> None:
    spec_path = make_spec(tmp_path)
    for name in ("candidate.json", "candidate2.json"):
        candidate_path = tmp_path / name
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate["controls"]["new_tokens"] = 3
        for sample in candidate["raw_samples"]:
            for token_ids in sample["generated_token_ids"]:
                token_ids.append(0)
        write_json(candidate_path, candidate)
    with pytest.raises(ValueError, match="workload or environment identity"):
        derive(spec_path)
