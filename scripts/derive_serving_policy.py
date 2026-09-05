#!/usr/bin/env python3
"""Derive a fail-closed, batch-aware serving policy from raw vLLM traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from schema_utils import validate_instance


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_trace(spec_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (spec_path.parent / path).resolve()


def bind_inputs(spec_path: Path, values: list[str]) -> list[dict]:
    bound = []
    for value in values:
        path = resolve_trace(spec_path, value)
        require(path.is_file(), f"bound evidence does not exist: {path}")
        bound.append({"path": value, "sha256": sha256(path)})
    return bound


def curve_by_batch(trace: dict) -> dict[int, dict]:
    return {int(point["batch_size"]): point for point in trace["service_curve"]}


def measured_samples(trace: dict) -> dict[tuple[int, int], list[list[int]]]:
    samples: dict[tuple[int, int], list[list[int]]] = {}
    for sample in trace["raw_samples"]:
        if sample.get("phase") != "measure":
            continue
        key = (int(sample["batch_size"]), int(sample["iteration"]))
        require(key not in samples, f"duplicate measured sample {key}")
        samples[key] = sample["generated_token_ids"]
    return samples


def compare_outputs(baselines: list[dict], candidates: list[dict]) -> dict[int, dict]:
    totals: dict[int, dict[str, int]] = {}
    for baseline in baselines:
        baseline_samples = measured_samples(baseline)
        for candidate in candidates:
            candidate_samples = measured_samples(candidate)
            require(
                baseline_samples.keys() == candidate_samples.keys(),
                "baseline and candidate measured sample identities differ",
            )
            for (batch_size, iteration), expected_requests in baseline_samples.items():
                actual_requests = candidate_samples[(batch_size, iteration)]
                require(
                    len(expected_requests) == len(actual_requests) == batch_size,
                    f"batch {batch_size}: generated request count is inconsistent",
                )
                row = totals.setdefault(
                    batch_size,
                    {"exact_requests": 0, "request_count": 0, "exact_tokens": 0, "token_count": 0},
                )
                for expected, actual in zip(expected_requests, actual_requests):
                    row["request_count"] += 1
                    row["exact_requests"] += int(expected == actual)
                    row["token_count"] += max(len(expected), len(actual))
                    row["exact_tokens"] += sum(
                        left == right for left, right in zip(expected, actual)
                    )
    return {
        batch_size: {
            **counts,
            "all_requests_exact": counts["exact_requests"] == counts["request_count"],
            "token_identity_fraction": counts["exact_tokens"] / counts["token_count"],
        }
        for batch_size, counts in sorted(totals.items())
    }


def validate_trace(trace: dict, variant: dict, trace_path: Path, *, qualified: bool = True) -> None:
    require(trace.get("schema_version") == "qwen35-vllm-batch-service-v1", f"{trace_path}: unsupported trace schema")
    require(trace.get("status") == "PASS", f"{trace_path}: trace status is not PASS")
    controls = trace.get("controls", {})
    environment = trace.get("environment", {})
    if qualified:
        cache_root = environment.get("vllm_cache_root")
        require(cache_root, f"{trace_path}: VLLM_CACHE_ROOT was not captured")
        require(
            controls.get("expected_vllm_cache_root") == cache_root,
            f"{trace_path}: expected and actual VLLM_CACHE_ROOT differ",
        )
        guarded = environment.get("guarded_source_sha256", {})
        require(guarded, f"{trace_path}: guarded source hashes are missing")
    for switch, expected in variant.get("required_switches", {}).items():
        expected_key = f"expected_{switch}"
        actual_key = f"actual_{switch}"
        if qualified:
            require(
                controls.get(expected_key) == expected
                and controls.get(actual_key) == expected,
                f"{trace_path}: runtime switch {switch} is not proven {expected}",
            )
        else:
            recorded = [
                controls[key]
                for key in (expected_key, actual_key)
                if controls.get(key) is not None
            ]
            require(
                all(value == expected for value in recorded),
                f"{trace_path}: challenger runtime switch {switch} contradicts {expected}",
            )
    curve = trace.get("service_curve", [])
    require(curve, f"{trace_path}: service curve is empty")
    curve_batches = [int(point["batch_size"]) for point in curve]
    require(len(curve_batches) == len(set(curve_batches)), f"{trace_path}: duplicate service-curve batch")
    for point in curve:
        require(point.get("all_requests_exact_length") is True, f"{trace_path}: output length check failed")
        require(
            float(point.get("median_aggregate_output_tokens_per_second", 0)) > 0,
            f"{trace_path}: non-positive throughput",
        )
    warmups = int(controls.get("warmups", -1))
    trials = int(controls.get("trials", -1))
    new_tokens = int(controls.get("new_tokens", -1))
    require(warmups >= 1 and trials >= 1 and new_tokens >= 1, f"{trace_path}: invalid sampling controls")
    for batch_size in curve_batches:
        for phase, expected_count in (("warmup", warmups), ("measure", trials)):
            samples = [
                sample for sample in trace.get("raw_samples", [])
                if int(sample.get("batch_size", -1)) == batch_size and sample.get("phase") == phase
            ]
            require(len(samples) == expected_count, f"{trace_path}: batch {batch_size} {phase} count differs from controls")
            for sample in samples:
                generated = sample.get("generated_token_ids", [])
                require(len(generated) == batch_size, f"{trace_path}: batch {batch_size} request count mismatch")
                require(all(len(ids) == new_tokens for ids in generated), f"{trace_path}: generated length mismatch")
        measured = [
            float(sample["aggregate_output_tokens_per_second"])
            for sample in trace["raw_samples"]
            if int(sample.get("batch_size", -1)) == batch_size and sample.get("phase") == "measure"
        ]
        reported = float(curve_by_batch(trace)[batch_size]["median_aggregate_output_tokens_per_second"])
        recomputed = float(statistics.median(measured))
        require(abs(reported - recomputed) <= max(1e-9, abs(recomputed) * 1e-12), f"{trace_path}: service curve does not recompute from raw samples")


def derive(spec_path: Path) -> dict:
    spec = load_json(spec_path)
    spec_schema = load_json(ROOT / "schemas" / "serving_policy_spec.schema.json")
    errors = validate_instance(spec, spec_schema)
    if errors:
        raise ValueError("serving policy spec validation failed:\n" + "\n".join(errors))

    variants: dict[str, dict] = {}
    cache_roots: list[str] = []
    common_identity: dict | None = None
    shared_source_identity: dict | None = None
    identity_control_names = spec["identity_controls"]
    variant_ids = [variant["id"] for variant in spec["variants"]]
    require(len(variant_ids) == len(set(variant_ids)), "variant IDs must be unique")
    contract_ids = [contract["id"] for contract in spec["contracts"]]
    require(len(contract_ids) == len(set(contract_ids)), "contract IDs must be unique")
    for variant_spec in spec["variants"]:
        traces = []
        challenger_traces = []
        inputs = []
        challenger_inputs = []
        require(
            len(variant_spec["traces"]) >= spec["minimum_independent_traces_per_variant"],
            f"{variant_spec['id']}: insufficient independent traces",
        )
        for trace_name in variant_spec["traces"]:
            path = resolve_trace(spec_path, trace_name)
            trace = load_json(path)
            validate_trace(trace, variant_spec, path)
            workload_identity = trace.get("workload_identity")
            require(workload_identity, f"{path}: cryptographic workload identity is missing")
            require(
                workload_identity.get("harness_sha256") == spec["expected_harness_sha256"],
                f"{path}: benchmark harness identity differs from the frozen spec",
            )
            require(
                workload_identity.get("prompt_token_ids_sha256") == spec["expected_prompt_token_ids_sha256"],
                f"{path}: prompt-token identity differs from the frozen spec",
            )
            identity = {
                "model": trace["model"],
                "workload_identity": workload_identity,
                "gpu": trace["environment"]["gpu"],
                "vllm": trace["environment"]["vllm"],
                "torch": trace["environment"]["torch"],
                "cuda": trace["environment"]["cuda"],
                "controls": {name: trace["controls"].get(name) for name in identity_control_names},
                "batch_sizes": sorted(curve_by_batch(trace)),
            }
            if common_identity is None:
                common_identity = identity
            require(identity == common_identity, f"{path}: workload or environment identity differs")
            source_identity = trace["environment"]["guarded_source_sha256"]
            if shared_source_identity is None:
                shared_source_identity = source_identity
            require(source_identity == shared_source_identity, f"{path}: guarded source identity differs")
            cache_roots.append(trace["environment"]["vllm_cache_root"])
            traces.append(trace)
            inputs.append({"path": trace_name, "sha256": sha256(path)})
        require(
            len({item["sha256"] for item in inputs}) == len(inputs),
            f"{variant_spec['id']}: repeated trace content is not independent evidence",
        )
        for trace_name in variant_spec.get("challenger_traces", []):
            require(variant_spec["role"] == "baseline", f"{variant_spec['id']}: only a baseline may have challenger traces")
            path = resolve_trace(spec_path, trace_name)
            trace = load_json(path)
            validate_trace(trace, variant_spec, path, qualified=False)
            challenger_identity = {
                "model": trace["model"],
                "gpu": trace["environment"]["gpu"],
                "vllm": trace["environment"]["vllm"],
                "torch": trace["environment"]["torch"],
                "cuda": trace["environment"]["cuda"],
                "controls": {name: trace["controls"].get(name) for name in identity_control_names},
                "batch_sizes": sorted(curve_by_batch(trace)),
            }
            qualified_comparable_identity = {
                key: value for key, value in common_identity.items() if key != "workload_identity"
            }
            require(
                challenger_identity == qualified_comparable_identity,
                f"{path}: challenger workload or environment identity differs",
            )
            challenger_traces.append(trace)
            challenger_inputs.append({"path": trace_name, "sha256": sha256(path), "use": "BASELINE_ENVELOPE_VETO_ONLY"})
        lifecycle = variant_spec.get("lifecycle", {})
        lifecycle_inputs = bind_inputs(spec_path, lifecycle.get("evidence", []))
        if variant_spec["role"] == "candidate":
            if lifecycle.get("requires_prepack"):
                require(lifecycle.get("prepack_ready") is True, f"{variant_spec['id']}: prepack is not lifecycle-ready")
            if lifecycle.get("requires_jit_warm"):
                require(lifecycle.get("jit_warm_ready") is True, f"{variant_spec['id']}: JIT warmup is not lifecycle-ready")
            if lifecycle.get("requires_prepack") or lifecycle.get("requires_jit_warm"):
                require(lifecycle_inputs, f"{variant_spec['id']}: lifecycle readiness evidence is missing")
        variants[variant_spec["id"]] = {
            "spec": variant_spec,
            "traces": traces,
            "challenger_traces": challenger_traces,
            "inputs": inputs,
            "challenger_inputs": challenger_inputs,
            "lifecycle_inputs": lifecycle_inputs,
        }

    if spec["require_distinct_vllm_cache_roots"]:
        require(len(cache_roots) == len(set(cache_roots)), "VLLM_CACHE_ROOT values are not isolated across traces")

    routes = []
    evidence = {}
    for contract in spec["contracts"]:
        contract_inputs = bind_inputs(spec_path, contract.get("evidence", []))
        if contract_inputs:
            evidence[f"contract:{contract['id']}"] = {
                "contract_inputs": contract_inputs,
            }
        require(contract["baseline"] in variants, f"{contract['id']}: baseline variant is unknown")
        require(variants[contract["baseline"]]["spec"]["role"] == "baseline", f"{contract['id']}: baseline role is invalid")
        require(contract["id"] in variants[contract["baseline"]]["spec"]["supported_contracts"], f"{contract['id']}: baseline does not support contract")
        for candidate_id in contract["candidates"]:
            require(candidate_id in variants, f"{contract['id']}: candidate variant {candidate_id} is unknown")
            require(variants[candidate_id]["spec"]["role"] == "candidate", f"{contract['id']}: {candidate_id} is not a candidate")
            require(contract["id"] in variants[candidate_id]["spec"]["supported_contracts"], f"{candidate_id}: numerical contract {contract['id']} is unsupported")
        baseline = variants[contract["baseline"]]
        baseline_curves = [
            curve_by_batch(trace)
            for trace in baseline["traces"] + baseline["challenger_traces"]
        ]
        batch_sizes = sorted(baseline_curves[0])
        candidate_rows: dict[str, dict[int, dict]] = {}
        for candidate_id in contract["candidates"]:
            candidate = variants[candidate_id]
            correctness = compare_outputs(baseline["traces"], candidate["traces"])
            candidate_curves = [curve_by_batch(trace) for trace in candidate["traces"]]
            rows = {}
            for batch_size in batch_sizes:
                baseline_tps = max(
                    float(curve[batch_size]["median_aggregate_output_tokens_per_second"])
                    for curve in baseline_curves
                )
                candidate_tps = min(
                    float(curve[batch_size]["median_aggregate_output_tokens_per_second"])
                    for curve in candidate_curves
                )
                rows[batch_size] = {
                    "baseline_conservative_tps": baseline_tps,
                    "candidate_conservative_tps": candidate_tps,
                    "conservative_speedup": candidate_tps / baseline_tps,
                    "correctness": correctness[batch_size],
                }
            candidate_rows[candidate_id] = rows
            evidence[f"{contract['id']}:{candidate_id}"] = {
                "baseline_inputs": baseline["inputs"],
                "baseline_challenger_inputs": baseline["challenger_inputs"],
                "candidate_inputs": candidate["inputs"],
                "candidate_lifecycle_inputs": candidate["lifecycle_inputs"],
                "by_batch": {str(key): value for key, value in rows.items()},
            }

        for batch_size in batch_sizes:
            eligible = []
            rejection_reasons = []
            for candidate_id, rows in candidate_rows.items():
                row = rows[batch_size]
                if row["conservative_speedup"] < contract["minimum_speedup"]:
                    rejection_reasons.append(f"{candidate_id}: speedup below threshold")
                    continue
                if contract["requires_stock_token_identity"] and not row["correctness"]["all_requests_exact"]:
                    rejection_reasons.append(f"{candidate_id}: stock-token identity failed")
                    continue
                eligible.append((row["conservative_speedup"], candidate_id, row))
            if eligible:
                _, selected, selected_row = max(eligible)
                reason = "candidate clears conservative speedup and numerical-contract gates"
            else:
                selected = contract["baseline"]
                selected_row = None
                reason = (
                    "; ".join(rejection_reasons)
                    or contract.get("baseline_only_reason")
                    or "no candidate admitted by the frozen numerical contract"
                )
            route = {
                "numerical_contract": contract["id"],
                "active_decode_batch": batch_size,
                "route": selected,
                "reason": reason,
            }
            if selected_row is not None:
                route["measured_conservative_speedup_over_baseline"] = selected_row["conservative_speedup"]
            routes.append(route)

    result = {
        "schema_version": "serving-policy-v1",
        "status": "BOUNDED_MEASURED_POLICY_NOT_GLOBAL_OPTIMUM",
        "deployment_point": spec["deployment_point"],
        "policy_input": {"path": spec_path.name, "sha256": sha256(spec_path)},
        "identity": common_identity,
        "global_optimum_proven": False,
        "routes": routes,
        "evidence": evidence,
        "hard_gates": [
            "all qualification traces PASS and share model, hardware, software, workload and batch identities",
            "qualification-trace guarded source hashes are present and identical",
            "VLLM_CACHE_ROOT expected/actual identity is proven and roots are isolated",
            "runtime switches match the intended implementation",
            "candidate prepack and JIT lifecycle requirements are ready before service",
            "candidate selection uses minimum candidate throughput over maximum baseline throughput",
            "older or less-qualified baseline evidence may raise the bar or veto a candidate but can never promote one",
            "stock-token identity is checked from paired measured generated token IDs",
        ],
        "limitations": spec["limitations"],
    }
    output_schema = load_json(ROOT / "schemas" / "serving_policy.schema.json")
    output_errors = validate_instance(result, output_schema)
    if output_errors:
        raise ValueError("derived serving policy validation failed:\n" + "\n".join(output_errors))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = derive(args.spec.resolve())
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
