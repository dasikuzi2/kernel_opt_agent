#!/usr/bin/env python3
"""Bounded SM89 learning experiment for the isolated gate-preprocessing stage.

This compares eager PyTorch, dynamic torch.compile, and the exact handwritten
Triton candidate, then screens a fixed launch-configuration matrix.  It is a
discovery-only experiment: launch choices and absolute timings do not qualify
the complete SM120 operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
from torch.profiler import ProfilerActivity, profile

from benchmark_gate_fusion_proxy import _candidate, _load_candidate, _reference


def _measure_once(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _measure_rotating(methods, *, warmup: int, iterations: int, trials: int):
    for _ in range(warmup):
        for _, fn in methods:
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name, _ in methods}
    for trial in range(trials):
        offset = trial % len(methods)
        order = methods[offset:] + methods[:offset]
        for name, fn in order:
            samples[name].append(_measure_once(fn, iterations))
    return samples


def _profile(fn, repeats: int = 3):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recording:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    kernels = [
        event for event in recording.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    names = Counter(event.name for event in kernels)
    return {
        "repeats": repeats,
        "cuda_kernel_events": len(kernels),
        "kernels_per_invocation": len(kernels) / repeats,
        "kernel_name_counts": [
            {"name": name[:240], "count": count}
            for name, count in names.most_common()
        ],
    }


def _configured_candidate(module, a, b, a_log, dt_bias, g, beta, block, warps):
    n_elements = a.numel()
    module._prepare_gates[(triton.cdiv(n_elements, block),)](
        a, b, a_log, dt_bias, g, beta,
        n_elements=n_elements,
        n_heads=a_log.numel(),
        BLOCK=block,
        num_warps=warps,
    )
    return g, beta


def _max_abs_pair(actual, expected):
    return max(
        float((actual_value - expected_value).abs().max().item())
        for actual_value, expected_value in zip(actual, expected)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--screening-set", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    candidate_path = args.candidate.resolve()
    screening_path = args.screening_set.resolve()
    module = _load_candidate(candidate_path)
    manifest = json.loads(screening_path.read_text(encoding="utf-8"))
    heads = int(manifest["selection_policy"]["heads"])
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    def compile_source(a, b, a_log, dt_bias):
        return _reference(a, b, a_log, dt_bias)

    compiled = torch.compile(
        compile_source,
        fullgraph=True,
        dynamic=True,
        mode="default",
    )
    inputs = []
    for case in manifest["cases"]:
        total = int(case["total_seq_len"])
        inputs.append((
            case,
            torch.randn((total, heads), device="cuda", dtype=torch.bfloat16, generator=generator),
            torch.randn((total, heads), device="cuda", dtype=torch.bfloat16, generator=generator),
            torch.randn((heads,), device="cuda", dtype=torch.float32, generator=generator) * 0.5,
            torch.randn((heads,), device="cuda", dtype=torch.float32, generator=generator) * 0.5,
        ))

    # Compile all frozen shapes before steady-state measurement.  This wall
    # time is separately reported because agent throughput matters too.
    compile_start = time.perf_counter()
    for _, a, b, a_log, dt_bias in inputs:
        compiled(a, b, a_log, dt_bias)
    torch.cuda.synchronize()
    compile_wall_seconds = time.perf_counter() - compile_start

    comparison_rows = []
    all_pass = True
    for case, a, b, a_log, dt_bias in inputs:
        expected = _reference(a, b, a_log, dt_bias)
        manual = _candidate(module, a, b, a_log, dt_bias)
        automatic = compiled(a, b, a_log, dt_bias)
        torch.cuda.synchronize()
        manual_error = _max_abs_pair(manual, expected)
        compiled_error = _max_abs_pair(automatic, expected)
        passed = manual_error <= 1e-5 and compiled_error <= 1e-5
        all_pass &= passed
        methods = [
            ("eager", lambda: _reference(a, b, a_log, dt_bias)),
            ("compiled", lambda: compiled(a, b, a_log, dt_bias)),
            ("manual", lambda: _candidate(module, a, b, a_log, dt_bias)),
        ]
        samples = _measure_rotating(
            methods,
            warmup=args.warmup,
            iterations=args.iterations,
            trials=args.trials,
        )
        medians = {name: statistics.median(values) for name, values in samples.items()}
        comparison_rows.append({
            **case,
            "correctness": {
                "status": "PASS" if passed else "FAIL",
                "atol": 1e-5,
                "rtol": 1e-5,
                "manual_max_abs": manual_error,
                "compiled_max_abs": compiled_error,
            },
            "latency_us": {
                name: {"median": medians[name], "samples": samples[name]}
                for name in ("eager", "compiled", "manual")
            },
            "speedup_vs_eager": {
                "compiled": medians["eager"] / medians["compiled"],
                "manual": medians["eager"] / medians["manual"],
            },
            "manual_speedup_vs_compiled": medians["compiled"] / medians["manual"],
        })

    profile_case, a, b, a_log, dt_bias = inputs[-1]
    mechanism = {
        "profile_case": profile_case,
        "eager": _profile(lambda: _reference(a, b, a_log, dt_bias)),
        "compiled": _profile(lambda: compiled(a, b, a_log, dt_bias)),
        "manual": _profile(lambda: _candidate(module, a, b, a_log, dt_bias)),
    }

    # Fixed, predeclared matrix.  Outputs are preallocated so this phase asks
    # only about launch geometry, not Python allocation overhead.
    matrix = [(block, warps) for block in (64, 128, 256, 512) for warps in (1, 2, 4)]
    sweep_rows = []
    for case, a, b, a_log, dt_bias in inputs:
        expected = _reference(a, b, a_log, dt_bias)
        g = torch.empty_like(expected[0])
        beta = torch.empty_like(expected[1])
        variants = []
        correctness = {}
        for block, warps in matrix:
            name = f"b{block}_w{warps}"
            fn = lambda block=block, warps=warps: _configured_candidate(
                module, a, b, a_log, dt_bias, g, beta, block, warps
            )
            actual = fn()
            torch.cuda.synchronize()
            error = _max_abs_pair(actual, expected)
            correctness[name] = {"max_abs": error, "status": "PASS" if error <= 1e-5 else "FAIL"}
            all_pass &= error <= 1e-5
            variants.append((name, fn))
        samples = _measure_rotating(
            variants,
            warmup=max(5, args.warmup // 3),
            iterations=args.iterations,
            trials=args.trials,
        )
        medians = {name: statistics.median(values) for name, values in samples.items()}
        winner = min(medians, key=medians.get)
        sweep_rows.append({
            **case,
            "correctness": correctness,
            "latency_us": {
                name: {"median": medians[name], "samples": samples[name]}
                for name, _ in variants
            },
            "winner": winner,
            "winner_us": medians[winner],
        })

    mean_by_variant = {
        f"b{block}_w{warps}": statistics.fmean(
            row["latency_us"][f"b{block}_w{warps}"]["median"] for row in sweep_rows
        )
        for block, warps in matrix
    }
    global_winner = min(mean_by_variant, key=mean_by_variant.get)
    oracle_mean = statistics.fmean(row["winner_us"] for row in sweep_rows)
    global_mean = mean_by_variant[global_winner]
    oracle_gain_percent = (global_mean / oracle_mean - 1.0) * 100.0
    if oracle_gain_percent < 2.0:
        dispatch_decision = "USE_ONE_GLOBAL_CONFIG"
    elif oracle_gain_percent > 5.0:
        dispatch_decision = "SHAPE_DISPATCH_WARRANTS_SM120_VALIDATION"
    else:
        dispatch_decision = "INCONCLUSIVE_DEFER_TO_SM120"

    comparison_means = {
        name: statistics.fmean(row["latency_us"][name]["median"] for row in comparison_rows)
        for name in ("eager", "compiled", "manual")
    }
    result = {
        "schema_version": "gate-fusion-sm89-learning-v1",
        "status": "PASS" if all_pass else "FAIL",
        "claim_scope": "DISCOVERY_ONLY_SM89_PROXY_NOT_SM120_OPERATOR_QUALIFICATION",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_identities": {
            "candidate": {"path": str(candidate_path), "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest()},
            "screening_set": {"path": str(screening_path), "sha256": hashlib.sha256(screening_path.read_bytes()).hexdigest()},
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "measurement": {
            "timer": "CUDA events",
            "order": "cyclic rotation within each trial",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trials": args.trials,
        },
        "path_comparison": {
            "torch_compile_mode": "default",
            "mode_selection": "bounded two-shape pre-screen rejected reduce-overhead because CUDA Graph output copies increased latency; default matched max-autotune-no-cudagraphs within noise",
            "torch_compile_wall_seconds_for_frozen_shapes": compile_wall_seconds,
            "cases": comparison_rows,
            "mean_latency_us": comparison_means,
            "manual_speedup_vs_compiled_mean_latency": comparison_means["compiled"] / comparison_means["manual"],
            "mechanism": mechanism,
        },
        "launch_sweep": {
            "matrix": [{"block": block, "num_warps": warps} for block, warps in matrix],
            "stop_rule": "exactly one pass over the fixed 4x3 matrix; no adaptive expansion",
            "allocation_policy": "outputs preallocated; measures kernel launch path",
            "cases": sweep_rows,
            "mean_latency_us_by_variant": mean_by_variant,
            "global_winner": global_winner,
            "global_winner_mean_us": global_mean,
            "per_shape_oracle_mean_us": oracle_mean,
            "per_shape_dispatch_gain_percent": oracle_gain_percent,
            "decision": dispatch_decision,
        },
        "transfer_policy": {
            "portable_evidence": [
                "correctness of the fusion equation over the frozen shapes",
                "launch-count reduction mechanism",
                "whether a compiler-generated path removes the same launch topology",
            ],
            "must_be_remeasured_on_sm120": [
                "absolute latency",
                "winning BLOCK and num_warps",
                "full recurrent GDN operator speedup",
            ],
        },
        "known_search_cost_issue": {
            "candidate_n_elements_is_triton_constexpr": True,
            "impact": "distinct tensor sizes may compile distinct kernel specializations",
            "next_test": "compare runtime n_elements after preserving exact math and qualification scope",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "comparison_mean_us": comparison_means,
        "manual_speedup_vs_compiled": result["path_comparison"]["manual_speedup_vs_compiled_mean_latency"],
        "compile_wall_seconds": compile_wall_seconds,
        "launch_global_winner": global_winner,
        "dispatch_gain_percent": oracle_gain_percent,
        "dispatch_decision": dispatch_decision,
        "output": str(args.output),
    }, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
