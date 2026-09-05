#!/usr/bin/env python3
"""Compare constexpr versus runtime extent specialization on the SM89 proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

from benchmark_gate_fusion_proxy import _load_candidate, _reference


def _launch(module, tensors, outputs, *, block, warps):
    a, b, a_log, dt_bias = tensors
    g, beta = outputs
    n_elements = a.numel()
    module._prepare_gates[(triton.cdiv(n_elements, block),)](
        a, b, a_log, dt_bias, g, beta,
        n_elements=n_elements,
        n_heads=a_log.numel(),
        BLOCK=block,
        num_warps=warps,
    )
    return g, beta


def _measure(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _cache_specializations(kernel):
    total = 0
    per_device = {}
    for device, cache in kernel.device_caches.items():
        # Triton 3.7 stores a five-item device tuple; item zero is the
        # compiled-kernel dictionary keyed by specialization signature.
        count = len(cache[0])
        per_device[str(device)] = count
        total += count
    return {"total": total, "per_device": per_device}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constexpr-candidate", required=True, type=Path)
    parser.add_argument("--runtime-candidate", required=True, type=Path)
    parser.add_argument("--screening-set", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    constexpr_path = args.constexpr_candidate.resolve()
    runtime_path = args.runtime_candidate.resolve()
    manifest = json.loads(args.screening_set.read_text(encoding="utf-8"))
    constexpr = _load_candidate(constexpr_path)
    runtime = _load_candidate(runtime_path)
    heads = int(manifest["selection_policy"]["heads"])
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    inputs = []
    for case in manifest["cases"]:
        total = int(case["total_seq_len"])
        tensors = (
            torch.randn((total, heads), device="cuda", dtype=torch.bfloat16, generator=generator),
            torch.randn((total, heads), device="cuda", dtype=torch.bfloat16, generator=generator),
            torch.randn((heads,), device="cuda", dtype=torch.float32, generator=generator) * 0.5,
            torch.randn((heads,), device="cuda", dtype=torch.float32, generator=generator) * 0.5,
        )
        expected = _reference(*tensors)
        outputs = (torch.empty_like(expected[0]), torch.empty_like(expected[1]))
        inputs.append((case, tensors, expected, outputs))

    compile_start = time.perf_counter()
    for _, tensors, _, outputs in inputs:
        _launch(constexpr, tensors, outputs, block=128, warps=1)
    torch.cuda.synchronize()
    constexpr_compile_path_seconds = time.perf_counter() - compile_start
    constexpr_cache = _cache_specializations(constexpr._prepare_gates)

    compile_start = time.perf_counter()
    for _, tensors, _, outputs in inputs:
        _launch(runtime, tensors, outputs, block=128, warps=1)
    torch.cuda.synchronize()
    runtime_compile_path_seconds = time.perf_counter() - compile_start
    runtime_cache = _cache_specializations(runtime._prepare_gates)

    rows = []
    all_pass = True
    for case, tensors, expected, outputs in inputs:
        methods = [
            ("constexpr", lambda: _launch(constexpr, tensors, outputs, block=128, warps=1)),
            ("runtime", lambda: _launch(runtime, tensors, outputs, block=128, warps=1)),
        ]
        for _ in range(20):
            for _, fn in methods:
                fn()
        torch.cuda.synchronize()
        samples = {name: [] for name, _ in methods}
        for trial in range(args.trials):
            order = methods if trial % 2 == 0 else list(reversed(methods))
            for name, fn in order:
                samples[name].append(_measure(fn, args.iterations))
        actual = methods[1][1]()
        torch.cuda.synchronize()
        max_abs = max(float((x - y).abs().max().item()) for x, y in zip(actual, expected))
        passed = max_abs <= 1e-5
        all_pass &= passed
        medians = {name: statistics.median(values) for name, values in samples.items()}
        rows.append({
            **case,
            "correctness": {"status": "PASS" if passed else "FAIL", "max_abs": max_abs},
            "latency_us": {
                name: {"median": medians[name], "samples": samples[name]}
                for name in samples
            },
            "runtime_over_constexpr_ratio": medians["runtime"] / medians["constexpr"],
        })

    mean_latency = {
        name: statistics.fmean(row["latency_us"][name]["median"] for row in rows)
        for name in ("constexpr", "runtime")
    }
    result = {
        "schema_version": "extent-specialization-sm89-proxy-v1",
        "status": "PASS" if all_pass else "FAIL",
        "claim_scope": "DISCOVERY_ONLY_SM89_PROXY_NOT_SM120_OPERATOR_QUALIFICATION",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "sources": {
            "constexpr": {"path": str(constexpr_path), "sha256": hashlib.sha256(constexpr_path.read_bytes()).hexdigest()},
            "runtime": {"path": str(runtime_path), "sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest()},
        },
        "fixed_launch": {"BLOCK": 128, "num_warps": 1},
        "compile_path": {
            "note": "process-local JIT cache counts are authoritative; wall time can benefit from Triton's persistent disk cache",
            "constexpr_seconds": constexpr_compile_path_seconds,
            "runtime_seconds": runtime_compile_path_seconds,
            "constexpr_specializations": constexpr_cache,
            "runtime_specializations": runtime_cache,
        },
        "measurement": {
            "timer": "CUDA events",
            "design": "paired alternating AB/BA",
            "iterations": args.iterations,
            "trials": args.trials,
        },
        "cases": rows,
        "mean_latency_us": mean_latency,
        "runtime_over_constexpr_ratio": mean_latency["runtime"] / mean_latency["constexpr"],
        "decision": "PREFER_RUNTIME_EXTENT" if mean_latency["runtime"] <= mean_latency["constexpr"] * 1.02 and runtime_cache["total"] < constexpr_cache["total"] else "DEFER",
        "limitations": [
            "persistent compiler cache makes wall-clock values non-cold",
            "SM89 launch choice must be remeasured on SM120",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "compile_path": result["compile_path"],
        "mean_latency_us": mean_latency,
        "runtime_over_constexpr_ratio": result["runtime_over_constexpr_ratio"],
        "decision": result["decision"],
        "output": str(args.output),
    }, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
