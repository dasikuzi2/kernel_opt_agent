#!/usr/bin/env python3
"""Correctness-check and screen the exact fused-gate candidate on a CUDA GPU.

This isolates the GDN gate-preprocessing stage so a non-SM120 GPU can reject or
retain the fusion hypothesis.  Results are discovery-only and cannot qualify
the complete GDN operator on another architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from torch.profiler import ProfilerActivity, profile


def _load_candidate(path: Path):
    # The candidate's gate kernel is exact, but its full wrapper imports an
    # SM120-only FlashInfer entrypoint.  Stub only that unused symbol so the
    # same source file can be loaded on SM89 without pretending the full
    # operator is supported.
    flashinfer = types.ModuleType("flashinfer")
    gdn_prefill = types.ModuleType("flashinfer.gdn_prefill")
    gdn_prefill.chunk_gated_delta_rule = lambda **_: None
    flashinfer.gdn_prefill = gdn_prefill
    previous = {
        name: sys.modules.get(name)
        for name in ("flashinfer", "flashinfer.gdn_prefill")
    }
    sys.modules["flashinfer"] = flashinfer
    sys.modules["flashinfer.gdn_prefill"] = gdn_prefill
    try:
        spec = importlib.util.spec_from_file_location("screened_gate_candidate", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load candidate from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, module_before in previous.items():
            if module_before is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module_before
    if not hasattr(module, "_prepare_gates"):
        raise AttributeError("candidate does not expose _prepare_gates")
    return module


def _reference(a, b, a_log, dt_bias):
    log_g = -torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias.float())
    return torch.exp(log_g), torch.sigmoid(b.float())


def _candidate(module, a, b, a_log, dt_bias):
    g = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    beta = torch.empty(b.shape, dtype=torch.float32, device=b.device)
    n_elements = a.numel()
    module._prepare_gates[(triton.cdiv(n_elements, 256),)](
        a, b, a_log, dt_bias, g, beta,
        n_elements=n_elements,
        n_heads=a_log.numel(),
        BLOCK=256,
    )
    return g, beta


def _measure_once(fn, *, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _measure_pair(eager_fn, fused_fn, *, warmup: int, iterations: int,
                  trials: int) -> tuple[list[float], list[float]]:
    for index in range(warmup):
        (eager_fn if index % 2 == 0 else fused_fn)()
    torch.cuda.synchronize()
    eager_samples = []
    fused_samples = []
    for trial in range(trials):
        order = ((eager_fn, eager_samples), (fused_fn, fused_samples))
        if trial % 2:
            order = tuple(reversed(order))
        for fn, samples in order:
            samples.append(_measure_once(fn, iterations=iterations))
    return eager_samples, fused_samples


def _profile_cuda_kernels(fn, *, repeats: int = 3) -> dict:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recording:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    kernels = [event for event in recording.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    names = Counter(event.name for event in kernels)
    return {
        "repeats": repeats,
        "cuda_kernel_events": len(kernels),
        "kernels_per_invocation": len(kernels) / repeats,
        "unique_kernel_names": len(names),
        "kernel_name_counts": [
            {"name": name[:240], "count": count}
            for name, count in names.most_common()
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--screening-set", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    candidate_path = args.candidate.resolve()
    screening_path = args.screening_set.resolve()
    module = _load_candidate(candidate_path)
    manifest = json.loads(screening_path.read_text(encoding="utf-8"))
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rows = []
    all_pass = True
    for case in manifest["cases"]:
        total = int(case["total_seq_len"])
        heads = int(manifest["selection_policy"]["heads"])
        a = torch.randn((total, heads), device="cuda", dtype=torch.bfloat16, generator=generator)
        b = torch.randn((total, heads), device="cuda", dtype=torch.bfloat16, generator=generator)
        a_log = torch.randn((heads,), device="cuda", dtype=torch.float32, generator=generator) * 0.5
        dt_bias = torch.randn((heads,), device="cuda", dtype=torch.float32, generator=generator) * 0.5

        expected_g, expected_beta = _reference(a, b, a_log, dt_bias)
        actual_g, actual_beta = _candidate(module, a, b, a_log, dt_bias)
        torch.cuda.synchronize()
        g_abs = (actual_g - expected_g).abs()
        beta_abs = (actual_beta - expected_beta).abs()
        matched = torch.isclose(actual_g, expected_g, atol=1e-5, rtol=1e-5).all()
        matched &= torch.isclose(actual_beta, expected_beta, atol=1e-5, rtol=1e-5).all()
        passed = bool(matched.item())
        all_pass &= passed

        eager_samples, fused_samples = _measure_pair(
            lambda: _reference(a, b, a_log, dt_bias),
            lambda: _candidate(module, a, b, a_log, dt_bias),
            warmup=args.warmup,
            iterations=args.iterations,
            trials=args.trials,
        )
        eager_median = statistics.median(eager_samples)
        fused_median = statistics.median(fused_samples)
        paired_speedups = [eager / fused for eager, fused in zip(eager_samples, fused_samples)]
        rows.append({
            **case,
            "correctness": {
                "status": "PASS" if passed else "FAIL",
                "atol": 1e-5,
                "rtol": 1e-5,
                "max_abs_g": float(g_abs.max().item()),
                "max_abs_beta": float(beta_abs.max().item()),
            },
            "eager_us": {"median": eager_median, "samples": eager_samples},
            "fused_us": {"median": fused_median, "samples": fused_samples},
            "speedup": statistics.median(paired_speedups),
            "paired_speedup_samples": paired_speedups,
        })

    # Exercise numerically awkward values independently of the random timing
    # inputs: negative softplus tails, saturated sigmoid inputs and a broad
    # learned decay range.
    stress_elements = 4096 * heads
    stress_a = torch.linspace(-20.0, 20.0, stress_elements, device="cuda").reshape(4096, heads).to(torch.bfloat16)
    stress_b = torch.linspace(20.0, -20.0, stress_elements, device="cuda").reshape(4096, heads).to(torch.bfloat16)
    stress_a_log = torch.linspace(-5.0, 5.0, heads, device="cuda", dtype=torch.float32)
    stress_bias = torch.linspace(-10.0, 10.0, heads, device="cuda", dtype=torch.float32)
    expected_g, expected_beta = _reference(stress_a, stress_b, stress_a_log, stress_bias)
    actual_g, actual_beta = _candidate(module, stress_a, stress_b, stress_a_log, stress_bias)
    torch.cuda.synchronize()
    stress_g_abs = (actual_g - expected_g).abs()
    stress_beta_abs = (actual_beta - expected_beta).abs()
    stress_pass = bool((
        torch.isclose(actual_g, expected_g, atol=1e-5, rtol=1e-5).all()
        & torch.isclose(actual_beta, expected_beta, atol=1e-5, rtol=1e-5).all()
    ).item())
    all_pass &= stress_pass
    mechanism = {
        "profile_case": "last screening case; launch topology is shape-invariant for this stage",
        "eager": _profile_cuda_kernels(lambda: _reference(a, b, a_log, dt_bias)),
        "fused": _profile_cuda_kernels(lambda: _candidate(module, a, b, a_log, dt_bias)),
    }

    result = {
        "schema_version": "gate-fusion-proxy-screen-v1",
        "status": "PASS" if all_pass else "FAIL",
        "claim_scope": "DISCOVERY_ONLY_SM89_PROXY_NOT_SM120_OPERATOR_QUALIFICATION",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_identities": {
            "candidate": {
                "path": str(candidate_path),
                "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            },
            "screening_set": {
                "path": str(screening_path),
                "sha256": hashlib.sha256(screening_path.read_bytes()).hexdigest(),
            },
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "measurement": {
            "timer": "CUDA events",
            "design": "paired, alternating AB/BA trial order",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trials": args.trials,
            "statistic": "median trial microseconds per invocation",
        },
        "cases": rows,
        "aggregate": {
            "mean_case_speedup": statistics.fmean(row["speedup"] for row in rows),
            "median_case_speedup": statistics.median(row["speedup"] for row in rows),
            "eager_mean_us": statistics.fmean(row["eager_us"]["median"] for row in rows),
            "fused_mean_us": statistics.fmean(row["fused_us"]["median"] for row in rows),
        },
        "numerical_stress": {
            "status": "PASS" if stress_pass else "FAIL",
            "ranges": {
                "a": [-20.0, 20.0],
                "b": [-20.0, 20.0],
                "A_log": [-5.0, 5.0],
                "dt_bias": [-10.0, 10.0],
            },
            "atol": 1e-5,
            "rtol": 1e-5,
            "max_abs_g": float(stress_g_abs.max().item()),
            "max_abs_beta": float(stress_beta_abs.max().item()),
        },
        "mechanism": mechanism,
        "limitations": [
            "synthetic values with frozen production shapes; tensor blobs are not present locally",
            "isolated gate preprocessing only; the SM120 recurrent GDN kernel is not executed",
            "SM89 result can reject a broken fusion but cannot establish SM120 speedup",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
