#!/usr/bin/env python3
"""Separate active-kernel, effective timeline, and allocation costs on SM89."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
from torch.profiler import ProfilerActivity, profile

from benchmark_gate_fusion_proxy import _candidate, _load_candidate, _reference


def _launch_preallocated(module, tensors, outputs):
    a, b, a_log, dt_bias = tensors
    g, beta = outputs
    n_elements = a.numel()
    module._prepare_gates[(triton.cdiv(n_elements, 256),)](
        a, b, a_log, dt_bias, g, beta,
        n_elements=n_elements,
        n_heads=a_log.numel(),
        BLOCK=256,
    )
    return g, beta


def _event_measure(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _host_measure(fn, iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def _active_kernel_samples(fn, repeats):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recording:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    samples = [
        float(event.self_device_time_total)
        for event in recording.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and event.name == "_prepare_gates"
    ]
    if len(samples) != repeats:
        raise RuntimeError(f"expected {repeats} gate events, found {len(samples)}")
    return samples


def _tool_state():
    return {
        "ncu": shutil.which("ncu") or shutil.which("nv-nsight-cu-cli"),
        "nsys": shutil.which("nsys"),
        "fallback": "PyTorch profiler CUPTI CUDA activity events",
    }


def _gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"UNAVAILABLE: {error}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--screening-set", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--profile-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    candidate_path = args.candidate.resolve()
    screening_path = args.screening_set.resolve()
    module = _load_candidate(candidate_path)
    manifest = json.loads(screening_path.read_text(encoding="utf-8"))
    heads = int(manifest["selection_policy"]["heads"])
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rows = []
    all_pass = True

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
        preallocated = lambda: _launch_preallocated(module, tensors, outputs)
        allocated = lambda: _candidate(module, *tensors)
        allocate_only = lambda: (
            torch.empty(tensors[0].shape, dtype=torch.float32, device="cuda"),
            torch.empty(tensors[1].shape, dtype=torch.float32, device="cuda"),
        )

        for _ in range(args.warmup):
            preallocated()
            allocated()
            allocate_only()
        torch.cuda.synchronize()
        actual = preallocated()
        torch.cuda.synchronize()
        max_abs = max(float((x - y).abs().max().item()) for x, y in zip(actual, expected))
        passed = max_abs <= 1e-5
        all_pass &= passed

        samples = {"preallocated": [], "allocated": [], "host_preallocated": [], "host_allocated": [], "host_allocate_only": []}
        for trial in range(args.trials):
            order = [("preallocated", preallocated), ("allocated", allocated)]
            if trial % 2:
                order.reverse()
            for name, fn in order:
                samples[name].append(_event_measure(fn, args.iterations))
                samples[f"host_{name}"].append(_host_measure(fn, args.iterations))
            samples["host_allocate_only"].append(_host_measure(allocate_only, args.iterations))

        active = _active_kernel_samples(preallocated, args.profile_repeats)
        preallocated_median = statistics.median(samples["preallocated"])
        allocated_median = statistics.median(samples["allocated"])
        active_median = statistics.median(active)
        rows.append({
            **case,
            "n_elements": total * heads,
            "correctness": {"status": "PASS" if passed else "FAIL", "max_abs": max_abs, "atol": 1e-5, "rtol": 1e-5},
            "cuda_event_effective_timeline_us": {
                "preallocated": {"median": preallocated_median, "samples": samples["preallocated"]},
                "allocated": {"median": allocated_median, "samples": samples["allocated"]},
            },
            "synchronized_host_wall_us": {
                "preallocated": {"median": statistics.median(samples["host_preallocated"]), "samples": samples["host_preallocated"]},
                "allocated": {"median": statistics.median(samples["host_allocated"]), "samples": samples["host_allocated"]},
                "allocate_only": {"median": statistics.median(samples["host_allocate_only"]), "samples": samples["host_allocate_only"]},
            },
            "cupti_kernel_active_us": {"median": active_median, "samples": active},
            "derived": {
                "allocation_path_delta_us": allocated_median - preallocated_median,
                "effective_minus_active_us": preallocated_median - active_median,
                "active_fraction_of_effective": active_median / preallocated_median,
            },
        })

    result = {
        "schema_version": "gate-cost-stack-sm89-proxy-v1",
        "status": "PASS" if all_pass else "FAIL",
        "claim_scope": "DISCOVERY_ONLY_SM89_PROXY_NOT_SM120_OPERATOR_QUALIFICATION",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_identities": {
            "candidate": {"path": str(candidate_path), "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest()},
            "screening_set": {"path": str(screening_path), "sha256": hashlib.sha256(screening_path.read_bytes()).hexdigest()},
            "experiment_source": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "driver_snapshot": _gpu_snapshot(),
            "display_mode": "Windows WDDM display-attached through WSL2",
        },
        "tool_capability": _tool_state(),
        "measurement": {
            "cuda_event_semantics": "effective GPU timeline around CPU-paced serial launches; may contain idle gaps",
            "cupti_semantics": "per-kernel CUDA activity duration collected separately from timing trials",
            "host_semantics": "synchronized end-to-end loop wall time",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trials": args.trials,
            "profile_repeats": args.profile_repeats,
            "order": "paired alternating AB/BA",
        },
        "cases": rows,
        "aggregate": {
            "preallocated_effective_mean_us": statistics.fmean(row["cuda_event_effective_timeline_us"]["preallocated"]["median"] for row in rows),
            "allocated_effective_mean_us": statistics.fmean(row["cuda_event_effective_timeline_us"]["allocated"]["median"] for row in rows),
            "cupti_active_mean_us": statistics.fmean(row["cupti_kernel_active_us"]["median"] for row in rows),
            "allocation_path_delta_mean_us": statistics.fmean(row["derived"]["allocation_path_delta_us"] for row in rows),
            "active_fraction_of_effective_mean": statistics.fmean(row["derived"]["active_fraction_of_effective"] for row in rows),
        },
        "allowed_claims": [
            "the isolated gate kernel is active for only part of the effective CPU-paced launch timeline on this SM89 WDDM/WSL2 setup",
            "preallocating outputs changes the isolated wrapper path cost by the measured amount",
        ],
        "forbidden_claims": [
            "occupancy, SFU utilization, L2 bandwidth or DRAM bandwidth without Nsight counters",
            "SM120 full-operator latency, SOTA or theoretical-optimum distance",
            "effective-minus-active is pure CUDA launch overhead",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "tool_capability": result["tool_capability"], "output": str(args.output)}, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
