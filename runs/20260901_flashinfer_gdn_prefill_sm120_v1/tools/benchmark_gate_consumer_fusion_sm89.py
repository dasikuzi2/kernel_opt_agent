#!/usr/bin/env python3
"""Causal SM89 proxy for materialized versus directly consumed gate values."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

from benchmark_gate_fusion_proxy import _load_candidate


@triton.jit
def _consume_materialized(g_ptr, beta_ptr, x_ptr, y_ptr, out_ptr,
                          n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(g_ptr + offsets, mask=mask)
    beta = tl.load(beta_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(out_ptr + offsets, g * x + beta * y, mask=mask)


@triton.jit
def _prepare_consume_fused(a_ptr, b_ptr, a_log_ptr, dt_bias_ptr,
                           x_ptr, y_ptr, out_ptr, n_elements,
                           n_heads: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    heads = offsets % n_heads
    a = tl.load(a_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask).to(tl.float32)
    a_log = tl.load(a_log_ptr + heads, mask=mask).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + heads, mask=mask).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    shifted = a + dt_bias
    softplus = tl.maximum(shifted, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(shifted)))
    g = tl.exp(-tl.exp(a_log) * softplus)
    beta = 1.0 / (1.0 + tl.exp(-b))
    tl.store(out_ptr + offsets, g * x + beta * y, mask=mask)


def _reference(a, b, a_log, dt_bias, x, y):
    g = torch.exp(-torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias.float()))
    beta = torch.sigmoid(b.float())
    return g * x.float() + beta * y.float()


def _materialized(module, tensors, buffers):
    a, b, a_log, dt_bias, x, y = tensors
    g, beta, materialized_out, _ = buffers
    n_elements = a.numel()
    grid = (triton.cdiv(n_elements, 256),)
    module._prepare_gates[grid](
        a, b, a_log, dt_bias, g, beta,
        n_elements=n_elements, n_heads=a_log.numel(), BLOCK=256,
    )
    _consume_materialized[grid](g, beta, x, y, materialized_out, n_elements, BLOCK=256)
    return materialized_out


def _fused(tensors, buffers):
    a, b, a_log, dt_bias, x, y = tensors
    _, _, _, fused_out = buffers
    n_elements = a.numel()
    _prepare_consume_fused[(triton.cdiv(n_elements, 256),)](
        a, b, a_log, dt_bias, x, y, fused_out, n_elements,
        n_heads=a_log.numel(), BLOCK=256,
    )
    return fused_out


def _measure(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _active_samples(fn, repeats):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recording:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    samples = defaultdict(list)
    for event in recording.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            samples[event.name].append(float(event.self_device_time_total))
    return dict(samples)


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
            torch.randn((total, heads), device="cuda", dtype=torch.bfloat16, generator=generator),
            torch.randn((total, heads), device="cuda", dtype=torch.bfloat16, generator=generator),
        )
        expected = _reference(*tensors)
        buffers = (
            torch.empty_like(expected),
            torch.empty_like(expected),
            torch.empty_like(expected),
            torch.empty_like(expected),
        )
        materialized = lambda: _materialized(module, tensors, buffers)
        fused = lambda: _fused(tensors, buffers)
        for _ in range(args.warmup):
            materialized()
            fused()
        torch.cuda.synchronize()
        materialized_result = materialized().clone()
        fused_result = fused().clone()
        torch.cuda.synchronize()
        materialized_error = float((materialized_result - expected).abs().max().item())
        fused_error = float((fused_result - expected).abs().max().item())
        cross_error = float((fused_result - materialized_result).abs().max().item())
        passed = materialized_error <= 1e-5 and fused_error <= 1e-5 and cross_error <= 1e-5
        all_pass &= passed

        samples = {"materialized": [], "fused": []}
        for trial in range(args.trials):
            order = [("materialized", materialized), ("fused", fused)]
            if trial % 2:
                order.reverse()
            for name, fn in order:
                samples[name].append(_measure(fn, args.iterations))
        medians = {name: statistics.median(values) for name, values in samples.items()}
        active_materialized = _active_samples(materialized, args.profile_repeats)
        active_fused = _active_samples(fused, args.profile_repeats)
        materialized_active_samples = [
            gate + consumer
            for gate, consumer in zip(
                active_materialized.get("_prepare_gates", []),
                active_materialized.get("_consume_materialized", []),
            )
        ]
        fused_active_samples = active_fused.get("_prepare_consume_fused", [])
        if len(materialized_active_samples) != args.profile_repeats or len(fused_active_samples) != args.profile_repeats:
            raise RuntimeError("CUPTI did not capture the expected producer-consumer kernel events")
        materialized_active = statistics.median(materialized_active_samples)
        fused_active = statistics.median(fused_active_samples)
        sink = float(fused_result[0, 0].item()) + float(fused_result[-1, -1].item())
        rows.append({
            **case,
            "n_elements": total * heads,
            "correctness": {
                "status": "PASS" if passed else "FAIL",
                "atol": 1e-5,
                "rtol": 1e-5,
                "materialized_max_abs": materialized_error,
                "fused_max_abs": fused_error,
                "cross_max_abs": cross_error,
                "live_output_sink": sink,
            },
            "effective_timeline_us": {
                "materialized": {"median": medians["materialized"], "samples": samples["materialized"]},
                "fused": {"median": medians["fused"], "samples": samples["fused"]},
            },
            "cupti_active_us": {
                "materialized_two_kernel_sum": {"median": materialized_active, "samples": materialized_active_samples},
                "fused_one_kernel": {"median": fused_active, "samples": fused_active_samples},
            },
            "speedup": {
                "effective_timeline": medians["materialized"] / medians["fused"],
                "cupti_active": materialized_active / fused_active,
            },
            "logical_intermediate_traffic_avoided_bytes": total * heads * 16,
        })

    effective_speedups = [row["speedup"]["effective_timeline"] for row in rows]
    active_speedups = [row["speedup"]["cupti_active"] for row in rows]
    result = {
        "schema_version": "gate-consumer-fusion-sm89-proxy-v1",
        "status": "PASS" if all_pass else "FAIL",
        "claim_scope": "DISCOVERY_ONLY_SYNTHETIC_CONSUMER_SM89_NOT_GDN_OR_SM120_QUALIFICATION",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "causal_question": "Does directly consuming gate values remove material launch and intermediate-tensor cost under matched pointwise geometry?",
        "source_identities": {
            "gate_candidate": {"path": str(candidate_path), "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest()},
            "screening_set": {"path": str(screening_path), "sha256": hashlib.sha256(screening_path.read_bytes()).hexdigest()},
            "experiment_source": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "controls": {
            "logical_work": "out = exp(-exp(A_log)*softplus(a+dt_bias))*x + sigmoid(b)*y",
            "matched": ["inputs", "outputs", "BLOCK=256", "grid", "dtype", "number of logical elements"],
            "changed": "materialize g/beta through a second kernel versus compute and consume them in one kernel",
            "dce_protection": "output correctness plus first/last-element live sink",
            "intermediate_traffic_formula": "2 FP32 tensors * (one logical write + one logical read) = 16 bytes/element; this is logical traffic, not a DRAM claim",
        },
        "measurement": {
            "effective_timer": "CUDA events around CPU-paced launches",
            "active_timer": "CUPTI per-kernel activity, collected separately",
            "order": "paired alternating AB/BA",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trials": args.trials,
            "profile_repeats": args.profile_repeats,
            "stop_rule": "one fixed architecture A/B over the six frozen shapes; no parameter expansion",
        },
        "cases": rows,
        "aggregate": {
            "mean_effective_speedup": statistics.fmean(effective_speedups),
            "median_effective_speedup": statistics.median(effective_speedups),
            "mean_cupti_active_speedup": statistics.fmean(active_speedups),
            "median_cupti_active_speedup": statistics.median(active_speedups),
            "materialized_effective_mean_us": statistics.fmean(row["effective_timeline_us"]["materialized"]["median"] for row in rows),
            "fused_effective_mean_us": statistics.fmean(row["effective_timeline_us"]["fused"]["median"] for row in rows),
            "materialized_active_mean_us": statistics.fmean(row["cupti_active_us"]["materialized_two_kernel_sum"]["median"] for row in rows),
            "fused_active_mean_us": statistics.fmean(row["cupti_active_us"]["fused_one_kernel"]["median"] for row in rows),
        },
        "decision_rule": {
            "promote_mechanism_if": "all correctness passes and both mean effective and CUPTI-active speedups exceed 1.05x",
            "decision": "PROMOTE_DIRECT_CONSUMPTION_MECHANISM" if all_pass and statistics.fmean(effective_speedups) > 1.05 and statistics.fmean(active_speedups) > 1.05 else "DO_NOT_PROMOTE",
        },
        "allowed_claims": [
            "direct producer-consumer fusion benefits this matched synthetic pointwise consumer on SM89",
            "the removed logical intermediate traffic and launch topology are candidate mechanisms for target-GPU validation",
        ],
        "forbidden_claims": [
            "the recurrent GDN consumer has the same dependency, register pressure, occupancy or speedup",
            "SM120 end-to-end speedup, SOTA or theoretical-optimum distance",
            "logical intermediate bytes reached DRAM",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "decision": result["decision_rule"]["decision"], "output": str(args.output)}, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
