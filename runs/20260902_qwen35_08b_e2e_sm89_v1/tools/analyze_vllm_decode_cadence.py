#!/usr/bin/env python3
"""Bound batch-1 decode work using kernels and CUDA Graph replay activity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def union_duration_ns(intervals: list[tuple[int, int]]) -> int:
    """Return the union length of half-open timestamp intervals."""
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--generated-steps", type=int, required=True)
    parser.add_argument("--request-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    weighted_tpot_ms = sum(
        float(case["weight"]) * float(case["median_tpot_ms"])
        for case in trace["cases"]
    )

    connection = sqlite3.connect(args.sqlite)
    try:
        kernel_activities = connection.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE end > start ORDER BY start"
        ).fetchall()
        graph_traces = connection.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
            "ORDER BY start"
        ).fetchall()
        graph_launches = connection.execute(
            "SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id=r.nameId "
            "WHERE s.value='cudaGraphLaunch_v10000' ORDER BY r.start"
        ).fetchall()
    finally:
        connection.close()

    launch_durations_ms = [(end - start) / 1e6 for start, end in graph_launches]
    launch_cadences_ms = [
        (graph_launches[index][0] - graph_launches[index - 1][0]) / 1e6
        for index in range(1, len(graph_launches))
    ]
    # Inter-request boundaries are much larger than ordinary decode cadence.
    within_request_cadences_ms = [value for value in launch_cadences_ms if value < 10.0]

    graph_trace_durations_ms = [(end - start) / 1e6 for start, end in graph_traces]
    total_kernel_ns = sum(end - start for start, end in kernel_activities)
    kernel_count = len(kernel_activities)
    standalone_kernel_ms_per_generated_step = (
        total_kernel_ns / 1e6 / args.generated_steps
    )
    graph_replay_ms_per_generated_step = (
        sum(graph_trace_durations_ms) / args.generated_steps
    )
    # Nsight's KERNEL table does not expand kernels executed by graph replay in
    # this trace mode. GRAPH_TRACE holds the missing replay body. The sum is an
    # upper bound because KERNEL also includes capture/prefill work and because
    # independent streams may overlap.
    observed_gpu_activity_ms_per_generated_step_upper_bound = (
        standalone_kernel_ms_per_generated_step + graph_replay_ms_per_generated_step
    )
    outside_observed_gpu_activity_ms_per_token_lower_bound = max(
        0.0,
        weighted_tpot_ms - observed_gpu_activity_ms_per_generated_step_upper_bound,
    )
    expected_graph_launches = args.generated_steps - args.request_count
    status = (
        "PASS"
        if len(graph_launches) == expected_graph_launches
        and len(graph_traces) == expected_graph_launches
        else "FAIL"
    )

    # Measure the actual GPU-busy union between adjacent decode graph starts.
    # The <10 ms filter removes the six inter-request boundaries, matching the
    # launch-cadence calculation above. Unlike duration summation, this does not
    # double count overlapping streams.
    all_gpu_activities = kernel_activities + graph_traces
    within_request_windows = [
        (graph_traces[index - 1][0], graph_traces[index][0])
        for index in range(1, len(graph_traces))
        if (graph_traces[index][0] - graph_traces[index - 1][0]) / 1e6 < 10.0
    ]
    within_request_graph_trace_start_cadences_ms = [
        (window_end - window_start) / 1e6
        for window_start, window_end in within_request_windows
    ]
    busy_ms = []
    idle_ms = []
    for window_start, window_end in within_request_windows:
        clipped = [
            (max(start, window_start), min(end, window_end))
            for start, end in all_gpu_activities
            if end > window_start and start < window_end
        ]
        busy = union_duration_ns(clipped) / 1e6
        window = (window_end - window_start) / 1e6
        busy_ms.append(busy)
        idle_ms.append(max(0.0, window - busy))

    result = {
        "schema_version": "vllm-decode-cadence-bound-v2",
        "status": status,
        "scope": "Profiled batch-1 greedy run. Standalone KERNEL activity includes capture/prefill, while GRAPH_TRACE records decode graph replay bodies omitted from KERNEL in this Nsight trace mode.",
        "source": {
            "sqlite": {"path": str(args.sqlite), "sha256": digest(args.sqlite)},
            "trace": {"path": str(args.trace), "sha256": digest(args.trace)},
        },
        "generated_steps": args.generated_steps,
        "request_count": args.request_count,
        "weighted_tpot_ms": weighted_tpot_ms,
        "cuda": {
            "standalone_kernel_instances": kernel_count,
            "standalone_kernel_total_ms": total_kernel_ns / 1e6,
            "standalone_kernel_ms_per_generated_step": standalone_kernel_ms_per_generated_step,
            "graph_trace_count": len(graph_traces),
            "graph_replay_total_ms": sum(graph_trace_durations_ms),
            "graph_replay_duration_ms": {
                "median": statistics.median(graph_trace_durations_ms),
                "mean": statistics.mean(graph_trace_durations_ms),
                "p90": percentile(graph_trace_durations_ms, 0.90),
                "min": min(graph_trace_durations_ms),
                "max": max(graph_trace_durations_ms),
            },
            "graph_replay_ms_per_generated_step": graph_replay_ms_per_generated_step,
            "observed_gpu_activity_ms_per_generated_step_upper_bound": observed_gpu_activity_ms_per_generated_step_upper_bound,
            "observed_gpu_activity_fraction_of_tpot_upper_bound": observed_gpu_activity_ms_per_generated_step_upper_bound / weighted_tpot_ms,
            "outside_observed_gpu_activity_ms_per_token_lower_bound": outside_observed_gpu_activity_ms_per_token_lower_bound,
            "outside_observed_gpu_activity_fraction_of_tpot_lower_bound": outside_observed_gpu_activity_ms_per_token_lower_bound / weighted_tpot_ms,
            "graph_launch_count": len(graph_launches),
            "expected_graph_launch_count": expected_graph_launches,
            "graph_launch_api_duration_ms": {
                "median": statistics.median(launch_durations_ms),
                "mean": statistics.mean(launch_durations_ms),
                "p90": percentile(launch_durations_ms, 0.90),
                "max": max(launch_durations_ms),
            },
            "within_request_graph_launch_start_cadence_ms": {
                "count": len(within_request_cadences_ms),
                "median": statistics.median(within_request_cadences_ms),
                "mean": statistics.mean(within_request_cadences_ms),
                "p90": percentile(within_request_cadences_ms, 0.90),
                "max": max(within_request_cadences_ms),
            },
            "within_request_gpu_busy_union_ms": {
                "count": len(busy_ms),
                "median": statistics.median(busy_ms),
                "mean": statistics.mean(busy_ms),
                "p90": percentile(busy_ms, 0.90),
                "min": min(busy_ms),
                "max": max(busy_ms),
            },
            "within_request_graph_trace_start_cadence_ms": {
                "count": len(within_request_graph_trace_start_cadences_ms),
                "median": statistics.median(
                    within_request_graph_trace_start_cadences_ms
                ),
                "mean": statistics.mean(within_request_graph_trace_start_cadences_ms),
                "p90": percentile(
                    within_request_graph_trace_start_cadences_ms, 0.90
                ),
                "max": max(within_request_graph_trace_start_cadences_ms),
            },
            "within_request_gpu_idle_ms": {
                "count": len(idle_ms),
                "median": statistics.median(idle_ms),
                "mean": statistics.mean(idle_ms),
                "p90": percentile(idle_ms, 0.90),
                "min": min(idle_ms),
                "max": max(idle_ms),
                "mean_fraction_of_launch_cadence": statistics.mean(idle_ms)
                / statistics.mean(within_request_graph_trace_start_cadences_ms),
            },
        },
        "correction": {
            "retracted_interpretation": "KERNEL-table duration alone represents all CUDA execution and leaves roughly 72% of TPOT outside kernels.",
            "reason": "The trace contains one GRAPH_TRACE activity for every decode cudaGraphLaunch, and those replay bodies are absent from the KERNEL-table duration sum.",
            "speedup_if_only_proven_residual_were_removed": weighted_tpot_ms / (
                weighted_tpot_ms
                - outside_observed_gpu_activity_ms_per_token_lower_bound
            ),
            "speedup_if_observed_mean_within_request_gpu_idle_were_removed": statistics.mean(
                within_request_graph_trace_start_cadences_ms
            )
            / (
                statistics.mean(within_request_graph_trace_start_cadences_ms)
                - statistics.mean(idle_ms)
            ),
            "qualification": "Activity durations can overlap and standalone KERNEL includes capture/prefill, so this is a conservative decomposition, not a theoretical optimum or an achievable speedup prediction.",
        },
        "decision": "Do not prioritize runtime orchestration from the retracted 72% figure. The graph replay body is material GPU work already covered by vLLM; require a complete graph-plus-standalone-kernel timeline and a >=3% whole-step mechanism before reopening runtime work.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
