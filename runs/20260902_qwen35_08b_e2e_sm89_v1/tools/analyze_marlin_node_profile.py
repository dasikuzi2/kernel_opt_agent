#!/usr/bin/env python3
"""Split a node-expanded Nsight trace into lm-head and backbone Marlin work.

The split is deliberately conservative.  It is accepted only when one exact
launch signature occurs an integral number of times per generated step and
the expected lm-head stratum is separated from every remaining launch in that
signature by a caller-selected duration ratio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _percent(part_ns: int, total_ns: int) -> float:
    return 100.0 * part_ns / total_ns if total_ns else 0.0


def _per_step_ms(duration_ns: int, generated_steps: int) -> float:
    return duration_ns / generated_steps / 1e6


def _elimination_speedup(total_ns: int, removable_ns: int) -> float | None:
    residual = total_ns - removable_ns
    return total_ns / residual if residual > 0 else None


def _speedup_needed_for_global_gain(
    contribution_ns: int, total_ns: int, global_gain_fraction: float
) -> float | None:
    required_saving_ns = total_ns * global_gain_fraction
    residual = contribution_ns - required_saving_ns
    if residual <= 0:
        return None
    return contribution_ns / residual


def analyze(
    sqlite_path: Path,
    generated_steps: int,
    kernel_substring: str,
    lm_head_launches_per_step: int = 1,
    minimum_gap_ratio: float = 5.0,
    global_materiality_fraction: float = 0.03,
    bindings: tuple[Path, ...] = (),
    decode_steps: int | None = None,
) -> dict:
    if generated_steps < 1:
        raise ValueError("generated_steps must be positive")
    if lm_head_launches_per_step < 1:
        raise ValueError("lm_head_launches_per_step must be positive")
    decode_steps = generated_steps if decode_steps is None else decode_steps
    if decode_steps < 1 or decode_steps > generated_steps:
        raise ValueError("decode_steps must be positive and no larger than generated_steps")
    if minimum_gap_ratio <= 1.0:
        raise ValueError("minimum_gap_ratio must be greater than one")
    if not 0.0 < global_materiality_fraction < 1.0:
        raise ValueError("global_materiality_fraction must be between zero and one")

    connection = sqlite3.connect(sqlite_path)
    try:
        kernel_rows = connection.execute(
            """
            select s.value, k.end - k.start,
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ
            from CUPTI_ACTIVITY_KIND_KERNEL k
            join StringIds s on s.id = k.demangledName
            where k.end > k.start
            """
        ).fetchall()
        graph_tables = [
            row[0]
            for row in connection.execute(
                """
                select name from sqlite_master
                where type = 'table' and name like 'CUPTI_ACTIVITY_KIND_GRAPH%'
                """
            )
        ]
        graph_rows = sum(
            connection.execute(f'select count(*) from "{table}"').fetchone()[0]
            for table in graph_tables
        )
    finally:
        connection.close()

    if not kernel_rows:
        raise ValueError("Nsight export contains no CUDA kernel activities")

    total_ns = sum(int(row[1]) for row in kernel_rows)
    matched_rows = [row for row in kernel_rows if kernel_substring in str(row[0])]
    if not matched_rows:
        raise ValueError(f"no CUDA kernel contains {kernel_substring!r}")

    signatures: dict[tuple, list[int]] = defaultdict(list)
    for name, duration, grid_x, grid_y, grid_z, block_x, block_y, block_z in matched_rows:
        # Geometry alone is insufficient: Marlin template variants with
        # different tile ownership can share the same grid and block shape.
        signature = (str(name), grid_x, grid_y, grid_z, block_x, block_y, block_z)
        signatures[signature].append(int(duration))

    dominant_signature, dominant_durations = max(
        signatures.items(), key=lambda item: (sum(item[1]), len(item[1]))
    )
    dominant_durations = sorted(dominant_durations, reverse=True)
    lm_head_count = generated_steps * lm_head_launches_per_step
    if len(dominant_durations) <= lm_head_count:
        raise ValueError("dominant signature has no launches below the lm-head stratum")

    lm_head_durations = dominant_durations[:lm_head_count]
    backbone_durations = dominant_durations[lm_head_count:]
    min_lm_head_ns = min(lm_head_durations)
    max_backbone_ns = max(backbone_durations)
    observed_gap_ratio = min_lm_head_ns / max_backbone_ns

    matched_ns = sum(int(row[1]) for row in matched_rows)
    dominant_ns = sum(dominant_durations)
    lm_head_ns = sum(lm_head_durations)
    backbone_ns = sum(backbone_durations)
    other_matched_ns = matched_ns - dominant_ns
    nonmatched_ns = total_ns - matched_ns
    integral_repetition = len(backbone_durations) % decode_steps == 0
    observed_per_generated_step = len(dominant_durations) / generated_steps
    observed_backbone_per_decode_step = len(backbone_durations) / decode_steps

    checks = [
        {
            "check": "NODE_EXPANDED_TRACE_HAS_NO_GRAPH_SUMMARY_ROWS",
            "observed_graph_rows": graph_rows,
            "status": "PASS" if graph_rows == 0 else "FAIL",
        },
        {
            "check": "BACKBONE_STRATUM_REPEATS_INTEGRALLY_PER_DECODE_STEP",
            "backbone_instances": len(backbone_durations),
            "decode_steps": decode_steps,
            "instances_per_decode_step": observed_backbone_per_decode_step,
            "status": "PASS" if integral_repetition else "FAIL",
        },
        {
            "check": "LM_HEAD_DURATION_STRATUM_IS_SEPARABLE",
            "minimum_lm_head_us": min_lm_head_ns / 1e3,
            "maximum_backbone_us": max_backbone_ns / 1e3,
            "observed_gap_ratio": observed_gap_ratio,
            "minimum_gap_ratio": minimum_gap_ratio,
            "status": "PASS" if observed_gap_ratio >= minimum_gap_ratio else "FAIL",
        },
    ]
    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"

    components = {}
    for component, duration_ns, instances in (
        ("all_gpu_kernels", total_ns, len(kernel_rows)),
        ("all_matching_kernels", matched_ns, len(matched_rows)),
        ("dominant_signature", dominant_ns, len(dominant_durations)),
        ("lm_head_stratum", lm_head_ns, len(lm_head_durations)),
        ("dominant_backbone_stratum", backbone_ns, len(backbone_durations)),
        ("other_matching_signatures", other_matched_ns, len(matched_rows) - len(dominant_durations)),
        ("nonmatching_kernels", nonmatched_ns, len(kernel_rows) - len(matched_rows)),
    ):
        components[component] = {
            "instances": instances,
            "total_ms": duration_ns / 1e6,
            "ms_per_generated_step": _per_step_ms(duration_ns, generated_steps),
            "share_of_gpu_kernel_time_percent": _percent(duration_ns, total_ns),
            "zero_cost_elimination_speedup_ceiling": _elimination_speedup(total_ns, duration_ns),
            "local_speedup_required_for_global_materiality": _speedup_needed_for_global_gain(
                duration_ns, total_ns, global_materiality_fraction
            ),
        }

    return {
        "schema_version": "marlin-node-profile-split-v1",
        "status": status,
        "claim_scope": "EMPIRICAL_CURRENT_SCHEDULE_NOT_GLOBAL_OPTIMUM",
        "source": {"path": str(sqlite_path), "sha256": digest(sqlite_path)},
        "bindings": [
            {"path": str(path), "sha256": digest(path)} for path in bindings
        ],
        "configuration": {
            "generated_steps": generated_steps,
            "decode_steps": decode_steps,
            "kernel_substring": kernel_substring,
            "lm_head_launches_per_step": lm_head_launches_per_step,
            "minimum_gap_ratio": minimum_gap_ratio,
            "global_materiality_fraction": global_materiality_fraction,
        },
        "dominant_launch_signature": {
            "grid": list(dominant_signature[1:4]),
            "block": list(dominant_signature[4:]),
            "kernel_name": dominant_signature[0],
            "instances_per_generated_step": observed_per_generated_step,
            "inferred_backbone_instances_per_decode_step": observed_backbone_per_decode_step,
        },
        "checks": checks,
        "components": components,
        "interpretation": {
            "selected_next_target": "dominant_backbone_stratum" if status == "PASS" else "NONE",
            "reason": (
                "The dominant repeated Marlin signature is larger than the lm-head residual and is the next empirical screening target."
                if status == "PASS"
                else "The automatic lm-head/backbone split is not identifiable from this trace."
            ),
            "warning": "Zero-cost elimination ceilings are Amdahl bounds, not achievable predictions.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--generated-steps", type=int, required=True)
    parser.add_argument(
        "--decode-steps",
        type=int,
        help="CUDA decode replays; excludes each request's first token produced by prefill",
    )
    parser.add_argument("--kernel-substring", default="marlin::Marlin")
    parser.add_argument("--lm-head-launches-per-step", type=int, default=1)
    parser.add_argument("--minimum-gap-ratio", type=float, default=5.0)
    parser.add_argument("--global-materiality-fraction", type=float, default=0.03)
    parser.add_argument("--bind", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.sqlite.resolve(),
        args.generated_steps,
        args.kernel_substring,
        args.lm_head_launches_per_step,
        args.minimum_gap_ratio,
        args.global_materiality_fraction,
        tuple(path.resolve() for path in args.bind),
        args.decode_steps,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
