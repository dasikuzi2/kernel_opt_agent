#!/usr/bin/env python3
"""Turn an Nsight Systems SQLite export into an opportunity and reachability map."""

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


def short_family(name: str) -> str:
    if "gemvx::kernel" in name:
        return "cublas_gemvx"
    if "flash_fwd" in name:
        return "flash_attention"
    if "rms_norm" in name or "layer_norm" in name:
        return "normalization"
    if "cublas" in name.lower() or "cutlass" in name.lower():
        return "cublas_or_cutlass_gemm"
    return name


def analyze(
    sqlite_path: Path,
    generated_steps: int | None,
    expectations: list[tuple[str, int]],
    top: int,
) -> dict:
    connection = sqlite3.connect(sqlite_path)
    try:
        columns = {
            row[1]
            for row in connection.execute(
                "pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)"
            )
        }
        required = {
            "start",
            "end",
            "demangledName",
            "gridX",
            "gridY",
            "gridZ",
            "blockX",
            "blockY",
            "blockZ",
        }
        if not required <= columns:
            missing = sorted(required - columns)
            raise ValueError(f"Nsight kernel table is missing columns: {missing}")
        rows = connection.execute(
            """
            select s.value, k.end - k.start,
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ
            from CUPTI_ACTIVITY_KIND_KERNEL k
            join StringIds s on s.id = k.demangledName
            where k.end > k.start
            """
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError("Nsight export contains no CUDA kernel activities")

    by_family: dict[str, dict] = defaultdict(
        lambda: {"total_ns": 0, "instances": 0, "names": set()}
    )
    by_signature: dict[tuple, dict] = defaultdict(
        lambda: {"total_ns": 0, "instances": 0}
    )
    total_ns = 0
    for name, duration, grid_x, grid_y, grid_z, block_x, block_y, block_z in rows:
        family = short_family(name)
        total_ns += duration
        by_family[family]["total_ns"] += duration
        by_family[family]["instances"] += 1
        by_family[family]["names"].add(name)
        signature = (
            family,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
        )
        by_signature[signature]["total_ns"] += duration
        by_signature[signature]["instances"] += 1

    families = []
    for family, value in by_family.items():
        item = {
            "family": family,
            "instances": value["instances"],
            "total_ms": value["total_ns"] / 1e6,
            "time_percent": value["total_ns"] * 100.0 / total_ns,
            "average_us": value["total_ns"] / value["instances"] / 1e3,
            "kernel_names": sorted(value["names"]),
        }
        if generated_steps:
            item["ms_per_generated_step"] = value["total_ns"] / generated_steps / 1e6
            item["instances_per_generated_step"] = (
                value["instances"] / generated_steps
            )
        families.append(item)
    families.sort(key=lambda item: item["total_ms"], reverse=True)

    signatures = []
    for signature, value in by_signature.items():
        family, grid_x, grid_y, grid_z, block_x, block_y, block_z = signature
        item = {
            "family": family,
            "grid": [grid_x, grid_y, grid_z],
            "block": [block_x, block_y, block_z],
            "instances": value["instances"],
            "total_ms": value["total_ns"] / 1e6,
            "average_us": value["total_ns"] / value["instances"] / 1e3,
        }
        if generated_steps:
            item["ms_per_generated_step"] = value["total_ns"] / generated_steps / 1e6
            item["instances_per_generated_step"] = (
                value["instances"] / generated_steps
            )
        signatures.append(item)
    signatures.sort(key=lambda item: item["total_ms"], reverse=True)

    proofs = []
    all_names = [row[0] for row in rows]
    for needle, minimum in expectations:
        observed = sum(1 for name in all_names if needle in name)
        proofs.append(
            {
                "kind": "KERNEL_INSTANCE_COUNT",
                "scope": f"CUDA kernels containing {needle!r}",
                "observed_count": observed,
                "minimum_count": minimum,
                # The Nsight SQLite source is the first (and authoritative)
                # evidence object emitted by this analyzer.
                "evidence_index": 0,
                "status": "PASS" if observed >= minimum else "FAIL",
            }
        )

    return {
        "schema_version": "nsys-cuda-opportunity-map-v1",
        "status": "PASS" if all(p["status"] == "PASS" for p in proofs) else "FAIL",
        "source": {
            "path": str(sqlite_path),
            "sha256": digest(sqlite_path),
        },
        "generated_steps": generated_steps,
        "total_gpu_kernel_ms": total_ns / 1e6,
        "kernel_instances": len(rows),
        "families": families[:top],
        "launch_signatures": signatures[:top],
        "execution_proofs": proofs,
    }


def parse_expectation(value: str) -> tuple[str, int]:
    try:
        needle, minimum_text = value.rsplit("=", 1)
        minimum = int(minimum_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected KERNEL_SUBSTRING=MINIMUM") from error
    if not needle or minimum < 1:
        raise argparse.ArgumentTypeError("kernel substring must be set and minimum positive")
    return needle, minimum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generated-steps", type=int)
    parser.add_argument("--expect-kernel", action="append", default=[], type=parse_expectation)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    if args.generated_steps is not None and args.generated_steps < 1:
        parser.error("--generated-steps must be positive")
    if args.top < 1:
        parser.error("--top must be positive")
    result = analyze(args.sqlite.resolve(), args.generated_steps, args.expect_kernel, args.top)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
