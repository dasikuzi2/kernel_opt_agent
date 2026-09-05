#!/usr/bin/env python3
"""Calibrate SM89 DRAM service with model-sized read-only and copy streams."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def reduce_stream_kernel(source, sink, elements: tl.constexpr, block: tl.constexpr):
    program = tl.program_id(0)
    offsets = program * block + tl.arange(0, block)
    values = tl.load(source + offsets, mask=offsets < elements, other=0.0).to(tl.float32)
    tl.store(sink + program, tl.sum(values, axis=0))


def time_cuda(operation, warmups: int, trials: int) -> list[float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def summarize(samples_ms: list[float], traffic_bytes: int) -> dict:
    median_ms = float(statistics.median(samples_ms))
    return {
        "raw_ms": samples_ms,
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "p90_ms": sorted(samples_ms)[max(0, int(0.9 * len(samples_ms)) - 1)],
        "traffic_bytes": traffic_bytes,
        "median_decimal_gb_per_second": traffic_bytes / (median_ms * 1e6),
        "median_gib_per_second": traffic_bytes / (median_ms / 1000.0) / (1024**3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--elements", type=int, default=772_843_296)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    block = 1024
    source = torch.empty(args.elements, dtype=torch.bfloat16, device="cuda")
    source.fill_(0.5)
    sink = torch.empty(triton.cdiv(args.elements, block), dtype=torch.float32, device="cuda")
    destination = torch.empty_like(source)

    read_samples = time_cuda(
        lambda: reduce_stream_kernel[(triton.cdiv(args.elements, block),)](
            source, sink, elements=args.elements, block=block
        ),
        args.warmups,
        args.trials,
    )
    copy_samples = time_cuda(lambda: destination.copy_(source), args.warmups, args.trials)
    input_bytes = args.elements * source.element_size()
    payload = {
        "schema_version": "sm89-memory-stream-calibration-v1",
        "status": "PASS",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "input": {
            "elements": args.elements,
            "dtype": "bfloat16",
            "input_bytes": input_bytes,
            "block_elements": block,
            "warmups": args.warmups,
            "trials": args.trials,
        },
        "read_only_reduce": summarize(read_samples, input_bytes + sink.numel() * sink.element_size()),
        "device_copy": summarize(copy_samples, input_bytes * 2),
        "interpretation": {
            "read_only_reduce": "input reads plus small per-program FP32 stores; no second reduction is included",
            "device_copy": "aggregate read plus write traffic; do not compare its GB/s directly with a read-only workload",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
