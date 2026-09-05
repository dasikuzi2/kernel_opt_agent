#!/usr/bin/env python3
"""Launch the exact gate candidate in a deterministic Nsight Compute target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from benchmark_gate_fusion_proxy import _candidate, _load_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--total-seq-len", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    module = _load_candidate(args.candidate.resolve())
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    shape = (args.total_seq_len, args.heads)
    a = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    b = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    a_log = torch.randn((args.heads,), device="cuda", dtype=torch.float32, generator=generator) * 0.5
    dt_bias = torch.randn((args.heads,), device="cuda", dtype=torch.float32, generator=generator) * 0.5
    for _ in range(args.warmup):
        result = _candidate(module, a, b, a_log, dt_bias)
    torch.cuda.synchronize()
    result = _candidate(module, a, b, a_log, dt_bias)
    torch.cuda.synchronize()
    sink = float(result[0][0, 0].item()) + float(result[1][-1, -1].item())
    print(json.dumps({
        "status": "PASS",
        "shape": list(shape),
        "warmup_launches": args.warmup,
        "profile_target_launches": 1,
        "live_output_sink": sink,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
