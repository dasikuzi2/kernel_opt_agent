#!/usr/bin/env python3
"""Build and screen the four-warp CUDA GDN recurrence+norm fusion."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import torch

from benchmark_gdn_recurrent_norm_fusion import (
    capture_graph,
    elapsed_us,
    launch_stock,
    single_us,
)


def build_library(source: Path) -> tuple[ctypes.CDLL, Path, list[str]]:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    output_dir = Path(tempfile.gettempdir()) / "kernel_opt_gdn_cuda"
    output_dir.mkdir(parents=True, exist_ok=True)
    library_path = output_dir / f"gdn_recurrent_norm_{digest}.so"
    command = [
        "/usr/bin/nvcc",
        "-O3",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        "-arch=sm_89",
        "-lineinfo",
        str(source),
        "-o",
        str(library_path),
    ]
    if not library_path.exists():
        subprocess.run(command, check=True)
    return ctypes.CDLL(str(library_path)), library_path, command


def configure_launcher(library: ctypes.CDLL):
    launcher = library.launch_gdn_recurrent_norm_fused_sm89
    launcher.argtypes = [ctypes.c_void_p] * 10 + [
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    launcher.restype = ctypes.c_int
    return launcher


def pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("gdn_recurrent_norm_fusion_cuda.cu"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--paired-repeats", type=int, default=31)
    args = parser.parse_args()

    library, library_path, build_command = build_library(args.source.resolve())
    launcher = configure_launcher(library)
    torch.manual_seed(29)
    device = "cuda"
    batch, heads, key_dim, value_dim = 1, 16, 128, 128
    mixed_qkv = torch.randn(
        batch, 3 * heads * key_dim, dtype=torch.bfloat16, device=device
    )
    a = torch.randn(batch, heads, dtype=torch.bfloat16, device=device)
    b = torch.randn_like(a)
    A_log = torch.randn(heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn_like(A_log)
    gate = torch.randn(batch, heads, value_dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(value_dim, dtype=torch.bfloat16, device=device)
    state_indices = torch.ones(batch, dtype=torch.int32, device=device)
    base_state = torch.randn(
        2, heads, value_dim, key_dim, dtype=torch.bfloat16, device=device
    )
    eps = 1e-6

    def cuda_call(state: torch.Tensor, out: torch.Tensor) -> None:
        error = launcher(
            pointer(mixed_qkv),
            pointer(a),
            pointer(b),
            pointer(A_log),
            pointer(dt_bias),
            pointer(gate),
            pointer(weight),
            pointer(state),
            pointer(state_indices),
            pointer(out),
            ctypes.c_float(eps),
            ctypes.c_int(batch),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        if error:
            raise RuntimeError(f"CUDA launch failed with cudaError={error}")

    stock_state = base_state.clone()
    stock_recurrent = torch.empty(
        batch, 1, heads, value_dim, dtype=torch.bfloat16, device=device
    )
    stock_holder = [torch.empty(0, device=device)]
    launch_stock(
        mixed_qkv, a, b, A_log, dt_bias, gate, weight, stock_state,
        state_indices, stock_recurrent, stock_holder, eps,
    )
    candidate_state = base_state.clone()
    candidate_out = torch.empty_like(gate)
    cuda_call(candidate_state, candidate_out)
    torch.cuda.synchronize()
    stock_out = stock_holder[0].reshape_as(candidate_out)
    correctness = {
        "state_equal": torch.equal(stock_state, candidate_state),
        "output_equal": torch.equal(stock_out, candidate_out),
        "max_state_abs": float((stock_state.float() - candidate_state.float()).abs().max()),
        "max_output_abs": float((stock_out.float() - candidate_out.float()).abs().max()),
        "state_mismatch_count": int(torch.count_nonzero(stock_state != candidate_state)),
        "output_mismatch_count": int(torch.count_nonzero(stock_out != candidate_out)),
    }

    stock_timing_state = base_state.clone()
    stock_timing_recurrent = torch.empty_like(stock_recurrent)
    stock_timing_holder = [torch.empty(0, device=device)]
    candidate_timing_state = base_state.clone()
    candidate_timing_out = torch.empty_like(candidate_out)
    stock_call = lambda: launch_stock(
        mixed_qkv, a, b, A_log, dt_bias, gate, weight, stock_timing_state,
        state_indices, stock_timing_recurrent, stock_timing_holder, eps,
    )
    candidate_call = lambda: cuda_call(candidate_timing_state, candidate_timing_out)

    # vLLM decode is graph replayed, so this is the primary timing gate.
    stock_graph = capture_graph(stock_call)
    candidate_graph = capture_graph(candidate_call)
    stock_samples = elapsed_us(stock_graph.replay, args.iterations, args.repeats)
    candidate_samples = elapsed_us(
        candidate_graph.replay, args.iterations, args.repeats
    )
    stock_median = statistics.median(stock_samples)
    candidate_median = statistics.median(candidate_samples)

    eviction = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, device=device)
    cold_stock, cold_candidate = [], []
    for index in range(args.paired_repeats):
        order = ((cold_stock, stock_graph.replay), (cold_candidate, candidate_graph.replay))
        if index % 2:
            order = tuple(reversed(order))
        for bucket, fn in order:
            bucket.append(single_us(fn, eviction))
    cold_stock_median = statistics.median(cold_stock)
    cold_candidate_median = statistics.median(cold_candidate)

    result = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "nvcc": "12.0.140",
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "temporary_library": str(library_path),
        "build_command": build_command,
        "candidate": "four_warps_per_head_parallel_bv32_recurrence_plus_gated_rmsnorm",
        "correctness": correctness,
        "cuda_graph_replay": {
            "stock_us": stock_median,
            "candidate_us": candidate_median,
            "speedup": stock_median / candidate_median,
            "stock_samples_us": stock_samples,
            "candidate_samples_us": candidate_samples,
        },
        "cache_evicted_graph_replay": {
            "stock_us": cold_stock_median,
            "candidate_us": cold_candidate_median,
            "speedup": cold_stock_median / cold_candidate_median,
            "stock_samples_us": cold_stock,
            "candidate_samples_us": cold_candidate,
        },
        "strictly_admissible": correctness["state_equal"] and correctness["output_equal"],
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
