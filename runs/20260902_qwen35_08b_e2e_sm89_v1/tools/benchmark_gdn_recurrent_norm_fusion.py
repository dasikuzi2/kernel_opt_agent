#!/usr/bin/env python3
"""Test an architecture-level packed-GDN decode fusion on Qwen3.5-0.8B.

The stock path launches 64 small recurrent-state programs per layer and then a
second gated RMSNorm kernel.  This candidate assigns one Triton program to an
entire value head, updates its four 32-row state slices sequentially, and keeps
the four output slices live long enough to perform head-wide RMSNorm + SiLU
gating in the same launch.

This is deliberately a bounded experiment, not an autotuning sweep.  It checks
the recurrent state and final BF16 output against the installed vLLM path before
collecting warm and cache-evicted paired timings.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
from vllm.triton_utils import tl, triton


@triton.jit
def _update_state_slice(
    mixed_qkv,
    state,
    q,
    k,
    decay,
    beta,
    token,
    head,
    state_idx,
    stride_mixed_token: tl.constexpr,
    stride_state_token: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    V_BASE: tl.constexpr,
):
    o_k = tl.arange(0, BK)
    o_v = V_BASE + tl.arange(0, 32)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_h = (
        state
        + state_idx * stride_state_token
        + head * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    h = tl.load(p_h, mask=mask_h, other=0.0).to(tl.float32)
    p_mixed = mixed_qkv + token * stride_mixed_token
    v = tl.load(
        p_mixed + (2 * H * K) + head * V + o_v,
        mask=mask_v,
        other=0.0,
    ).to(tl.float32)

    h *= decay
    v -= tl.sum(h * k[None, :], axis=1)
    v *= beta
    h += v[:, None] * k[None, :]
    out = tl.sum(h * q[None, :], axis=1)
    tl.store(p_h, h.to(p_h.dtype.element_ty), mask=mask_h)
    return out


@triton.jit
def fused_recurrent_rmsnorm_gated_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    gate,
    weight,
    state,
    state_indices,
    out,
    scale,
    eps: tl.constexpr,
    stride_mixed_token: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_indices: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // HV
    head = pid % HV
    query_head = head // (HV // H)
    state_idx = tl.load(state_indices + token * stride_indices).to(tl.int64)

    if state_idx <= 0:
        offsets = tl.arange(0, 128)
        tl.store(out + pid * V + offsets, 0.0, mask=offsets < V)
        return

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    p_mixed = mixed_qkv + token * stride_mixed_token
    q = tl.load(p_mixed + query_head * K + o_k, mask=mask_k, other=0.0).to(
        tl.float32
    )
    k = tl.load(
        p_mixed + H * K + query_head * K + o_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
    k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    q *= scale

    a_value = tl.load(a + token * stride_a_token + head).to(tl.float32)
    b_value = tl.load(b + token * stride_b_token + head).to(tl.float32)
    A_log_value = tl.load(A_log + head).to(tl.float32)
    dt_bias_value = tl.load(dt_bias + head).to(tl.float32)
    x = a_value + dt_bias_value
    softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    decay = tl.exp(-tl.exp(A_log_value) * softplus)
    beta = tl.sigmoid(b_value)

    o0 = _update_state_slice(
        mixed_qkv, state, q, k, decay, beta, token, head, state_idx,
        stride_mixed_token, stride_state_token, H, HV, K, V, BK, 0,
    )
    o1 = _update_state_slice(
        mixed_qkv, state, q, k, decay, beta, token, head, state_idx,
        stride_mixed_token, stride_state_token, H, HV, K, V, BK, 32,
    )
    o2 = _update_state_slice(
        mixed_qkv, state, q, k, decay, beta, token, head, state_idx,
        stride_mixed_token, stride_state_token, H, HV, K, V, BK, 64,
    )
    o3 = _update_state_slice(
        mixed_qkv, state, q, k, decay, beta, token, head, state_idx,
        stride_mixed_token, stride_state_token, H, HV, K, V, BK, 96,
    )

    base = pid * V
    slice_offsets = tl.arange(0, 32)
    # Match the stock two-kernel contract: recurrence first materializes BF16,
    # then RMSNorm reloads FP32 values and performs one 128-wide reduction.
    tl.store(out + base + slice_offsets, o0)
    tl.store(out + base + 32 + slice_offsets, o1)
    tl.store(out + base + 64 + slice_offsets, o2)
    tl.store(out + base + 96 + slice_offsets, o3)
    tl.debug_barrier()

    offsets = tl.arange(0, 128)
    recurrent_bf16 = tl.load(out + base + offsets).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(recurrent_bf16 * recurrent_bf16) / V + eps)
    gate_values = tl.load(gate + base + offsets).to(tl.float32)
    weights = tl.load(weight + offsets).to(tl.float32)
    y = (
        recurrent_bf16
        * inv_rms
        * weights
        * (gate_values * tl.sigmoid(gate_values))
    )
    tl.store(out + base + offsets, y)


def launch_fused(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    batch = mixed_qkv.shape[0]
    heads, value_dim, key_dim = state.shape[-3:]
    query_heads = (mixed_qkv.shape[1] - heads * value_dim) // (2 * key_dim)
    assert value_dim == 128 and key_dim == 128
    fused_recurrent_rmsnorm_gated_kernel[(batch * heads,)](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        gate,
        weight,
        state,
        state_indices,
        out,
        key_dim**-0.5,
        eps,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        state.stride(0),
        state_indices.stride(0),
        query_heads,
        heads,
        key_dim,
        value_dim,
        triton.next_power_of_2(key_dim),
        num_warps=1,
        num_stages=3,
    )


def launch_stock(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    recurrent_out: torch.Tensor,
    output_holder: list[torch.Tensor],
    eps: float,
) -> None:
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=state.shape[-1] ** -0.5,
        initial_state=state,
        out=recurrent_out,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    output_holder[0] = rmsnorm_fn(
        recurrent_out.reshape(-1, recurrent_out.shape[-1]),
        weight,
        None,
        z=gate.reshape(-1, gate.shape[-1]),
        eps=eps,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
    )


def elapsed_us(fn, iterations: int, repeats: int) -> list[float]:
    for _ in range(25):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return samples


def single_us(fn, eviction: torch.Tensor | None) -> float:
    if eviction is not None:
        eviction.add_(1)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def capture_graph(fn) -> torch.cuda.CUDAGraph:
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--paired-repeats", type=int, default=31)
    parser.add_argument("--cold-cache-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()

    torch.manual_seed(23)
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

    stock_state = base_state.clone()
    stock_recurrent = torch.empty(
        batch, 1, heads, value_dim, dtype=torch.bfloat16, device=device
    )
    stock_holder = [torch.empty(0, device=device)]
    launch_stock(
        mixed_qkv, a, b, A_log, dt_bias, gate, weight, stock_state,
        state_indices, stock_recurrent, stock_holder, eps,
    )
    fused_state = base_state.clone()
    fused_out = torch.empty_like(gate)
    launch_fused(
        mixed_qkv, a, b, A_log, dt_bias, gate, weight, fused_state,
        state_indices, fused_out, eps,
    )
    torch.cuda.synchronize()
    stock_out = stock_holder[0].reshape_as(fused_out)

    correctness = {
        "state_equal": torch.equal(stock_state, fused_state),
        "output_equal": torch.equal(stock_out, fused_out),
        "max_state_abs": float((stock_state.float() - fused_state.float()).abs().max()),
        "max_output_abs": float((stock_out.float() - fused_out.float()).abs().max()),
    }

    stock_timing_state = base_state.clone()
    fused_timing_state = base_state.clone()
    stock_timing_recurrent = torch.empty_like(stock_recurrent)
    stock_timing_holder = [torch.empty(0, device=device)]
    fused_timing_out = torch.empty_like(fused_out)
    stock_call = lambda: launch_stock(
        mixed_qkv, a, b, A_log, dt_bias, gate, weight, stock_timing_state,
        state_indices, stock_timing_recurrent, stock_timing_holder, eps,
    )
    fused_call = lambda: launch_fused(
        mixed_qkv, a, b, A_log, dt_bias, gate, weight, fused_timing_state,
        state_indices, fused_timing_out, eps,
    )
    stock_samples = elapsed_us(stock_call, args.iterations, args.repeats)
    fused_samples = elapsed_us(fused_call, args.iterations, args.repeats)
    eviction = torch.empty(
        args.cold_cache_bytes // 4, dtype=torch.float32, device=device
    )
    cold_stock, cold_fused = [], []
    for index in range(args.paired_repeats):
        order = ((cold_stock, stock_call), (cold_fused, fused_call))
        if index % 2:
            order = tuple(reversed(order))
        for bucket, fn in order:
            bucket.append(single_us(fn, eviction))

    stock_median = statistics.median(stock_samples)
    fused_median = statistics.median(fused_samples)
    cold_stock_median = statistics.median(cold_stock)
    cold_fused_median = statistics.median(cold_fused)
    stock_graph = capture_graph(stock_call)
    fused_graph = capture_graph(fused_call)
    stock_graph_samples = elapsed_us(
        stock_graph.replay, args.iterations, args.repeats
    )
    fused_graph_samples = elapsed_us(
        fused_graph.replay, args.iterations, args.repeats
    )
    stock_graph_median = statistics.median(stock_graph_samples)
    fused_graph_median = statistics.median(fused_graph_samples)
    result = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "shape": {"batch": batch, "heads": heads, "K": key_dim, "V": value_dim},
        "candidate": "one_program_per_head_sequential_bv32_recurrence_plus_gated_rmsnorm",
        "correctness": correctness,
        "warm": {
            "stock_us": stock_median,
            "fused_us": fused_median,
            "speedup": stock_median / fused_median,
            "stock_samples_us": stock_samples,
            "fused_samples_us": fused_samples,
        },
        "cache_evicted_paired": {
            "eviction_bytes": args.cold_cache_bytes,
            "stock_us": cold_stock_median,
            "fused_us": cold_fused_median,
            "speedup": cold_stock_median / cold_fused_median,
            "stock_samples_us": cold_stock,
            "fused_samples_us": cold_fused,
        },
        "cuda_graph_replay": {
            "stock_us": stock_graph_median,
            "fused_us": fused_graph_median,
            "speedup": stock_graph_median / fused_graph_median,
            "stock_samples_us": stock_graph_samples,
            "fused_samples_us": fused_graph_samples,
        },
        "admissible": all(correctness[key] for key in ("state_equal", "output_equal")),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
