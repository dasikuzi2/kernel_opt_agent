"""Fuse GDN gate preprocessing without specializing on every tensor extent."""

import torch
import triton
import triton.language as tl

from flashinfer.gdn_prefill import chunk_gated_delta_rule


@triton.jit(
    do_not_specialize=["n_elements"],
    do_not_specialize_on_alignment=["n_elements"],
)
def _prepare_gates(a_ptr, b_ptr, a_log_ptr, dt_bias_ptr, g_ptr, beta_ptr,
                   n_elements, n_heads: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    heads = offsets % n_heads

    a = tl.load(a_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask).to(tl.float32)
    a_log = tl.load(a_log_ptr + heads, mask=mask).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + heads, mask=mask).to(tl.float32)

    x = a + dt_bias
    softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    g = tl.exp(-tl.exp(a_log) * softplus)
    beta = 1.0 / (1.0 + tl.exp(-b))
    tl.store(g_ptr + offsets, g, mask=mask)
    tl.store(beta_ptr + offsets, beta, mask=mask)


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    g = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    beta = torch.empty(b.shape, dtype=torch.float32, device=b.device)
    n_elements = a.numel()
    n_heads = A_log.numel()
    # Keep the original launch geometry until the target SM120 is measured;
    # the SM89 proxy only licenses removing redundant extent specializations.
    _prepare_gates[(triton.cdiv(n_elements, 256),)](
        a, b, A_log, dt_bias, g, beta,
        n_elements=n_elements, n_heads=n_heads, BLOCK=256,
    )
    return chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        use_cp="auto",
    )
