"""Test the CP routing boundary with the exact baseline gate equation."""

import torch
import torch.nn.functional as F

from flashinfer.gdn_prefill import chunk_gated_delta_rule


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    log_g = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float())
    g = torch.exp(log_g)
    beta = torch.sigmoid(b.float())
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
        use_cp=True,
    )
