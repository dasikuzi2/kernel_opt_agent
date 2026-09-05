"""Opt-in vLLM mixed-precision kernel routing for one bounded candidate.

Python imports ``sitecustomize`` in both the benchmark process and vLLM's
spawned engine process.  The patch is therefore process-local and leaves the
installed vLLM tree unchanged.
"""

from __future__ import annotations

import os


raw_shapes = os.environ.get("VLLM_SM89_W4_HUMMING_SHAPES", "").strip()
if raw_shapes:
    selected_shapes = {
        tuple(map(int, item.lower().split("x", 1)))
        for item in raw_shapes.split(",")
        if item.strip()
    }
    expected_shapes = {(1024, 2048), (1024, 3584)}
    if selected_shapes != expected_shapes:
        raise RuntimeError(
            f"bounded hybrid candidate requires {sorted(expected_shapes)}, "
            f"observed {sorted(selected_shapes)}"
        )

    import torch
    import vllm.model_executor.kernels.linear as linear_registry
    import vllm.model_executor.layers.quantization.auto_gptq as auto_gptq
    from vllm.model_executor.kernels.linear.mixed_precision.humming import (
        HummingLinearKernel,
    )

    original_choose = linear_registry.choose_mp_linear_kernel

    def choose_shape_routed_kernel(config, compute_capability=None):
        n = int(config.partition_weight_shape[1])
        k = int(config.partition_weight_shape[0])
        if (n, k) in selected_shapes:
            if config.act_type != torch.bfloat16:
                raise RuntimeError("hybrid candidate requires BF16 activations")
            if config.group_size != 128 or config.has_g_idx:
                raise RuntimeError("hybrid candidate requires group128 without g_idx")
            supported, reason = HummingLinearKernel.can_implement(config)
            if not supported:
                raise RuntimeError(f"Humming rejected selected shape N={n},K={k}: {reason}")
            print(f"SM89_HYBRID_SELECT HummingLinearKernel N={n} K={k}", flush=True)
            return HummingLinearKernel
        return original_choose(config, compute_capability)

    linear_registry.choose_mp_linear_kernel = choose_shape_routed_kernel
    # auto_gptq imports the function symbol directly, so replace its bound
    # reference as well as the registry export.
    auto_gptq.choose_mp_linear_kernel = choose_shape_routed_kernel
