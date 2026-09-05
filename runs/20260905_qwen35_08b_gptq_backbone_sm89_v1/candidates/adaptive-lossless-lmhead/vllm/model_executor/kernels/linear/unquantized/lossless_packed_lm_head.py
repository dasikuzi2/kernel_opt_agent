# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Losslessly packed BF16 language-model head for small-token decode.

This backend is intentionally scoped to ``ParallelLMHead`` rather than the
generic unquantized linear dispatcher.  Language-model heads have a distinct
weight lifecycle (often tied to the input embedding), tensor-parallel vocab
sharding, and a highly asymmetric decode shape.  Keeping the backend here lets
the existing logits processor retain gather, padding, scaling, and sampling
semantics while unsupported shapes fall back to the stock quantization method.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_PACK_BLOCK = 256
_MIN_DENSE_BYTES = 64 * 1024 * 1024
_PACK_WORKSPACE_HEADROOM = 256 * 1024 * 1024
_MAX_K = 8192
_MAX_NUMEL = 2**31 - 1
_STATE_PREFIX = "_vllm_lossless_lm_head_"


@dataclass(frozen=True, slots=True)
class PackedLayoutPlan:
    n: int
    k: int
    numel: int
    padded_numel: int
    block_count: int
    fallback_blocks: int
    dense_bytes: int
    packed_bytes: int
    packed_fraction: float


@dataclass(frozen=True, slots=True)
class PackedLaunchConfig:
    block_n: int
    num_warps: int


def choose_launch_config(k: int) -> PackedLaunchConfig:
    """Select a bounded register footprint for an arbitrary supported K."""
    if k <= 1024:
        return PackedLaunchConfig(block_n=16, num_warps=8)
    if k <= 2048:
        return PackedLaunchConfig(block_n=8, num_warps=8)
    if k <= 4096:
        return PackedLaunchConfig(block_n=4, num_warps=8)
    return PackedLaunchConfig(block_n=2, num_warps=8)


def packed_storage_bytes(
    *, numel: int, padded_numel: int, block_count: int, fallback_blocks: int
) -> int:
    """Return materialized bytes for the exact base-plus-delta layout."""
    del numel  # Included in the signature to make the accounting boundary explicit.
    return (
        padded_numel  # sign + mantissa
        + padded_numel // 2  # two four-bit exponent deltas per byte
        + block_count  # base exponent
        + block_count * 4  # fallback slot, int32
        + fallback_blocks * _PACK_BLOCK * 2  # exact BF16 fallback blocks
    )


def is_statically_eligible(weight: torch.Tensor) -> tuple[bool, str | None]:
    if not current_platform.is_cuda():
        return False, "requires CUDA"
    if not weight.is_cuda:
        return False, "weight is not resident on CUDA"
    if weight.dtype != torch.bfloat16:
        return False, "weight is not BF16"
    if weight.ndim != 2:
        return False, "weight is not a matrix"
    if not weight.is_contiguous():
        return False, "weight is not contiguous row-major"
    n, k = (int(value) for value in weight.shape)
    if n * k > _MAX_NUMEL:
        return False, "head exceeds the kernel's signed 32-bit index space"
    if n * k * weight.element_size() < _MIN_DENSE_BYTES:
        return False, "head is too small for packing overhead to be material"
    if k <= 0 or k > _MAX_K:
        return False, f"K={k} is outside the supported range 1..{_MAX_K}"
    return True, None


def _block_bits(weight: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Return a padded int16 block view for one bounded chunk."""
    flat = weight.view(torch.int16).reshape(-1)
    valid_end = min(end, flat.numel())
    chunk = flat[start:valid_end]
    if valid_end != end:
        padded = torch.zeros(end - start, dtype=torch.int16, device=weight.device)
        padded[: chunk.numel()] = chunk
        chunk = padded
    return chunk.reshape(-1, _PACK_BLOCK)


@torch.no_grad()
def pack_bf16_weight(
    weight: torch.Tensor,
    *,
    max_packed_fraction: float,
    chunk_blocks: int = 16384,
) -> tuple[PackedLayoutPlan, tuple[torch.Tensor, ...]] | None:
    """Pack a BF16 matrix exactly with bounded temporary GPU memory.

    The first pass records only one base byte and one packability bit per
    256-value block.  A second chunked pass fills the final buffers directly,
    avoiding the multi-GiB whole-matrix int32 temporaries of the prototype.
    """
    eligible, reason = is_statically_eligible(weight)
    if not eligible:
        logger.info_once("Lossless lm-head backend skipped: %s", reason)
        return None
    if not 0.5 <= max_packed_fraction <= 1.0:
        raise ValueError("lm_head_max_packed_fraction must be in [0.5, 1.0]")
    if chunk_blocks <= 0:
        raise ValueError("chunk_blocks must be positive")

    n, k = (int(value) for value in weight.shape)
    numel = weight.numel()
    block_count = cdiv(numel, _PACK_BLOCK)
    padded_numel = block_count * _PACK_BLOCK
    base_exponent = torch.empty(block_count, dtype=torch.uint8, device=weight.device)
    packable = torch.empty(block_count, dtype=torch.bool, device=weight.device)

    for block_start in range(0, block_count, chunk_blocks):
        block_end = min(block_start + chunk_blocks, block_count)
        bits = _block_bits(
            weight, block_start * _PACK_BLOCK, block_end * _PACK_BLOCK
        )
        exponents = torch.bitwise_and(
            torch.bitwise_right_shift(bits.to(torch.int32), 7), 0xFF
        )
        minimum = exponents.amin(dim=1)
        maximum = exponents.amax(dim=1)
        base_exponent[block_start:block_end] = minimum.to(torch.uint8)
        packable[block_start:block_end] = maximum - minimum <= 15

    fallback_blocks = int((~packable).sum().item())
    dense_bytes = numel * weight.element_size()
    packed_bytes = packed_storage_bytes(
        numel=numel,
        padded_numel=padded_numel,
        block_count=block_count,
        fallback_blocks=fallback_blocks,
    )
    plan = PackedLayoutPlan(
        n=n,
        k=k,
        numel=numel,
        padded_numel=padded_numel,
        block_count=block_count,
        fallback_blocks=fallback_blocks,
        dense_bytes=dense_bytes,
        packed_bytes=packed_bytes,
        packed_fraction=packed_bytes / dense_bytes,
    )
    if plan.packed_fraction > max_packed_fraction:
        logger.info_once(
            "Lossless lm-head backend skipped: packed fraction %.3f exceeds %.3f",
            plan.packed_fraction,
            max_packed_fraction,
        )
        return None
    free_bytes, _ = torch.cuda.mem_get_info(weight.device)
    required_free_bytes = plan.packed_bytes + _PACK_WORKSPACE_HEADROOM
    if required_free_bytes > free_bytes:
        logger.info_once(
            "Lossless lm-head backend skipped: needs %.3f MiB free including "
            "packing headroom, but only %.3f MiB is available",
            required_free_bytes / (1024 * 1024),
            free_bytes / (1024 * 1024),
        )
        return None

    sign_mantissa = torch.empty(
        padded_numel, dtype=torch.uint8, device=weight.device
    )
    exponent_nibbles = torch.empty(
        padded_numel // 2, dtype=torch.uint8, device=weight.device
    )
    fallback_slot = torch.full(
        (block_count,), -1, dtype=torch.int32, device=weight.device
    )
    fallback_bits = torch.empty(
        (fallback_blocks, _PACK_BLOCK), dtype=torch.int16, device=weight.device
    )

    next_fallback = 0
    for block_start in range(0, block_count, chunk_blocks):
        block_end = min(block_start + chunk_blocks, block_count)
        bits = _block_bits(
            weight, block_start * _PACK_BLOCK, block_end * _PACK_BLOCK
        )
        flat_bits = bits.reshape(-1).to(torch.int32)
        flat_start = block_start * _PACK_BLOCK
        flat_end = block_end * _PACK_BLOCK
        sign_mantissa[flat_start:flat_end] = torch.bitwise_or(
            torch.bitwise_and(flat_bits, 0x7F),
            torch.bitwise_and(torch.bitwise_right_shift(flat_bits, 8), 0x80),
        ).to(torch.uint8)

        exponents = torch.bitwise_and(torch.bitwise_right_shift(flat_bits, 7), 0xFF)
        repeated_base = base_exponent[block_start:block_end].repeat_interleave(
            _PACK_BLOCK
        ).to(torch.int32)
        deltas = exponents - repeated_base
        local_packable = packable[block_start:block_end].repeat_interleave(
            _PACK_BLOCK
        )
        deltas = torch.where(local_packable, deltas, 0).reshape(-1, 2)
        exponent_nibbles[flat_start // 2 : flat_end // 2] = torch.bitwise_or(
            deltas[:, 0], torch.bitwise_left_shift(deltas[:, 1], 4)
        ).to(torch.uint8)

        failed_local = torch.nonzero(
            ~packable[block_start:block_end], as_tuple=False
        ).flatten()
        failed_count = failed_local.numel()
        if failed_count:
            failed_global = failed_local + block_start
            fallback_slot[failed_global] = torch.arange(
                next_fallback,
                next_fallback + failed_count,
                dtype=torch.int32,
                device=weight.device,
            )
            fallback_bits[next_fallback : next_fallback + failed_count] = bits[
                failed_local
            ]
            next_fallback += failed_count

    assert next_fallback == fallback_blocks
    return plan, (
        sign_mantissa,
        exponent_nibbles,
        base_exponent,
        fallback_slot,
        fallback_bits,
    )


@triton.jit
def _lossless_packed_bf16_lm_head_kernel(
    x_ptr,
    sign_mantissa_ptr,
    exponent_nibbles_ptr,
    base_exponent_ptr,
    fallback_slot_ptr,
    fallback_bits_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PACK_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    valid = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)
    linear = offsets_n[:, None] * K + offsets_k[None, :]
    block_id = linear // PACK_BLOCK
    in_block = linear % PACK_BLOCK
    sm = tl.load(sign_mantissa_ptr + linear, mask=valid, other=0).to(tl.int32)
    pair = tl.load(
        exponent_nibbles_ptr + linear // 2, mask=valid, other=0
    ).to(tl.int32)
    delta = (pair >> ((linear & 1) * 4)) & 0xF
    base = tl.load(base_exponent_ptr + block_id, mask=valid, other=0).to(tl.int32)
    slot = tl.load(fallback_slot_ptr + block_id, mask=valid, other=-1)
    packed_bits = ((sm & 0x80) << 8) | ((base + delta) << 7) | (sm & 0x7F)
    exact_fallback = tl.load(
        fallback_bits_ptr + slot * PACK_BLOCK + in_block,
        mask=valid & (slot >= 0),
        other=0,
    ).to(tl.int32) & 0xFFFF
    fp32_bits = tl.where(slot >= 0, exact_fallback, packed_bits) << 16
    weight = tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=f,r",
        [fp32_bits],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    x = tl.load(x_ptr + offsets_k, mask=offsets_k < K, other=0.0)
    accum = tl.sum(weight * x[None, :].to(tl.float32), axis=1)
    tl.store(output_ptr + offsets_n, accum, mask=offsets_n < N)


class _PackedBF16LMHeadKernel(VllmJitKernel["_PackedBF16LMHeadKernel.CompileKey"]):
    @dataclass(frozen=True, slots=True)
    class CompileKey:
        n: int
        k: int
        pack_block: int
        block_n: int
        num_warps: int

    def dispatch(
        self, *, n: int, k: int, pack_block: int, block_n: int, num_warps: int
    ) -> CompileKey:
        return self.CompileKey(
            n=n,
            k=k,
            pack_block=pack_block,
            block_n=block_n,
            num_warps=num_warps,
        )

    def get_warmup_keys(self, **kwargs) -> list[CompileKey]:
        return [self.dispatch(**kwargs)]

    def compile(self, compile_key: CompileKey) -> None:
        n = compile_key.n
        k = compile_key.k
        pack_block = compile_key.pack_block
        block_n = compile_key.block_n
        num_warps = compile_key.num_warps
        _lossless_packed_bf16_lm_head_kernel.warmup(
            TritonWarmupTensor(torch.bfloat16, shape=(1, k)),
            TritonWarmupTensor(torch.uint8),
            TritonWarmupTensor(torch.uint8),
            TritonWarmupTensor(torch.uint8),
            TritonWarmupTensor(torch.int32),
            TritonWarmupTensor(torch.int16),
            TritonWarmupTensor(torch.bfloat16, shape=(1, n)),
            N=n,
            K=k,
            BLOCK_N=block_n,
            BLOCK_K=triton.next_power_of_2(k),
            PACK_BLOCK=pack_block,
            num_warps=num_warps,
            num_stages=1,
            grid=(1,),
        )


_PACKED_KERNEL = _PackedBF16LMHeadKernel()


def _packed_bf16_lm_head_impl(
    x: torch.Tensor,
    sign_mantissa: torch.Tensor,
    exponent_nibbles: torch.Tensor,
    base_exponent: torch.Tensor,
    fallback_slot: torch.Tensor,
    fallback_bits: torch.Tensor,
    n: int,
    k: int,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    output = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    _lossless_packed_bf16_lm_head_kernel[(cdiv(n, block_n),)](
        x,
        sign_mantissa,
        exponent_nibbles,
        base_exponent,
        fallback_slot,
        fallback_bits,
        output,
        N=n,
        K=k,
        BLOCK_N=block_n,
        BLOCK_K=triton.next_power_of_2(k),
        PACK_BLOCK=_PACK_BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def _packed_bf16_lm_head_fake(
    x: torch.Tensor,
    sign_mantissa: torch.Tensor,
    exponent_nibbles: torch.Tensor,
    base_exponent: torch.Tensor,
    fallback_slot: torch.Tensor,
    fallback_bits: torch.Tensor,
    n: int,
    k: int,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    del (
        sign_mantissa,
        exponent_nibbles,
        base_exponent,
        fallback_slot,
        fallback_bits,
        k,
        block_n,
        num_warps,
    )
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    op_name="lossless_packed_bf16_lm_head",
    op_func=_packed_bf16_lm_head_impl,
    mutates_args=[],
    fake_impl=_packed_bf16_lm_head_fake,
)


def _set_buffer(layer: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    if name in layer._buffers:
        setattr(layer, name, value)
    else:
        layer.register_buffer(name, value, persistent=False)


def prepare_lossless_packed_lm_head(layer: torch.nn.Module) -> bool:
    """Prepare one unquantized ParallelLMHead if the configured policy admits it."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None:
        return False
    kernel_config = config.kernel_config
    backend = getattr(kernel_config, "lm_head_backend", "torch")
    if backend == "torch":
        return False
    if backend != "lossless_packed":
        raise ValueError(f"Unsupported lm_head_backend={backend!r}")

    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    packed = pack_bf16_weight(
        weight,
        max_packed_fraction=float(
            getattr(kernel_config, "lm_head_max_packed_fraction", 0.90)
        ),
    )
    if packed is None:
        return False
    plan, tensors = packed
    launch = choose_launch_config(plan.k)
    names = (
        "sign_mantissa",
        "exponent_nibbles",
        "base_exponent",
        "fallback_slot",
        "fallback_bits",
    )
    for name, tensor in zip(names, tensors):
        _set_buffer(layer, _STATE_PREFIX + name, tensor)
    layer._vllm_lossless_lm_head_meta = (
        plan.n,
        plan.k,
        launch.block_n,
        launch.num_warps,
    )
    _PACKED_KERNEL.register_warmup(
        n=plan.n,
        k=plan.k,
        pack_block=_PACK_BLOCK,
        block_n=launch.block_n,
        num_warps=launch.num_warps,
    )
    logger.info_once(
        "Lossless packed BF16 lm-head ready: shape=(%d,%d), %.3f MiB "
        "(%.3f of dense), %d fallback blocks",
        plan.n,
        plan.k,
        plan.packed_bytes / (1024 * 1024),
        plan.packed_fraction,
        plan.fallback_blocks,
    )
    return True


def try_apply_lossless_packed_lm_head(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """Run the prepared M=1 backend or return ``None`` for stock fallback."""
    meta = getattr(layer, "_vllm_lossless_lm_head_meta", None)
    if meta is None or bias is not None:
        return None
    n, k, block_n, num_warps = meta
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.shape[-1] != k
        or x.numel() != k
        or not x.is_contiguous()
    ):
        return None
    args = [
        x,
        *(
            getattr(layer, _STATE_PREFIX + name)
            for name in (
                "sign_mantissa",
                "exponent_nibbles",
                "base_exponent",
                "fallback_slot",
                "fallback_bits",
            )
        ),
        n,
        k,
        block_n,
        num_warps,
    ]
    return torch.ops.vllm.lossless_packed_bf16_lm_head(*args)
