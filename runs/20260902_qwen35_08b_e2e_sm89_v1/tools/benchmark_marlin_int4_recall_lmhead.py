#!/usr/bin/env python3
"""Screen vLLM Marlin W4A16/W4A8 as approximate lm-head shortlist scans."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types

from benchmark_exact_bf16_packed_lmhead import launch_packed, pack_exact_bf16


def measure_us(
    fn, iterations: int, repeats: int, warmup_iterations: int
) -> tuple[float, list[float]]:
    for _ in range(warmup_iterations):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples), samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--vectors", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shortlist", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument(
        "--include-w4a8-int8",
        action="store_true",
        help="Also measure the known-bad experimental W4A8 path.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > args.vectors:
        raise ValueError("batch-size must be in [1, vectors]")

    torch.manual_seed(20260905)
    model_path = args.model.resolve()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).eval().to("cuda")
    # Marlin uses [K, N], while nn.Linear stores [N, K].
    dense_weight = model.lm_head.weight.detach().t().contiguous()
    k, n = dense_weight.shape
    del model
    torch.cuda.empty_cache()

    packed_exact, exact_pack_check = pack_exact_bf16(dense_weight.t().contiguous())
    del packed_exact["packed_blocks"]
    vectors = (
        torch.randn((args.vectors, k), device="cuda", dtype=torch.bfloat16) * 0.02
    ).contiguous()
    exact_outputs = []
    for vector in vectors:
        output = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
        launch_packed(vector[None, :], packed_exact, output, n, k, 16, 8)
        exact_outputs.append(output)
    torch.cuda.synchronize()

    exact_scratch = torch.empty(
        (args.batch_size, n), device="cuda", dtype=torch.bfloat16
    )
    if args.batch_size == 1:
        exact_fn = lambda: launch_packed(  # noqa: E731
            vectors[0:1], packed_exact, exact_scratch, n, k, 16, 8
        )
        exact_label = "exact-packed BF16 lm-head"
    else:
        exact_fn = lambda: torch.nn.functional.linear(  # noqa: E731
            vectors[: args.batch_size], dense_weight.t()
        )
        exact_label = "torch BF16 linear"
    exact_us, exact_samples = measure_us(
        exact_fn, args.iterations, args.repeats, args.warmup_iterations
    )

    candidates = [("w4a16", None)]
    if args.include_w4a8_int8:
        candidates.append(("w4a8_int8", torch.int8))

    results = []
    for label, input_dtype in candidates:
        _, marlin_weight, marlin_scales, g_idx, sort_indices, _ = marlin_quantize(
            dense_weight,
            scalar_types.uint4b8,
            args.group_size,
            False,
            input_dtype=input_dtype,
        )
        workspace = marlin_make_workspace_new(torch.device("cuda", 0))
        zero_points = torch.empty(0, dtype=torch.int32, device="cuda")
        input_global_scale = (
            torch.tensor(1.0, dtype=torch.float32, device="cuda")
            if input_dtype == torch.int8
            else None
        )

        def run(vector: torch.Tensor = vectors[: args.batch_size]) -> torch.Tensor:
            return apply_gptq_marlin_linear(
                vector,
                marlin_weight,
                marlin_scales,
                zero_points,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.uint4b8,
                output_size_per_partition=n,
                input_size_per_partition=k,
                is_k_full=True,
                input_global_scale=input_global_scale,
                input_dtype=input_dtype,
            )

        candidate_us, candidate_samples = measure_us(
            run, args.iterations, args.repeats, args.warmup_iterations
        )
        batch_candidate = run(vectors[: args.batch_size])
        batch_exact = torch.nn.functional.linear(
            vectors[: args.batch_size], dense_weight.t()
        )
        torch.cuda.synchronize()
        batch_ranks = []
        for row in range(args.batch_size):
            exact_top1 = batch_exact[row].argmax().item()
            winner_score = batch_candidate[row, exact_top1]
            batch_ranks.append(
                int((batch_candidate[row] > winner_score).sum().item()) + 1
            )
        ranks = []
        top1_equal = 0
        max_abs_errors = []
        for vector, exact_output in zip(vectors, exact_outputs, strict=True):
            candidate_output = run(vector[None, :])
            torch.cuda.synchronize()
            exact_top1 = exact_output.argmax().item()
            winner_score = candidate_output[0, exact_top1]
            rank = int((candidate_output[0] > winner_score).sum().item()) + 1
            ranks.append(rank)
            top1_equal += int(candidate_output.argmax().item() == exact_top1)
            max_abs_errors.append(
                float((candidate_output.float() - exact_output.float()).abs().max())
            )
        results.append(
            {
                "candidate": label,
                "batch_size": args.batch_size,
                "group_size": args.group_size,
                "control": exact_label,
                "median_control_us": exact_us,
                "speedup_vs_control": exact_us / candidate_us,
                # Retained for compatibility with the original M=1 receipt.
                "median_exact_packed_us": exact_us,
                "median_candidate_us": candidate_us,
                "speedup_vs_exact_packed": exact_us / candidate_us,
                "candidate_samples_us": candidate_samples,
                "quality": {
                    "random_vectors": args.vectors,
                    "top1_equal": top1_equal,
                    "exact_winner_recalled_in_topk": sum(
                        rank <= args.shortlist for rank in ranks
                    ),
                    "shortlist": args.shortlist,
                    "exact_winner_approximate_ranks": ranks,
                    "maximum_rank": max(ranks),
                    "maximum_abs_logit_errors": max_abs_errors,
                    "measured_batch_exact_winner_approximate_ranks": batch_ranks,
                },
            }
        )
        del marlin_weight, marlin_scales, g_idx, sort_indices, workspace
        torch.cuda.empty_cache()

    payload = {
        "schema_version": "sm89-vllm-marlin-int4-recall-lmhead-screen-v1",
        "status": "PASS",
        "scope": {
            "model": str(model_path),
            "gpu": torch.cuda.get_device_name(0),
            "weight_shape_kn": [k, n],
            "batch_size": args.batch_size,
            "control": exact_label,
            "candidate": "vLLM Marlin GPTQ uint4b8 scan for shortlist recall",
        },
        "control_weight_bit_exact": exact_pack_check,
        "exact_samples_us": exact_samples,
        "results": results,
        "best_speed": max(results, key=lambda item: item["speedup_vs_exact_packed"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
