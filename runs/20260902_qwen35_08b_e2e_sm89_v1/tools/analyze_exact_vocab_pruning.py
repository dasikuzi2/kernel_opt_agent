#!/usr/bin/env python3
"""Measure whether certified vocabulary pruning can avoid BF16 lm-head reads.

Rows are clustered offline.  At decode time each cluster has the exact bound

    max_i <w_i, h> <= <c_g, h> + sum_b R[g,b] * ||h_b||_2,

where R[g,b] is the maximum block residual norm inside cluster g.  Scoring one
representative per cluster supplies a real lower bound; only clusters whose
upper bound can still beat it need their full BF16 weight rows read.  This tool
evaluates both that executable plan and an optimistic oracle-lower-bound plan
on real autoregressive hidden states before any production kernel is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


NATURAL_REQUESTS = (
    (
        "zh-explanation",
        "请用通俗但准确的语言解释：为什么大语言模型逐 token 解码通常受显存带宽限制？给出一个简单的数量级估算。",
    ),
    (
        "python-code",
        "Write a complete Python function that merges overlapping half-open intervals. Include type hints, a concise explanation, and three edge-case tests.",
    ),
    (
        "reasoning",
        "A shop discounts an item by 20%, then raises the discounted price by 25%. Explain step by step whether the final price equals the original price.",
    ),
    (
        "editing",
        "Rewrite the following paragraph to be concise and professional without losing any facts: Our team ran several experiments over the last two weeks, but because each trial used a different environment and no frozen baseline, the reported improvements cannot yet be compared reliably.",
    ),
    (
        "systems-design",
        "Design a bounded GPU-kernel optimization loop for an autonomous agent. Focus on candidate diversity, fail-fast checks, correctness gates, measurement noise, and stopping rules.",
    ),
    (
        "translation",
        "Translate into natural Chinese and briefly explain the technical meaning: Speculative decoding reduces latency only when accepted draft tokens amortize the verification cost.",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile input")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def distribution(values: list[float]) -> dict:
    return {
        "min": min(values),
        "p10": percentile(values, 0.10),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


@torch.inference_mode()
def collect_hidden_states(model, tokenizer, new_tokens: int) -> tuple[torch.Tensor, list[str], dict]:
    hidden_states = []
    labels = []
    generated = {}
    for case_id, request in NATURAL_REQUESTS:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": request}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer.encode(rendered, add_special_tokens=False, return_tensors="pt").to("cuda")
        step_input = input_ids
        past_key_values = None
        tokens = []
        for _ in range(new_tokens):
            output = model(
                input_ids=step_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = output.hidden_states[-1][:, -1, :].contiguous()
            logits = model.lm_head(hidden)
            token = logits.float().argmax(dim=-1)
            hidden_states.append(hidden.detach())
            labels.append(case_id)
            tokens.append(int(token.item()))
            past_key_values = output.past_key_values
            step_input = token.view(1, 1)
        generated[case_id] = tokens
    return torch.cat(hidden_states, dim=0), labels, generated


def assign_rows(weight: torch.Tensor, centers: torch.Tensor, chunk_rows: int) -> tuple[torch.Tensor, float]:
    assignments = torch.empty(weight.shape[0], device=weight.device, dtype=torch.int64)
    center_norm = centers.square().sum(dim=1)
    objective_sum = 0.0
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        rows = weight[start:end]
        similarity = 2.0 * (rows @ centers.T)
        distance = rows.square().sum(dim=1, keepdim=True) + center_norm[None, :] - similarity
        best_distance, best = distance.min(dim=1)
        assignments[start:end] = best
        objective_sum += float(best_distance.sum().item())
    return assignments, objective_sum / weight.shape[0]


def update_centers(
    weight: torch.Tensor,
    assignments: torch.Tensor,
    old_centers: torch.Tensor,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    clusters = old_centers.shape[0]
    sums = torch.zeros_like(old_centers)
    counts = torch.bincount(assignments, minlength=clusters)
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        sums.index_add_(0, assignments[start:end], weight[start:end])
    nonempty = counts > 0
    centers = old_centers.clone()
    centers[nonempty] = sums[nonempty] / counts[nonempty, None]
    return centers, counts


def build_bounds_and_representatives(
    weight: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
    block_dims: int,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    clusters, hidden_size = centers.shape
    blocks = hidden_size // block_dims
    radii = torch.zeros((clusters, blocks), device=weight.device, dtype=torch.float32)
    row_distances = torch.empty(weight.shape[0], device=weight.device, dtype=torch.float32)
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        cluster_ids = assignments[start:end]
        residual = weight[start:end] - centers[cluster_ids]
        row_distances[start:end] = residual.square().sum(dim=1)
        block_norms = residual.reshape(-1, blocks, block_dims).square().sum(dim=2).sqrt()
        radii.scatter_reduce_(
            0,
            cluster_ids[:, None].expand(-1, blocks),
            block_norms,
            reduce="amax",
            include_self=True,
        )

    minimum_distance = torch.full((clusters,), float("inf"), device=weight.device)
    minimum_distance.scatter_reduce_(
        0, assignments, row_distances, reduce="amin", include_self=True
    )
    row_ids = torch.arange(weight.shape[0], device=weight.device, dtype=torch.int64)
    representative_candidates = torch.where(
        row_distances <= minimum_distance[assignments] + 1e-7,
        row_ids,
        torch.full_like(row_ids, weight.shape[0]),
    )
    representative_ids = torch.full(
        (clusters,), weight.shape[0], device=weight.device, dtype=torch.int64
    )
    representative_ids.scatter_reduce_(
        0,
        assignments,
        representative_candidates,
        reduce="amin",
        include_self=True,
    )
    if bool((representative_ids == weight.shape[0]).any()):
        raise RuntimeError("empty cluster survived final assignment")
    maximum_radius = float(radii.max().item())
    return radii, representative_ids, row_distances, maximum_radius


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--kmeans-iterations", type=int, default=4)
    parser.add_argument("--block-dims", type=int, default=64)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()

    torch.manual_seed(20260905)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).eval().to("cuda")
    hidden, labels, generated = collect_hidden_states(model, tokenizer, args.new_tokens)
    weight_shape = list(model.lm_head.weight.shape)
    weight = model.lm_head.weight.detach().float().contiguous()
    del model
    torch.cuda.empty_cache()

    vocab_size, hidden_size = weight.shape
    if args.block_dims <= 0 or hidden_size % args.block_dims:
        raise ValueError("--block-dims must divide hidden size")
    if args.clusters <= 1 or args.clusters >= vocab_size:
        raise ValueError("invalid cluster count")

    initial_ids = torch.linspace(
        0, vocab_size - 1, args.clusters, device="cuda", dtype=torch.float64
    ).round().to(torch.int64)
    centers = weight[initial_ids].clone()
    objectives = []
    for _ in range(args.kmeans_iterations):
        assignments, objective = assign_rows(weight, centers, args.chunk_rows)
        centers, counts = update_centers(weight, assignments, centers, args.chunk_rows)
        objectives.append(objective)
    assignments, final_objective = assign_rows(weight, centers, args.chunk_rows)
    centers, counts = update_centers(weight, assignments, centers, args.chunk_rows)
    # Reassign once to the actual centers used by the bounds.
    assignments, final_objective = assign_rows(weight, centers, args.chunk_rows)
    centers, counts = update_centers(weight, assignments, centers, args.chunk_rows)
    if int((counts == 0).sum().item()):
        raise RuntimeError("k-means produced empty clusters")

    radii, representative_ids, _, maximum_radius = build_bounds_and_representatives(
        weight, centers, assignments, args.block_dims, args.chunk_rows
    )
    representatives = weight[representative_ids]
    row_norms = weight.square().sum(dim=1).sqrt()
    blocks = hidden_size // args.block_dims
    group_sizes = counts.float()

    metrics = defaultdict(list)
    per_case = defaultdict(lambda: defaultdict(list))
    all_bounds_valid = True
    all_winners_survive = True
    seed_hits = 0
    top_ids = []
    for query_index, hidden_bf16 in enumerate(hidden):
        query = hidden_bf16.float()
        raw_scores = weight @ query
        rounded_scores = raw_scores.to(torch.bfloat16).float()
        top_values, top_indices = torch.topk(rounded_scores, 2)
        top1 = float(top_values[0].item())
        top_id = int(top_indices[0].item())
        margin = float((top_values[0] - top_values[1]).item())

        center_scores = centers @ query
        query_block_norms = query.reshape(blocks, args.block_dims).square().sum(dim=1).sqrt()
        upper = center_scores + (radii * query_block_norms[None, :]).sum(dim=1)
        # A conservative BF16 rounding allowance; exact dot scores are rounded
        # before production argmax, so a pruned raw upper must also cover that.
        upper_guard = upper + upper.abs() * (2.0**-7) + 1e-4

        actual_group_max = torch.full(
            (args.clusters,), float("-inf"), device="cuda", dtype=torch.float32
        )
        actual_group_max.scatter_reduce_(
            0, assignments, rounded_scores, reduce="amax", include_self=True
        )
        violation = float((actual_group_max - upper_guard).max().item())
        bounds_valid = violation <= 0.0
        all_bounds_valid &= bounds_valid

        representative_scores = (representatives @ query).to(torch.bfloat16).float()
        seed_value, seed_cluster = representative_scores.max(dim=0)
        seed_token = int(representative_ids[int(seed_cluster.item())].item())
        seed_hits += int(seed_token == top_id)

        oracle_surviving = upper_guard >= top1
        executable_surviving = upper_guard >= float(seed_value.item())
        winner_cluster = int(assignments[top_id].item())
        all_winners_survive &= bool(executable_surviving[winner_cluster].item())
        oracle_rows = float(group_sizes[oracle_surviving].sum().item())
        executable_rows = float(group_sizes[executable_surviving].sum().item())

        # Per-query bytes: BF16 centers + BF16 representatives + FP32 block
        # radii + BF16 rows in all surviving clusters. This intentionally
        # ignores launch/indexing overhead and is therefore optimistic.
        index_bytes = (
            2 * args.clusters * hidden_size
            + 2 * args.clusters * hidden_size
            + 4 * args.clusters * blocks
        )
        full_weight_bytes = 2 * vocab_size * hidden_size
        executable_fraction = executable_rows / vocab_size
        oracle_fraction = oracle_rows / vocab_size
        byte_fraction = (index_bytes + 2 * executable_rows * hidden_size) / full_weight_bytes
        row_norm_surviving = row_norms * query.norm() >= top1
        row_norm_fraction = float(row_norm_surviving.float().mean().item())
        row_norm_byte_fraction = (
            4 * vocab_size + 2 * row_norm_fraction * vocab_size * hidden_size
        ) / full_weight_bytes

        row = {
            "top1_margin_bf16": margin,
            "oracle_surviving_row_fraction": oracle_fraction,
            "executable_surviving_row_fraction": executable_fraction,
            "executable_total_byte_fraction": byte_fraction,
            "oracle_row_norm_surviving_fraction": row_norm_fraction,
            "oracle_row_norm_total_byte_fraction": row_norm_byte_fraction,
            "surviving_cluster_fraction": float(executable_surviving.float().mean().item()),
            "bound_violation": max(violation, 0.0),
        }
        for key, value in row.items():
            metrics[key].append(value)
            per_case[labels[query_index]][key].append(value)
        top_ids.append(top_id)

    summaries = {key: distribution(values) for key, values in metrics.items()}
    case_summaries = {
        case_id: {key: distribution(values) for key, values in case_metrics.items()}
        for case_id, case_metrics in per_case.items()
    }
    median_byte_fraction = summaries["executable_total_byte_fraction"]["median"]
    p90_byte_fraction = summaries["executable_total_byte_fraction"]["p90"]
    row_norm_median_byte_fraction = summaries[
        "oracle_row_norm_total_byte_fraction"
    ]["median"]
    row_norm_p90_byte_fraction = summaries[
        "oracle_row_norm_total_byte_fraction"
    ]["p90"]
    # Lm-head is about 2.0575 ms of the 8.079 ms strict TPOT. Estimate only
    # bytes saved and charge no bound/index overhead: an optimistic ceiling.
    optimistic_saving_ms = 2.0575 * (1.0 - median_byte_fraction)
    optimistic_whole_step_speedup = 8.079 / (8.079 - optimistic_saving_ms)
    row_norm_optimistic_saving_ms = 2.0575 * (1.0 - row_norm_median_byte_fraction)
    row_norm_optimistic_whole_step_speedup = 8.079 / (
        8.079 - row_norm_optimistic_saving_ms
    )
    viable = (
        all_bounds_valid
        and all_winners_survive
        and p90_byte_fraction <= 0.75
        and optimistic_whole_step_speedup >= 1.03
    )
    row_norm_viable = (
        row_norm_p90_byte_fraction <= 0.75
        and row_norm_optimistic_whole_step_speedup >= 1.03
    )
    result = {
        "schema_version": "qwen35-exact-vocab-pruning-feasibility-v1",
        "status": "PASS",
        "scope": {
            "model": str(model_path),
            "config_sha256": sha256(model_path / "config.json"),
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16 model and hidden states; FP32 offline index construction and conservative bounds",
            "weight_shape": weight_shape,
            "queries": len(labels),
            "cases": len(NATURAL_REQUESTS),
            "generated_tokens_per_case": args.new_tokens,
        },
        "index": {
            "clusters": args.clusters,
            "block_dims": args.block_dims,
            "blocks": blocks,
            "kmeans_iterations": args.kmeans_iterations,
            "mean_squared_objective_by_iteration": objectives,
            "final_mean_squared_objective": final_objective,
            "minimum_cluster_rows": int(counts.min().item()),
            "median_cluster_rows": float(counts.float().median().item()),
            "maximum_cluster_rows": int(counts.max().item()),
            "maximum_block_radius": maximum_radius,
        },
        "correctness": {
            "all_empirical_group_bounds_valid": all_bounds_valid,
            "maximum_observed_bound_violation": max(metrics["bound_violation"]),
            "all_true_winner_clusters_survive_executable_bound": all_winners_survive,
            "representative_seed_top1_hit_rate": seed_hits / len(labels),
        },
        "distributions": summaries,
        "per_case": case_summaries,
        "generated_token_ids": generated,
        "reference_top_token_ids_sha256": hashlib.sha256(
            json.dumps(top_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "optimistic_projection": {
            "strict_frontier_tpot_ms": 8.079,
            "lm_head_ms": 2.0575,
            "median_total_byte_fraction": median_byte_fraction,
            "optimistic_saving_ms": optimistic_saving_ms,
            "optimistic_whole_step_speedup": optimistic_whole_step_speedup,
            "oracle_row_norm_median_total_byte_fraction": row_norm_median_byte_fraction,
            "oracle_row_norm_p90_total_byte_fraction": row_norm_p90_byte_fraction,
            "oracle_row_norm_optimistic_saving_ms": row_norm_optimistic_saving_ms,
            "oracle_row_norm_optimistic_whole_step_speedup": row_norm_optimistic_whole_step_speedup,
            "warning": "This assumes latency scales with bytes and charges zero cost for bound evaluation, indexing, gathers and irregular surviving groups."
        },
        "decision": (
            "IMPLEMENT_GPU_CLUSTER_PROTOTYPE"
            if viable
            else "INVESTIGATE_ROW_NORM_INDEX"
            if row_norm_viable
            else "STOP_EXACT_NORM_AND_CLUSTER_BOUNDS_TOO_LOOSE"
        ),
        "reopen_condition": "Use a materially tighter certified index whose p90 total byte fraction is <=0.75 on a larger hidden-state corpus.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
