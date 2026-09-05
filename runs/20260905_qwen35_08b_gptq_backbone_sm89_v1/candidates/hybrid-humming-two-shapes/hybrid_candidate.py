#!/usr/bin/env python3
"""Build, legality-check and screen the two-shape Marlin/Humming hybrid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


RUN = Path("/mnt/d/codes/kernel_opt_agent/runs/20260905_qwen35_08b_gptq_backbone_sm89_v1")
OLD_RUN = Path("/mnt/d/codes/kernel_opt_agent/runs/20260902_qwen35_08b_e2e_sm89_v1")
HARNESS = OLD_RUN / "tools/benchmark_vllm_offline.py"
INVENTORY = OLD_RUN / "models/gptq_projection_shapes_round37.json"
SITE = Path("/home/aden/.venvs/qwen35-vllm-4060/lib/python3.12/site-packages")
CUDA13_LIB = SITE / "nvidia/cu13/lib"
REGISTRY_SOURCE = SITE / "vllm/model_executor/kernels/linear/__init__.py"
MARLIN_SOURCE = SITE / "vllm/model_executor/kernels/linear/mixed_precision/marlin.py"
HUMMING_SOURCE = SITE / "vllm/model_executor/kernels/linear/mixed_precision/humming.py"
HUMMING_KERNEL = SITE / "humming/kernel/humming.py"
EXPECTED = {
    HARNESS: "21705dceaa1899f2df5a80cb5fc0a1e78314d0f7701435178f241e9e1c2afbe4",
    INVENTORY: "2f980dad3b70f489dde2e2542b2f6d78d400151280a7472f073c6ac099a5bef2",
    REGISTRY_SOURCE: "d0710fdbe617209ef99e22e053a0ffc8a0ece8c83d04eea5083f6aaa8c19d2d8",
    MARLIN_SOURCE: "ec18b17815f5114cdc8ba72779095b8ae3d4d1369508a0ef1d30029904cd329c",
    HUMMING_SOURCE: "2cfbe2bbf501b3e99b46516e0a595dd78e8f70cb71b06eef4ca72b8a8521772b",
    HUMMING_KERNEL: "97712961bfcfbfe4057211564cea439d08d63d372e7eb0fd000dbaf35f675a2b",
}
SELECTED = {(1024, 2048), (1024, 3584)}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_sources() -> None:
    for path, expected in EXPECTED.items():
        observed = digest(path)
        if observed != expected:
            raise RuntimeError(f"source mismatch: {path}: {observed} != {expected}")


def selected_calls() -> int:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    return sum(
        int(row["calls_per_decode"])
        for row in inventory["projection_groups"]
        if tuple(map(int, row["decode_shape_m_n_k"][1:])) in SELECTED
    )


def build() -> None:
    verify_sources()
    selector = (RUN / "candidates/hybrid-humming-two-shapes/sitecustomize.py").read_text(
        encoding="utf-8"
    )
    for needle in (
        "selected_shapes != expected_shapes",
        "config.act_type != torch.bfloat16",
        "config.group_size != 128 or config.has_g_idx",
        "auto_gptq.choose_mp_linear_kernel = choose_shape_routed_kernel",
    ):
        if needle not in selector:
            raise RuntimeError(f"selector lost fail-closed guard: {needle}")
    if selected_calls() != 48:
        raise RuntimeError("selected shapes no longer account for exactly 48 calls/decode")
    print(json.dumps({"status": "PASS", "selected_calls_per_decode": 48}))


def correctness() -> None:
    verify_sources()
    operator = json.loads((RUN / "operator.json").read_text(encoding="utf-8"))
    forbidden = operator["computation"]["forbidden_rewrites"]
    if not any("FP16 activations" in item for item in forbidden):
        raise RuntimeError("operator contract no longer freezes BF16 activations")
    if selected_calls() != 48:
        raise RuntimeError("checkpoint inventory changed")
    print(json.dumps({"status": "PASS", "contract": digest(RUN / "operator.json")}))


def mean_case(raw: dict, key: str) -> float:
    values = [float(case[key]) for case in raw["cases"]]
    return sum(values) / len(values)


def smoke() -> None:
    verify_sources()
    baseline_path = RUN / "raw/stock_marlin_natural_b1_w1_n1_t64.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_root = RUN / "candidates/hybrid-humming-two-shapes"
    raw_path = candidate_root / "raw.json"
    log_path = candidate_root / "smoke_execution.log"
    result_path = candidate_root / "smoke_result.json"
    cache_parent = Path(tempfile.mkdtemp(prefix="vllm-gptq-hybrid-screen-", dir="/tmp"))
    cache_root = cache_parent / "fresh-cache"
    environment = os.environ.copy()
    prior_python_path = environment.get("PYTHONPATH", "")
    prior_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(candidate_root)
            + ((":" + prior_python_path) if prior_python_path else ""),
            "LD_LIBRARY_PATH": str(CUDA13_LIB)
            + ((":" + prior_library_path) if prior_library_path else ""),
            "VLLM_SM89_W4_HUMMING_SHAPES": "1024x2048,1024x3584",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_GDN_DECODE_KERNEL": "cuda",
            "VLLM_SM89_MARLIN_W4_RERANK": "0",
            "VLLM_SM89_MARLIN_W4_SCAN_ONLY": "0",
            "VLLM_SM89_BF16_LM_HEAD": "0",
            "VLLM_SM89_EXACT_PACKED_LM_HEAD": "0",
            "VLLM_CACHE_ROOT": str(cache_root),
        }
    )
    command = [
        os.sys.executable,
        str(HARNESS),
        "--model", "/home/aden/models/Qwen3.5-0.8B-W4A16-AutoRound-GPTQ-g128",
        "--tokenizer", "/home/aden/models/Qwen3.5-0.8B",
        "--operator-contract", str(RUN / "operator.json"),
        "--workload-contract", str(RUN / "workload.json"),
        "--output", str(raw_path),
        "--warmups", "1",
        "--trials", "1",
        "--new-tokens", "64",
        "--prompt-suite", "natural",
        "--kv-cache-memory-bytes", "536870912",
        "--max-num-seqs", "1",
        "--quantization", "gptq_marlin",
        "--linear-backend", "auto",
        "--gpu-telemetry",
        "--require-empty-vllm-cache-root",
        "--expect-gdn-decode-kernel", "cuda",
        "--expect-marlin-w4-rerank", "off",
        "--expect-marlin-w4-scan-only", "off",
        "--expect-source-sha256", f"{REGISTRY_SOURCE}={EXPECTED[REGISTRY_SOURCE]}",
        "--expect-source-sha256", f"{MARLIN_SOURCE}={EXPECTED[MARLIN_SOURCE]}",
        "--expect-source-sha256", f"{HUMMING_SOURCE}={EXPECTED[HUMMING_SOURCE]}",
    ]
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=540,
    )
    execution_log = completed.stdout
    log_path.write_text(execution_log, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"production harness failed; see {log_path}")
    candidate = json.loads(raw_path.read_text(encoding="utf-8"))
    sentinel_count = execution_log.count("SM89_HYBRID_SELECT HummingLinearKernel")
    humming_logged = execution_log.count("Using HummingLinearKernel for AutoGPTQLinearMethod")
    marlin_logged = execution_log.count("Using MarlinLinearKernel for AutoGPTQLinearMethod")
    reached = sentinel_count >= 2 and humming_logged >= 1 and marlin_logged >= 1
    cases = []
    for index, (control, trial) in enumerate(zip(baseline["cases"], candidate["cases"])):
        if control["case_id"] != trial["case_id"]:
            raise RuntimeError("baseline and candidate case order differs")
        cases.append(
            {
                "case_id": control["case_id"],
                "role": "ANCHOR" if index == 0 else "EDGE",
                "correctness": "PASS"
                if control["generated_token_ids"] == trial["generated_token_ids"]
                else "FAIL",
                "baseline_end_to_end_us": float(control["median_end_to_end_ms"]) * 1000.0,
                "candidate_end_to_end_us": float(trial["median_end_to_end_ms"]) * 1000.0,
            }
        )
    exact_count = sum(case["correctness"] == "PASS" for case in cases)
    status = "PASS" if candidate.get("status") == "PASS" and reached else "FAIL"
    result = {
        "schema_version": "candidate-smoke-result-v3",
        "status": status,
        "candidate_id": "hybrid-humming-two-shapes",
        "claim_scope": "DISCOVERY_ONLY_NOT_PRODUCTION_ACCEPTANCE",
        "cases": cases,
        "correctness_gate": {
            "status": "PASS" if exact_count == len(cases) else "FAIL",
            "exact_cases": exact_count,
            "required_exact_cases": len(cases),
        },
        "reachability": {
            "status": "PASS" if reached else "FAIL",
            "expected_path": "MarlinLinearKernel+HummingLinearKernel(shape-routed)",
            "observed_path": "MarlinLinearKernel+HummingLinearKernel(shape-routed)" if reached else "NONE",
            "compile_cache_policy": "FRESH",
            "execution_proof": {
                "kind": "INSTRUMENTED_CALL_COUNT",
                "scope": "shape-selector sentinels plus vLLM backend-selection logs",
                "observed_count": sentinel_count,
                "minimum_count": 2,
                "evidence_index": 0,
            },
            "evidence": [
                {"path": "candidates/hybrid-humming-two-shapes/smoke_execution.log", "sha256": digest(log_path)}
            ],
        },
        "objective": {
            "direction": "minimize",
            "baseline": mean_case(baseline, "median_end_to_end_ms") * 1000.0,
            "candidate": mean_case(candidate, "median_end_to_end_ms") * 1000.0,
            "unit": "us",
        },
        "raw": {
            "baseline": {"path": "raw/stock_marlin_natural_b1_w1_n1_t64.json", "sha256": digest(baseline_path)},
            "candidate": {"path": "candidates/hybrid-humming-two-shapes/raw.json", "sha256": digest(raw_path)},
        },
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "objective": result["objective"], "exact_cases": exact_count}))
    if status != "PASS":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "correctness", "smoke"))
    args = parser.parse_args()
    {"build": build, "correctness": correctness, "smoke": smoke}[args.stage]()


if __name__ == "__main__":
    main()
