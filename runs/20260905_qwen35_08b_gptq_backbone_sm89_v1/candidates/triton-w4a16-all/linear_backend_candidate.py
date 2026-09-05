#!/usr/bin/env python3
"""Build, legality-check and screen one stock vLLM linear backend end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


RUN = Path("/mnt/d/codes/kernel_opt_agent/runs/20260905_qwen35_08b_gptq_backbone_sm89_v1")
HARNESS = Path("/mnt/d/codes/kernel_opt_agent/runs/20260902_qwen35_08b_e2e_sm89_v1/tools/benchmark_vllm_offline.py")
VLLM_ROOT = Path("/home/aden/.venvs/qwen35-vllm-4060/lib/python3.12/site-packages/vllm")
BACKEND_SOURCE = VLLM_ROOT / "model_executor/kernels/linear/mixed_precision/triton_w4a16.py"
REGISTRY_SOURCE = VLLM_ROOT / "model_executor/kernels/linear/__init__.py"
EXPECTED = {
    HARNESS: "21705dceaa1899f2df5a80cb5fc0a1e78314d0f7701435178f241e9e1c2afbe4",
    BACKEND_SOURCE: "3cf3d3681b1bb4efc06d7a7e43adafaf8d5ef019de4cfaadbed5bcece83d0ebb",
    REGISTRY_SOURCE: "d0710fdbe617209ef99e22e053a0ffc8a0ece8c83d04eea5083f6aaa8c19d2d8",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_sources() -> None:
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = digest(path)
        if observed != expected:
            raise RuntimeError(f"source mismatch: {path}: {observed} != {expected}")


def build() -> None:
    verify_sources()
    source = BACKEND_SOURCE.read_text(encoding="utf-8")
    registry = REGISTRY_SOURCE.read_text(encoding="utf-8")
    required = (
        "class TritonW4A16LinearKernel",
        "torch.float16, torch.bfloat16",
        "current_platform.is_rocm() or current_platform.is_cuda()",
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise RuntimeError(f"backend source lost required contracts: {missing}")
    if '"triton": {' not in registry or "TritonW4A16LinearKernel" not in registry:
        raise RuntimeError("Triton W4A16 backend is not registered")
    print(json.dumps({"status": "PASS", "sources": {str(p): digest(p) for p in EXPECTED}}))


def correctness() -> None:
    verify_sources()
    operator = json.loads((RUN / "operator.json").read_text(encoding="utf-8"))
    forbidden = operator["computation"]["forbidden_rewrites"]
    if not any("FP16 activations" in item for item in forbidden):
        raise RuntimeError("operator contract no longer freezes BF16 activations")
    if "torch.bfloat16" not in BACKEND_SOURCE.read_text(encoding="utf-8"):
        raise RuntimeError("candidate backend does not admit BF16 activations")
    print(json.dumps({"status": "PASS", "contract": digest(RUN / "operator.json")}))


def mean_case(raw: dict, key: str) -> float:
    values = [float(case[key]) for case in raw["cases"]]
    return sum(values) / len(values)


def smoke() -> None:
    verify_sources()
    baseline_path = RUN / "raw/stock_marlin_natural_b1_w1_n1_t64.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_root = RUN / "candidates/triton-w4a16-all"
    raw_path = candidate_root / "raw.json"
    log_path = candidate_root / "smoke_execution.log"
    smoke_path = candidate_root / "smoke_result.json"
    if raw_path.is_file() and log_path.is_file():
        # Attempt 1 already produced a complete causal output; a result-schema
        # repair must bind that immutable evidence rather than spending another
        # GPU run on the same candidate.
        execution_log = log_path.read_text(encoding="utf-8")
    else:
        cache_parent = Path(tempfile.mkdtemp(prefix="vllm-gptq-triton-screen-", dir="/tmp"))
        cache_root = cache_parent / "fresh-cache"
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
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
            "--linear-backend", "triton",
            "--gpu-telemetry",
            "--require-empty-vllm-cache-root",
            "--expect-gdn-decode-kernel", "cuda",
            "--expect-marlin-w4-rerank", "off",
            "--expect-marlin-w4-scan-only", "off",
            "--expect-source-sha256", f"{BACKEND_SOURCE}={EXPECTED[BACKEND_SOURCE]}",
            "--expect-source-sha256", f"{REGISTRY_SOURCE}={EXPECTED[REGISTRY_SOURCE]}",
        ]
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
        execution_log = completed.stdout
        log_path.write_text(execution_log, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"production harness failed; see {log_path}")
    candidate = json.loads(raw_path.read_text(encoding="utf-8"))
    expected_path = "TritonW4A16LinearKernel"
    observed_count = execution_log.count(expected_path)
    cases = []
    for index, (control, trial) in enumerate(zip(baseline["cases"], candidate["cases"])):
        if control["case_id"] != trial["case_id"]:
            raise RuntimeError("baseline and candidate case order differs")
        exact = control["generated_token_ids"] == trial["generated_token_ids"]
        cases.append(
            {
                "case_id": control["case_id"],
                "role": "ANCHOR" if index == 0 else "EDGE",
                "correctness": "PASS" if exact else "FAIL",
                "baseline_end_to_end_us": float(control["median_end_to_end_ms"]) * 1000.0,
                "candidate_end_to_end_us": float(trial["median_end_to_end_ms"]) * 1000.0,
            }
        )
    execution_status = (
        "PASS"
        if candidate.get("status") == "PASS"
        and observed_count >= 1
        else "FAIL"
    )
    exact_case_count = sum(case["correctness"] == "PASS" for case in cases)
    result = {
        "schema_version": "candidate-smoke-result-v3",
        # A successfully reached candidate is a valid causal screen even when
        # performance or numerical acceptance rejects it.  Those are not
        # compiler/harness failures and must not enter the technical-repair loop.
        "status": execution_status,
        "candidate_id": "triton-w4a16-all",
        "claim_scope": "DISCOVERY_ONLY_NOT_PRODUCTION_ACCEPTANCE",
        "cases": cases,
        "correctness_gate": {
            "status": "PASS" if exact_case_count == len(cases) else "FAIL",
            "exact_cases": exact_case_count,
            "required_exact_cases": len(cases),
        },
        "reachability": {
            "status": "PASS" if observed_count >= 1 else "FAIL",
            "expected_path": expected_path,
            "observed_path": expected_path if observed_count >= 1 else "NONE",
            "compile_cache_policy": "FRESH",
            "execution_proof": {
                "kind": "INSTRUMENTED_CALL_COUNT",
                "scope": "vLLM backend-selection log entries",
                "observed_count": observed_count,
                "minimum_count": 1,
                "evidence_index": 0,
            },
            "evidence": [{"path": "candidates/triton-w4a16-all/smoke_execution.log", "sha256": digest(log_path)}],
        },
        "objective": {
            "direction": "minimize",
            "baseline": mean_case(baseline, "median_end_to_end_ms") * 1000.0,
            "candidate": mean_case(candidate, "median_end_to_end_ms") * 1000.0,
            "unit": "us",
        },
        "raw": {
            "baseline": {"path": "raw/stock_marlin_natural_b1_w1_n1_t64.json", "sha256": digest(baseline_path)},
            "candidate": {"path": "candidates/triton-w4a16-all/raw.json", "sha256": digest(raw_path)},
        },
    }
    smoke_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": execution_status, "objective": result["objective"], "exact_cases": exact_case_count}))
    if execution_status != "PASS":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "correctness", "smoke"))
    args = parser.parse_args()
    {"build": build, "correctness": correctness, "smoke": smoke}[args.stage]()


if __name__ == "__main__":
    main()
