#!/usr/bin/env python3
"""Validate and screen the lossless SM89 BF16 output-head layout."""

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
UTILS = VLLM_ROOT / "model_executor/layers/utils.py"
EMBEDDING = VLLM_ROOT / "model_executor/layers/vocab_parallel_embedding.py"
EXPECTED = {
    HARNESS: "21705dceaa1899f2df5a80cb5fc0a1e78314d0f7701435178f241e9e1c2afbe4",
    UTILS: "2b9e6193612d83762e76a2af50dfdb2348ffa61e4711d666e6d678f8b61d4e43",
    EMBEDDING: "a5d271fe4a73b44967c8ad1e2b0b4b9153429a021a966bc1b0459f1f5a8fc3e4",
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
    source = UTILS.read_text(encoding="utf-8")
    embedding = EMBEDDING.read_text(encoding="utf-8")
    required = (
        "def _sm89_exact_packed_bf16_gemv_kernel(",
        "def _get_sm89_exact_packed_weight(",
        'os.environ.get("VLLM_SM89_EXACT_PACKED_LM_HEAD", "0") == "1"',
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise RuntimeError(f"candidate source lost required implementation: {missing}")
    if "sm89_bf16_gemv_impl(warmup_input, layer.weight.data)" not in embedding:
        raise RuntimeError("load-time pack and JIT warm hook is absent")
    print(json.dumps({"status": "PASS", "sources": {str(p): digest(p) for p in EXPECTED}}))


def correctness() -> None:
    verify_sources()
    operator = json.loads((RUN / "operator.json").read_text(encoding="utf-8"))
    forbidden = operator["computation"]["forbidden_rewrites"]
    if not any("quantization" in item.lower() for item in forbidden):
        raise RuntimeError("operator contract no longer forbids additional quantization")
    source = UTILS.read_text(encoding="utf-8")
    if "weight.view(torch.int16)" not in source or "fallback_bits" not in source:
        raise RuntimeError("bit-preserving reconstruction and fallback are absent")
    print(json.dumps({"status": "PASS", "contract": digest(RUN / "operator.json")}))


def mean_case(raw: dict, key: str) -> float:
    values = [float(case[key]) for case in raw["cases"]]
    return sum(values) / len(values)


def smoke() -> None:
    verify_sources()
    baseline_path = RUN / "raw/stock_marlin_natural_b1_w1_n1_t64.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_root = RUN / "candidates/exact-packed-bf16-lmhead"
    raw_path = candidate_root / "raw.json"
    log_path = candidate_root / "smoke_execution.log"
    smoke_path = candidate_root / "smoke_result.json"
    if raw_path.is_file() and log_path.is_file():
        execution_log = log_path.read_text(encoding="utf-8")
    else:
        cache_parent = Path(tempfile.mkdtemp(prefix="vllm-gptq-exact-head-screen-", dir="/tmp"))
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
                "VLLM_SM89_BF16_LM_HEAD": "1",
                "VLLM_SM89_EXACT_PACKED_LM_HEAD": "1",
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
            "--linear-backend", "marlin",
            "--gpu-telemetry",
            "--require-empty-vllm-cache-root",
            "--expect-gdn-decode-kernel", "cuda",
            "--expect-sm89-lm-head", "triton",
            "--expect-exact-packed-lm-head", "on",
            "--expect-marlin-w4-rerank", "off",
            "--expect-marlin-w4-scan-only", "off",
            "--expect-source-sha256", f"{UTILS}={EXPECTED[UTILS]}",
            "--expect-source-sha256", f"{EMBEDDING}={EXPECTED[EMBEDDING]}",
        ]
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=360,
        )
        execution_log = completed.stdout
        log_path.write_text(execution_log, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"production harness failed; see {log_path}")

    candidate = json.loads(raw_path.read_text(encoding="utf-8"))
    sentinel = "SM89 exact-packed lm_head cache ready"
    observed_count = execution_log.count(sentinel)
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
    exact_case_count = sum(case["correctness"] == "PASS" for case in cases)
    reached = observed_count >= 1
    execution_status = "PASS" if candidate.get("status") == "PASS" and reached else "FAIL"
    result = {
        "schema_version": "candidate-smoke-result-v3",
        "status": execution_status,
        "candidate_id": "exact-packed-bf16-lmhead",
        "claim_scope": "DISCOVERY_ONLY_NOT_PRODUCTION_ACCEPTANCE",
        "cases": cases,
        "correctness_gate": {
            "status": "PASS" if exact_case_count == len(cases) else "FAIL",
            "exact_cases": exact_case_count,
            "required_exact_cases": len(cases),
        },
        "reachability": {
            "status": "PASS" if reached else "FAIL",
            "expected_path": "SM89 exact-packed BF16 lm_head",
            "observed_path": "SM89 exact-packed BF16 lm_head" if reached else "NONE",
            "compile_cache_policy": "FRESH",
            "execution_proof": {
                "kind": "INSTRUMENTED_CALL_COUNT",
                "scope": "vLLM exact-packed cache-ready log sentinel",
                "observed_count": observed_count,
                "minimum_count": 1,
                "evidence_index": 0,
            },
            "evidence": [{"path": "candidates/exact-packed-bf16-lmhead/smoke_execution.log", "sha256": digest(log_path)}],
        },
        "objective": {
            "direction": "minimize",
            "baseline": mean_case(baseline, "median_end_to_end_ms") * 1000.0,
            "candidate": mean_case(candidate, "median_end_to_end_ms") * 1000.0,
            "unit": "us",
        },
        "raw": {
            "baseline": {"path": "raw/stock_marlin_natural_b1_w1_n1_t64.json", "sha256": digest(baseline_path)},
            "candidate": {"path": "candidates/exact-packed-bf16-lmhead/raw.json", "sha256": digest(raw_path)},
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
