#!/usr/bin/env python3
"""Build, legality-check and screen vLLM's Humming W4A16 backend."""

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
SITE = Path("/home/aden/.venvs/qwen35-vllm-4060/lib/python3.12/site-packages")
CUDA13_LIB = SITE / "nvidia/cu13/lib"
BACKEND_SOURCE = SITE / "vllm/model_executor/kernels/linear/mixed_precision/humming.py"
REGISTRY_SOURCE = SITE / "vllm/model_executor/kernels/linear/__init__.py"
HUMMING_INIT = SITE / "humming/__init__.py"
HUMMING_KERNEL = SITE / "humming/kernel/humming.py"
HUMMING_TUNER = SITE / "humming/tune/sm8x.py"
HUMMING_LAUNCHER = SITE / "humming/_native/x86_64/libhumming_launcher.so"
EXPECTED = {
    HARNESS: "21705dceaa1899f2df5a80cb5fc0a1e78314d0f7701435178f241e9e1c2afbe4",
    BACKEND_SOURCE: "2cfbe2bbf501b3e99b46516e0a595dd78e8f70cb71b06eef4ca72b8a8521772b",
    REGISTRY_SOURCE: "d0710fdbe617209ef99e22e053a0ffc8a0ece8c83d04eea5083f6aaa8c19d2d8",
    HUMMING_INIT: "4ada2947dc99e51c5441f71f8360b857fba7c6afded7b041acc3032ea7a35ef6",
    HUMMING_KERNEL: "97712961bfcfbfe4057211564cea439d08d63d372e7eb0fd000dbaf35f675a2b",
    HUMMING_TUNER: "421d6e7a31f8390fc793b974d87670ee302cc8dd02d091e65bdc6faf8d012d12",
    HUMMING_LAUNCHER: "97623b4157123f59bd8506727f7b52a9d8f934ebdc9329bbd5db164af5b088ff",
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
        "class HummingLinearKernel",
        "return 75",
        "Humming does not support act-order (g_idx)",
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise RuntimeError(f"backend source lost required contracts: {missing}")
    if '"humming": {' not in registry or "HummingLinearKernel" not in registry:
        raise RuntimeError("Humming backend is not registered")
    print(json.dumps({"status": "PASS", "sources": {str(p): digest(p) for p in EXPECTED}}))


def correctness() -> None:
    verify_sources()
    operator = json.loads((RUN / "operator.json").read_text(encoding="utf-8"))
    forbidden = operator["computation"]["forbidden_rewrites"]
    if not any("FP16 activations" in item for item in forbidden):
        raise RuntimeError("operator contract no longer freezes BF16 activations")
    model_config = json.loads(
        Path("/home/aden/models/Qwen3.5-0.8B-W4A16-AutoRound-GPTQ-g128/config.json").read_text(
            encoding="utf-8"
        )
    )
    quantization = model_config.get("quantization_config", {})
    if quantization.get("desc_act") is not False:
        raise RuntimeError("Humming legality requires desc_act=false (no g_idx)")
    print(
        json.dumps(
            {
                "status": "PASS",
                "contract": digest(RUN / "operator.json"),
                "desc_act": quantization.get("desc_act"),
            }
        )
    )


def mean_case(raw: dict, key: str) -> float:
    values = [float(case[key]) for case in raw["cases"]]
    return sum(values) / len(values)


def smoke() -> None:
    verify_sources()
    baseline_path = RUN / "raw/stock_marlin_natural_b1_w1_n1_t64.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_root = RUN / "candidates/humming-w4a16-all"
    raw_path = candidate_root / "raw.json"
    log_path = candidate_root / "smoke_execution.log"
    smoke_path = candidate_root / "smoke_result.json"
    if raw_path.is_file() and log_path.is_file():
        execution_log = log_path.read_text(encoding="utf-8")
    else:
        cache_parent = Path(tempfile.mkdtemp(prefix="vllm-gptq-humming-screen-", dir="/tmp"))
        cache_root = cache_parent / "fresh-cache"
        environment = os.environ.copy()
        prior_library_path = environment.get("LD_LIBRARY_PATH", "")
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
                # Humming launches its bundled NVRTC helper as a subprocess.
                # The CUDA 13 wheel keeps libnvrtc-builtins beside libnvrtc,
                # outside the system loader's default CUDA 12 search path.
                "LD_LIBRARY_PATH": str(CUDA13_LIB)
                + ((":" + prior_library_path) if prior_library_path else ""),
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
            "--linear-backend", "humming",
            "--gpu-telemetry",
            "--require-empty-vllm-cache-root",
            "--expect-gdn-decode-kernel", "cuda",
            "--expect-marlin-w4-rerank", "off",
            "--expect-marlin-w4-scan-only", "off",
            "--expect-source-sha256", f"{BACKEND_SOURCE}={EXPECTED[BACKEND_SOURCE]}",
            "--expect-source-sha256", f"{REGISTRY_SOURCE}={EXPECTED[REGISTRY_SOURCE]}",
            "--expect-source-sha256", f"{HUMMING_KERNEL}={EXPECTED[HUMMING_KERNEL]}",
            "--expect-source-sha256", f"{HUMMING_TUNER}={EXPECTED[HUMMING_TUNER]}",
            "--expect-source-sha256", f"{HUMMING_LAUNCHER}={EXPECTED[HUMMING_LAUNCHER]}",
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
    expected_path = "HummingLinearKernel"
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
    execution_status = "PASS" if candidate.get("status") == "PASS" and observed_count else "FAIL"
    exact_case_count = sum(case["correctness"] == "PASS" for case in cases)
    result = {
        "schema_version": "candidate-smoke-result-v3",
        "status": execution_status,
        "candidate_id": "humming-w4a16-all",
        "claim_scope": "DISCOVERY_ONLY_NOT_PRODUCTION_ACCEPTANCE",
        "cases": cases,
        "correctness_gate": {
            "status": "PASS" if exact_case_count == len(cases) else "FAIL",
            "exact_cases": exact_case_count,
            "required_exact_cases": len(cases),
        },
        "reachability": {
            "status": "PASS" if observed_count else "FAIL",
            "expected_path": expected_path,
            "observed_path": expected_path if observed_count else "NONE",
            "compile_cache_policy": "FRESH",
            "execution_proof": {
                "kind": "INSTRUMENTED_CALL_COUNT",
                "scope": "vLLM backend-selection log entries",
                "observed_count": observed_count,
                "minimum_count": 1,
                "evidence_index": 0,
            },
            "evidence": [
                {"path": "candidates/humming-w4a16-all/smoke_execution.log", "sha256": digest(log_path)}
            ],
        },
        "objective": {
            "direction": "minimize",
            "baseline": mean_case(baseline, "median_end_to_end_ms") * 1000.0,
            "candidate": mean_case(candidate, "median_end_to_end_ms") * 1000.0,
            "unit": "us",
        },
        "raw": {
            "baseline": {
                "path": "raw/stock_marlin_natural_b1_w1_n1_t64.json",
                "sha256": digest(baseline_path),
            },
            "candidate": {
                "path": "candidates/humming-w4a16-all/raw.json",
                "sha256": digest(raw_path),
            },
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
