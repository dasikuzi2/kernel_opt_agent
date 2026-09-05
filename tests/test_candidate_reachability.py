#!/usr/bin/env python3
"""Dependency-free tests for candidate path/source/cache reachability guards."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from candidate_discovery import validate_smoke_result

SCRIPT = (
    ROOT
    / "runs"
    / "20260902_qwen35_08b_e2e_sm89_v1"
    / "tools"
    / "benchmark_vllm_offline.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_vllm_offline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "candidate.py"
        source.write_bytes(b"candidate-v1\n")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()

        observed = MODULE.expected_source_hashes([f"{source}={digest}"])
        assert observed[str(source.resolve())] == digest

        try:
            MODULE.expected_source_hashes([f"{source}={'0' * 64}"])
        except RuntimeError as exc:
            assert "candidate source mismatch" in str(exc)
        else:
            raise AssertionError("mismatched source hash was accepted")

        empty_cache = root / "empty-cache"
        assert MODULE.require_empty_vllm_cache_root(str(empty_cache)).endswith(
            "empty-cache"
        )
        empty_cache.mkdir()
        (empty_cache / "stale-artifact").write_text("stale", encoding="utf-8")
        try:
            MODULE.require_empty_vllm_cache_root(str(empty_cache))
        except RuntimeError as exc:
            assert "may be stale" in str(exc)
        else:
            raise AssertionError("non-empty compile cache was accepted")

        run = root / "run"
        candidate_source = run / "candidates" / "c1" / "kernel.py"
        candidate_source.parent.mkdir(parents=True)
        candidate_source.write_text("candidate = True\n", encoding="utf-8")
        smoke_path = run / "smoke.json"
        smoke = {
            "schema_version": "candidate-smoke-result-v3",
            "status": "PASS",
            "candidate_id": "c1",
            "objective": {
                "direction": "minimize",
                "baseline": 10.0,
                "candidate": 8.0,
            },
            "cases": [
                {"case_id": "anchor", "role": "ANCHOR"},
                {"case_id": "edge", "role": "EDGE"},
            ],
            "reachability": {
                "status": "PASS",
                "expected_path": "candidate-kernel",
                "observed_path": "fallback-kernel",
                "compile_cache_policy": "SOURCE_HASHED",
                "execution_proof": {
                    "kind": "KERNEL_INSTANCE_COUNT",
                    "scope": "candidate kernel inside compiled decode graph",
                    "observed_count": 1,
                    "minimum_count": 1,
                    "evidence_index": 0,
                },
                "evidence": [
                    {
                        "path": "candidates/c1/kernel.py",
                        "sha256": hashlib.sha256(
                            candidate_source.read_bytes()
                        ).hexdigest(),
                    }
                ],
            },
        }
        smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
        try:
            validate_smoke_result(run, smoke_path, {"candidate_id": "c1"})
        except ValueError as exc:
            assert "execution path was not reached" in str(exc)
        else:
            raise AssertionError("unreachable candidate passed the smoke gate")

        smoke["reachability"]["observed_path"] = "candidate-kernel"
        smoke["reachability"]["execution_proof"]["observed_count"] = 0
        smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
        try:
            validate_smoke_result(run, smoke_path, {"candidate_id": "c1"})
        except ValueError as exc:
            assert "execution count" in str(exc)
        else:
            raise AssertionError("zero runtime kernel count passed reachability")

        smoke["reachability"]["execution_proof"].update(
            {"kind": "DIRECT_SENTINEL", "observed_count": 1}
        )
        smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
        try:
            validate_smoke_result(run, smoke_path, {"candidate_id": "c1"})
        except ValueError as exc:
            assert "compiled candidates" in str(exc)
        else:
            raise AssertionError("compiled candidate accepted a host-only sentinel")

    print("candidate reachability test: PASS")


if __name__ == "__main__":
    main()
