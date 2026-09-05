#!/usr/bin/env python3
"""Regression test for CUDA Graph replay accounting in the cadence audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20260902_qwen35_08b_e2e_sm89_v1"


def test_cuda_graph_replay_is_not_misclassified_as_host_time(tmp_path: Path) -> None:
    output = tmp_path / "cadence.json"
    subprocess.run(
        [
            sys.executable,
            str(RUN / "tools" / "analyze_vllm_decode_cadence.py"),
            "--sqlite",
            str(RUN / "profiles" / "nsys2025_int8_bf16_rerank_g1024k128.sqlite"),
            "--trace",
            str(RUN / "traces" / "vllm_nsys2025_int8_bf16_rerank_g1024k128_w1_n1_t32.json"),
            "--generated-steps",
            "192",
            "--request-count",
            "6",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["schema_version"] == "vllm-decode-cadence-bound-v2"
    assert result["status"] == "PASS"
    cuda = result["cuda"]
    assert cuda["graph_trace_count"] == cuda["graph_launch_count"] == 186
    assert 2.8 < cuda["graph_replay_duration_ms"]["median"] < 3.2
    assert cuda["outside_observed_gpu_activity_fraction_of_tpot_lower_bound"] < 0.1
    assert cuda["within_request_gpu_busy_union_ms"]["mean"] > 3.9
    assert cuda["within_request_gpu_idle_ms"]["mean_fraction_of_launch_cadence"] < 0.1
    assert "72%" in result["correction"]["retracted_interpretation"]
