from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def test_trace_intake_preserves_every_case_and_equal_weight(tmp_path: Path) -> None:
    definition = tmp_path / "definition.json"
    workloads = tmp_path / "workloads.jsonl"
    operator = tmp_path / "operator.json"
    workload = tmp_path / "workload.json"
    write(
        definition,
        {
            "name": "trace-op",
            "description": "y = x",
            "constraints": ["N > 0"],
            "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
            "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
            "reference": "def run(x): return x",
        },
    )
    records = [
        {
            "definition": "trace-op",
            "workload": {
                "uuid": f"case-{index}",
                "axes": {"N": size},
                "inputs": {"x": {"type": "safetensors", "path": f"blob/{index}.safetensors"}},
            },
        }
        for index, size in enumerate((1, 16, 4096))
    ]
    workloads.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/kernel_opt.py"),
            "trace-intake",
            "--definition",
            str(definition),
            "--workloads",
            str(workloads),
            "--dataset-revision",
            "fixture-revision",
            "--operator-output",
            str(operator),
            "--workload-output",
            str(workload),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    frozen_operator = json.loads(operator.read_text(encoding="utf-8"))
    frozen_workload = json.loads(workload.read_text(encoding="utf-8"))
    assert frozen_operator["source_identity"]["record_count"] == 3
    assert [case["id"] for case in frozen_workload["cases"]] == ["case-0", "case-1", "case-2"]
    assert sum(case["weight"] for case in frozen_workload["cases"]) == 1.0
    assert [case["boundary"] for case in frozen_workload["cases"]] == [True, False, True]
    assert "baseline_gpu_active_us / candidate_gpu_active_us" in frozen_workload["objective"]["aggregation"]
