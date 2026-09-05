#!/usr/bin/env python3
"""Dependency-free tests for the generation degeneration gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_generation_quality.py"


def run(payload: dict, expected: int) -> dict:
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "trace.json"
        output = Path(temp) / "quality.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(output)],
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        assert completed.returncode == expected, completed.stderr
        return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    valid = run(
        {"cases": [{"case_id": "normal", "generated_token_ids": list(range(32))}]},
        0,
    )
    assert valid["status"] == "PASS"

    degenerate = run(
        {
            "cases": [
                {
                    "case_id": "special-token-loop",
                    "generated_token_ids": [248068, 271, 248069, 271] * 32,
                }
            ]
        },
        2,
    )
    assert degenerate["status"] == "FAIL"
    assert degenerate["cases"][0]["reason"] == "degenerate low-diversity token cycle"

    malformed = run({"output": {"generated_token_ids": [1, "bad"]}}, 2)
    assert malformed["status"] == "FAIL"
    print("generation quality gate test: PASS")


if __name__ == "__main__":
    main()
