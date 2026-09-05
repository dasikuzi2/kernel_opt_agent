#!/usr/bin/env python3
"""Regression tests for the conservative Marlin node-profile split."""

from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "runs/20260902_qwen35_08b_e2e_sm89_v1/tools/analyze_marlin_node_profile.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_marlin_node_profile", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _make_trace(path: Path, long_duration: int, short_duration: int) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table StringIds(id integer primary key, value text not null);
        create table CUPTI_ACTIVITY_KIND_KERNEL(
            start integer not null, end integer not null,
            demangledName integer not null,
            gridX integer not null, gridY integer not null, gridZ integer not null,
            blockX integer not null, blockY integer not null, blockZ integer not null
        );
        insert into StringIds values(1, 'void marlin::Marlin<int4>()');
        insert into StringIds values(2, 'other_kernel');
        insert into StringIds values(3, 'void marlin::Marlin<int4, prefill>()');
        """
    )
    cursor = 0
    rows = []
    # Two generated steps, each with one long head launch and three backbone launches.
    for _ in range(2):
        for duration in (long_duration, short_duration, short_duration, short_duration):
            rows.append((cursor, cursor + duration, 1, 24, 1, 1, 256, 1, 1))
            cursor += duration
    rows.append((cursor, cursor + 100, 2, 1, 1, 1, 32, 1, 1))
    cursor += 100
    # Same geometry, different template: it must not contaminate the dominant
    # decode signature or its integral per-step repetition.
    rows.append((cursor, cursor + 250, 3, 24, 1, 1, 256, 1, 1))
    connection.executemany(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?,?)", rows
    )
    connection.commit()
    connection.close()


def test_clear_stratum_is_split_and_bounded() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "clear.sqlite"
        _make_trace(path, long_duration=1000, short_duration=100)
        result = MODULE.analyze(path, 2, "marlin::Marlin", minimum_gap_ratio=5.0)
    assert result["status"] == "PASS"
    assert result["dominant_launch_signature"]["instances_per_generated_step"] == 4
    assert result["dominant_launch_signature"]["inferred_backbone_instances_per_decode_step"] == 3
    assert result["components"]["lm_head_stratum"]["instances"] == 2
    assert result["components"]["dominant_backbone_stratum"]["instances"] == 6
    assert result["interpretation"]["selected_next_target"] == "dominant_backbone_stratum"


def test_ambiguous_duration_stratum_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "ambiguous.sqlite"
        _make_trace(path, long_duration=400, short_duration=100)
        result = MODULE.analyze(path, 2, "marlin::Marlin", minimum_gap_ratio=5.0)
    assert result["status"] == "FAIL"
    assert result["interpretation"]["selected_next_target"] == "NONE"
