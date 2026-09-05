import json

import pytest

from scripts.materialize_flashinfer_subset import filter_workloads


def _line(identifier: str) -> str:
    return json.dumps({"definition": "op", "workload": {"uuid": identifier}})


def test_filter_workloads_is_exact_and_deterministic() -> None:
    result = filter_workloads([_line("b"), _line("a"), _line("c")], {"a", "c"})
    assert [json.loads(line)["workload"]["uuid"] for line in result] == ["a", "c"]


def test_filter_workloads_fails_when_manifest_is_stale() -> None:
    with pytest.raises(ValueError, match="absent"):
        filter_workloads([_line("a")], {"missing"})
