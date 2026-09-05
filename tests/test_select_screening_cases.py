from scripts.select_screening_cases import select_cases


def _case(identifier: str, total: int, sequences: int) -> dict:
    return {"id": identifier, "parameters": {"total_seq_len": total, "num_seqs": sequences}}


def test_selects_both_routes_and_sequence_extremes() -> None:
    workload = {"cases": [
        _case("cp-short", 8, 1),
        _case("cp-mid", 64, 2),
        _case("cp-long", 4096, 3),
        _case("regular-short", 128, 8),
        _case("regular-mid", 1024, 16),
        _case("regular-long", 8192, 64),
    ]}
    selected = select_cases(
        workload,
        heads=8,
        sm_count=170,
        threshold_numerator=1,
        threshold_denominator=3,
    )
    assert selected["claim_scope"] == "DISCOVERY_ONLY_NOT_PRODUCTION_ACCEPTANCE"
    assert selected["population"] == {"total": 6, "cp": 3, "non_cp": 3}
    assert {case["id"] for case in selected["cases"]} == {
        "cp-short", "cp-mid", "cp-long",
        "regular-short", "regular-mid", "regular-long",
    }
