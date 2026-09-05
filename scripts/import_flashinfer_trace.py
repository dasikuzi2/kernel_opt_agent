#!/usr/bin/env python3
"""Convert one FlashInfer-Trace definition/workload pair into frozen intake contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def tensor_contract(name: str, spec: dict) -> dict:
    shape = spec.get("shape")
    if shape is None:
        shape = []
    if not isinstance(shape, list):
        raise ValueError(f"tensor {name!r}: shape must be a list or null")
    description = str(spec.get("description", "")).strip()
    optional = bool(spec.get("optional", False))
    qualifiers = ["contiguous in declared axis order"]
    if description:
        qualifiers.append(description)
    if optional:
        qualifiers.append("optional ABI argument")
    return {
        "name": name,
        "dtype": str(spec["dtype"]),
        "shape": shape,
        "layout": "; ".join(qualifiers),
        "lifetime": "caller-owned input" if optional else "call-scoped",
    }


def load_workloads(path: Path, definition_name: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("definition") != definition_name:
            raise ValueError(
                f"{path}:{line_number}: expected definition {definition_name!r}, "
                f"got {record.get('definition')!r}"
            )
        workload = record.get("workload")
        if not isinstance(workload, dict) or not workload.get("uuid"):
            raise ValueError(f"{path}:{line_number}: workload and workload.uuid are required")
        rows.append(workload)
    if not rows:
        raise ValueError(f"{path}: no workload records")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--operator-output", type=Path, required=True)
    parser.add_argument("--workload-output", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--entrypoint", default="run")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--required-matched-ratio", type=float, default=1.0)
    args = parser.parse_args()

    if args.atol <= 0 or args.rtol <= 0:
        raise ValueError("atol and rtol must be positive")
    if not 0 < args.required_matched_ratio <= 1:
        raise ValueError("required-matched-ratio must be in (0, 1]")

    definition = read_object(args.definition)
    name = str(definition.get("name", "")).strip()
    if not name:
        raise ValueError("definition.name is required")
    rows = load_workloads(args.workloads, name)
    source_identity = {
        "format": "FlashInfer-Trace",
        "dataset_revision": args.dataset_revision,
        "definition": {"path": str(args.definition.resolve()), "sha256": digest(args.definition)},
        "workloads": {"path": str(args.workloads.resolve()), "sha256": digest(args.workloads)},
        "record_count": len(rows),
    }
    reference = definition.get("reference")
    reference_sha = hashlib.sha256(str(reference).encode("utf-8")).hexdigest()
    operator = {
        "schema_version": "operator-contract-v1",
        "name": name,
        "computation": {
            "definition": str(definition.get("description") or f"FlashInfer-Trace operator {name}"),
            "dependencies": [
                "The executable mathematical reference is frozen by reference_sha256.",
                *[str(value) for value in definition.get("constraints", [])],
            ],
            "legal_rewrites": [
                "Any decomposition, fusion, tiling, scheduling or specialization preserving the ABI and numerical acceptance rule.",
            ],
            "forbidden_rewrites": [
                "Dropping or changing outputs, state transitions, variable-length boundaries, dtypes or declared layouts.",
                "Special-casing input values or workload UUIDs instead of the declared axes and tensor metadata.",
            ],
        },
        "inputs": [tensor_contract(key, value) for key, value in definition.get("inputs", {}).items()],
        "outputs": [tensor_contract(key, value) for key, value in definition.get("outputs", {}).items()],
        "numerics": {
            "reference": f"Frozen FlashInfer-Trace reference; sha256={reference_sha}",
            "acceptance": (
                f"All outputs: atol={args.atol:g}, rtol={args.rtol:g}, "
                f"matched element ratio >= {args.required_matched_ratio:g}; NaN/Inf mismatch fails."
            ),
            "accumulation_dtype": "as fixed by the mathematical reference; recurrent state is float32",
            "deterministic_required": True,
        },
        "abi": {
            "entrypoint": args.entrypoint,
            "mutable_arguments": [],
            "aliasing": ["inputs and returned outputs must not alias unless the definition explicitly permits it"],
            "layout_constraints": ["tensor shapes and declared axis order are preserved"],
        },
        "source_identity": {**source_identity, "reference_sha256": reference_sha},
    }
    weight = 1.0 / len(rows)
    cases = []
    for row in rows:
        inputs = row.get("inputs", {})
        cases.append(
            {
                "id": str(row["uuid"]),
                "parameters": {
                    **row.get("axes", {}),
                    "input_roles": sorted(inputs),
                    "trace_tensor_source": next(
                        (str(value.get("path")) for value in inputs.values() if isinstance(value, dict) and value.get("path")),
                        None,
                    ),
                },
                "weight": weight,
                "mode": "prefill/direct/variable-length",
                "boundary": False,
            }
        )
    shortest = min(range(len(cases)), key=lambda i: (cases[i]["parameters"].get("total_seq_len", 0), cases[i]["id"]))
    longest = max(range(len(cases)), key=lambda i: (cases[i]["parameters"].get("total_seq_len", 0), cases[i]["parameters"].get("num_seqs", 0), cases[i]["id"]))
    cases[shortest]["boundary"] = True
    cases[longest]["boundary"] = True
    workload = {
        "schema_version": "workload-v1",
        "name": f"{name}-official-equal-weight-trace",
        "cases": cases,
        "objective": {
            "metric": "gpu_active_us",
            "statistic": "mean",
            "direction": "minimize",
            "aggregation": "arithmetic mean of baseline_gpu_active_us / candidate_gpu_active_us over equally weighted cases",
            "correctness_gate": "any failing workload makes the definition score zero",
        },
        "execution": {
            "graph_mode": "off",
            "cache_semantics": "warm compiled solution; one warmup, five timed iterations, three trials",
            "concurrency": "exclusive target GPU; competing-load gate required",
            "upstream_layout": "contiguous tensors in the definition-declared axis order",
            "downstream_layout": "output and new_state layouts fixed by the definition",
            "call_frequency": "each official trace record has equal scoring weight",
        },
        "source_identity": source_identity,
    }
    for output, data in ((args.operator_output, operator), (args.workload_output, workload)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"operator": str(args.operator_output), "workload": str(args.workload_output), "cases": len(cases)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)
