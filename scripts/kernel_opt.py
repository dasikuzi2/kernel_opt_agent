#!/usr/bin/env python3
"""Single public command surface for the evidence-closed optimization workflow."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True


@dataclass(frozen=True)
class Command:
    script: str
    summary: str


COMMAND_GROUPS: dict[str, dict[str, Command]] = {
    "run lifecycle": {
        "new-run": Command("new_run.py", "freeze intake and create a run"),
        "trace-intake": Command("import_flashinfer_trace.py", "convert FlashInfer Trace into frozen intake contracts"),
        "next": Command("optimizer_step.py", "select the next evidence-driven action"),
        "advance": Command("advance_run.py", "validate and advance one phase gate"),
        "audit": Command("audit_repository.py", "verify reusable-zone purity"),
    },
    "candidate discovery": {
        "opportunity": Command("opportunity_map.py", "validate and rank global gain opportunities before implementation"),
        "method": Command("method_library.py", "match transferable optimization methods to ranked opportunities"),
        "candidate": Command("candidate_discovery.py", "manage fast production-candidate discovery and repair"),
    },
    "hardware": {
        "hardware-discover": Command("discover_hardware.py", "query the target device and software stack"),
        "hardware-init": Command("init_hardware_evidence.py", "initialize the official-evidence manifest"),
        "hardware-add-source": Command("add_official_hardware_source.py", "archive one official source receipt"),
        "hardware-add-fact": Command("add_documented_hardware_fact.py", "bind one fact to an official locator"),
        "hardware-validate": Command("validate_hardware_evidence.py", "validate target identity and official evidence"),
        "measurement-register": Command("register_measurement.py", "register an immutable hardware measurement"),
        "ncu-probe": Command("probe_ncu_access.py", "record whether required NCU counters are accessible"),
    },
    "final binary": {
        "sass-archive": Command("archive_final_binary_sass.py", "hash-bind binary, tools and disassembly"),
        "sass-count": Command("count_sass.py", "classify all final-binary instruction sites"),
        "resources-discover": Command("discover_resources.py", "derive the material resource set"),
    },
    "measurement": {
        "p0-calibrate": Command("calibrate_p0.py", "qualify timing and launch semantics"),
        "service-curve-fit": Command("fit_service_curve.py", "fit latency and throughput service curves"),
        "service-policy": Command("derive_serving_policy.py", "derive a guarded batch-aware serving policy"),
        "paired-compare": Command("compare_paired.py", "compare interleaved baseline/candidate samples"),
    },
    "experiment": {
        "experiment-rank": Command("rank_experiments.py", "rank requests by bounded weighted value"),
        "experiment-materialize": Command("materialize_experiment.py", "seal source, commands and artifacts"),
        "experiment-approve": Command("approve_experiment.py", "independently approve a candidate-driven experiment"),
        "experiment-dispatch": Command("dispatch_experiment.py", "dispatch only a supervisor-approved experiment"),
        "experiment-withdraw": Command("withdraw_experiment.py", "withdraw and require fresh supervisor review"),
        "experiment-revise": Command("revise_experiment.py", "archive an attempt and force review or replanning"),
        "experiment-execute": Command("execute_experiment.py", "execute and hash-bind fresh outputs"),
        "experiment-bind": Command("bind_experiment_result.py", "bind a result to its execution receipt"),
        "experiment-apply": Command("apply_model_updates.py", "apply recomputable field-level transforms"),
        "experiment-reconcile": Command("reconcile_experiment_result.py", "close all affected global models"),
    },
    "microbenchmark": {
        "microbench-query": Command("query_microbench_catalog.py", "find a reusable atomic probe"),
        "microbench-new": Command("new_microbench_candidate.py", "create a run-local probe candidate"),
        "microbench-reproduce": Command("execute_microbench_reproduction.py", "run one cold reproduction contract"),
        "microbench-promote": Command("promote_microbench.py", "promote a qualified generic probe"),
        "microbench-harvest": Command("harvest_microbenches.py", "process all run-local candidates"),
    },
    "certification": {
        "certify": Command("emit_certificate.py", "recompute and emit the limit certificate"),
        "report-validate": Command("validate_human_review_report.py", "validate review-report semantics"),
        "report-render": Command("render_human_review_report.py", "render the validated Chinese HTML report"),
    },
}
COMMANDS = {name: command for group in COMMAND_GROUPS.values() for name, command in group.items()}


def epilog() -> str:
    lines = ["commands:"]
    for group, commands in COMMAND_GROUPS.items():
        lines.append(f"  {group}:")
        lines.extend(f"    {name:<24} {command.summary}" for name, command in commands.items())
    lines.extend(("", "arguments after COMMAND are forwarded unchanged; use COMMAND --help for details"))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=epilog(), formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, forwarded = parser.parse_known_args()
    target = Path(__file__).resolve().with_name(COMMANDS[args.command].script)
    if not target.is_file():
        parser.error(f"command implementation is missing: {target}")
    # The public command must preserve reusable-zone purity.  Child command
    # modules import shared helpers from scripts/; disable bytecode generation
    # before exec so normal framework use never creates scripts/__pycache__.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.call([sys.executable, str(target), *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
