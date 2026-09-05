#!/usr/bin/env python3
"""Record a bounded llama.cpp perplexity screen for quantization candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_model(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("model must use LABEL=PATH") from error
    if not label:
        raise argparse.ArgumentTypeError("model label cannot be empty")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perplexity", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, type=parse_model)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=8)
    args = parser.parse_args()

    executable = args.perplexity.resolve()
    corpus = args.corpus.resolve()
    results = []
    for label, raw_model in args.model:
        model = raw_model.resolve()
        command = [
            str(executable),
            "-m", str(model),
            "-f", str(corpus),
            "-ngl", "all",
            "-fa", "on",
            "-c", str(args.context_size),
            "--chunks", str(args.chunks),
            "--ppl-output-type", "1",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        raw_output = completed.stdout + completed.stderr
        match = re.search(
            r"Final estimate:\s+PPL\s+=\s+([0-9.]+)\s+\+/-\s+([0-9.]+)",
            raw_output,
        )
        if completed.returncode != 0 or match is None:
            raise RuntimeError(
                f"perplexity failed for {label}, code={completed.returncode}: "
                f"{raw_output[-2000:]}"
            )
        results.append(
            {
                "label": label,
                "model_path": str(model),
                "model_sha256": sha256(model),
                "model_size_bytes": model.stat().st_size,
                "perplexity": float(match.group(1)),
                "reported_error": float(match.group(2)),
                "command": command,
            }
        )

    baseline = next((item for item in results if item["label"] == "bf16"), None)
    if baseline is None:
        raise ValueError("one model must be labeled bf16")
    for item in results:
        item["relative_perplexity_vs_bf16"] = (
            item["perplexity"] / baseline["perplexity"]
        )

    payload = {
        "schema_version": "llamacpp-quantization-quality-screen-v1",
        "status": "PASS",
        "claim_scope": "DISCOVERY_QUALITY_SCREEN_NOT_TASK_QUALIFICATION",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "binary": {
            "path": str(executable),
            "sha256": sha256(executable),
        },
        "corpus": {
            "path": str(corpus),
            "sha256": sha256(corpus),
            "size_bytes": corpus.stat().st_size,
        },
        "controls": {
            "context_size": args.context_size,
            "chunks": args.chunks,
            "gpu_layers": "all",
            "flash_attention": True,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
