#!/usr/bin/env python3
"""Benchmark a persistent llama.cpp server on the frozen natural request suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


NATURAL_REQUESTS = (
    (
        "zh-explanation",
        "请用通俗但准确的语言解释：为什么大语言模型逐 token 解码通常受显存带宽限制？给出一个简单的数量级估算。",
    ),
    (
        "python-code",
        "Write a complete Python function that merges overlapping half-open intervals. Include type hints, a concise explanation, and three edge-case tests.",
    ),
    (
        "reasoning",
        "A shop discounts an item by 20%, then raises the discounted price by 25%. Explain step by step whether the final price equals the original price.",
    ),
    (
        "editing",
        "Rewrite the following paragraph to be concise and professional without losing any facts: Our team ran several experiments over the last two weeks, but because each trial used a different environment and no frozen baseline, the reported improvements cannot yet be compared reliably.",
    ),
    (
        "systems-design",
        "Design a bounded GPU-kernel optimization loop for an autonomous agent. Focus on candidate diversity, fail-fast checks, correctness gates, measurement noise, and stopping rules.",
    ),
    (
        "translation",
        "Translate into natural Chinese and briefly explain the technical meaning: Speculative decoding reduces latency only when accepted draft tokens amortize the verification cost.",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def post_json(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_until_ready(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.25)
    raise TimeoutError(f"llama-server did not become ready: {last_error}")


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def sample_gpu(stop: threading.Event, destination: list[dict]) -> None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return
    query = (
        "power.draw,clocks.current.graphics,clocks.current.memory,"
        "temperature.gpu,utilization.gpu"
    )
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    while not stop.is_set():
        completed = subprocess.run(
            [
                executable,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            creationflags=creation_flags,
        )
        fields = [item.strip() for item in completed.stdout.strip().split(",")]
        if completed.returncode == 0 and len(fields) == 5:
            try:
                destination.append(
                    {
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                        "power_w": float(fields[0]),
                        "graphics_clock_mhz": float(fields[1]),
                        "memory_clock_mhz": float(fields[2]),
                        "temperature_c": float(fields[3]),
                        "gpu_utilization_percent": float(fields[4]),
                    }
                )
            except ValueError:
                pass
        stop.wait(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument(
        "--spec-type",
        choices=("none", "draft-mtp"),
        default="none",
        help="Optional llama.cpp speculative decoding method.",
    )
    parser.add_argument("--spec-draft-n-max", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.warmups < 1 or args.trials < 1 or args.new_tokens < 1:
        raise ValueError("warmups, trials, and new-tokens must be positive")

    server_path = args.server.resolve()
    model_path = args.model.resolve()
    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        str(server_path),
        "-m", str(model_path),
        "-ngl", "all",
        "-fa", "on",
        "-c", "4096",
        "-np", "1",
        "-t", str(args.threads),
        "-tb", str(args.threads),
        "--reasoning", "off",
        "--no-cache-prompt",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--metrics",
    ]
    if args.spec_type != "none":
        command.extend(
            [
                "--spec-type", args.spec_type,
                "--spec-draft-n-max", str(args.spec_draft_n_max),
            ]
        )
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags,
    )
    samples: list[dict] = []
    gpu_samples: list[dict] = []
    gpu_stop = threading.Event()
    gpu_thread: threading.Thread | None = None
    server_log = ""
    started = time.perf_counter()
    try:
        wait_until_ready(base_url, process, args.startup_timeout)
        init_seconds = time.perf_counter() - started
        gpu_thread = threading.Thread(
            target=sample_gpu, args=(gpu_stop, gpu_samples), daemon=True
        )
        gpu_thread.start()

        def request(case_id: str, prompt: str, phase: str, iteration: int) -> dict:
            body = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": args.new_tokens,
                "seed": 20260902,
                "stream": False,
                "ignore_eos": True,
                "cache_prompt": False,
            }
            request_started = time.perf_counter()
            response = post_json(
                f"{base_url}/v1/chat/completions", body, timeout=120.0
            )
            wall_ms = (time.perf_counter() - request_started) * 1000.0
            choice = response["choices"][0]
            content = choice["message"].get("content") or ""
            reasoning = choice["message"].get("reasoning_content") or ""
            timings = response.get("timings") or {}
            usage = response.get("usage") or {}
            generated_tokens = int(
                usage.get("completion_tokens", timings.get("predicted_n", 0))
            )
            return {
                "case_id": case_id,
                "phase": phase,
                "iteration": iteration,
                "generated_tokens": generated_tokens,
                "end_to_end_ms": wall_ms,
                "prompt_ms": float(timings.get("prompt_ms", 0.0)),
                "decode_ms": float(timings.get("predicted_ms", 0.0)),
                "output_tokens_per_second": float(
                    timings.get("predicted_per_second", 0.0)
                ),
                "content_sha256": hashlib.sha256(
                    (reasoning + "\n" + content).encode("utf-8")
                ).hexdigest(),
                "content": content,
                "reasoning_content": reasoning,
            }

        case_ids = [case_id for case_id, _ in NATURAL_REQUESTS]
        requests = dict(NATURAL_REQUESTS)
        for iteration in range(args.warmups):
            shift = iteration % len(case_ids)
            order = case_ids[shift:] + case_ids[:shift]
            for case_id in order:
                samples.append(request(case_id, requests[case_id], "warmup", iteration))
        for iteration in range(args.trials):
            shift = (iteration + args.warmups) % len(case_ids)
            order = case_ids[shift:] + case_ids[:shift]
            for case_id in order:
                samples.append(request(case_id, requests[case_id], "measure", iteration))

        summaries = []
        for case_id, prompt in NATURAL_REQUESTS:
            measured = [
                sample
                for sample in samples
                if sample["case_id"] == case_id and sample["phase"] == "measure"
            ]
            content_hash = measured[0]["content_sha256"]
            repeatable = all(sample["content_sha256"] == content_hash for sample in measured)
            full_length = all(
                sample["generated_tokens"] == args.new_tokens for sample in measured
            )
            summaries.append(
                {
                    "case_id": case_id,
                    "weight": 1.0 / len(NATURAL_REQUESTS),
                    "prompt_utf8_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "generated_tokens": args.new_tokens,
                    "correctness": "PASS" if repeatable and full_length else "FAIL",
                    "content_sha256": content_hash,
                    "sample_content": measured[0]["content"],
                    "median_end_to_end_ms": median(
                        [sample["end_to_end_ms"] for sample in measured]
                    ),
                    "median_prompt_ms": median(
                        [sample["prompt_ms"] for sample in measured]
                    ),
                    "median_decode_ms": median(
                        [sample["decode_ms"] for sample in measured]
                    ),
                    "median_output_tokens_per_second": median(
                        [sample["output_tokens_per_second"] for sample in measured]
                    ),
                }
            )
        weighted_e2e_ms = sum(
            case["weight"] * case["median_end_to_end_ms"] for case in summaries
        )
        weighted_decode_tps = sum(
            case["weight"] * case["median_output_tokens_per_second"]
            for case in summaries
        )
        version = subprocess.run(
            [str(server_path), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        payload = {
            "schema_version": "qwen35-llamacpp-natural-discovery-v1",
            "status": (
                "PASS"
                if all(case["correctness"] == "PASS" for case in summaries)
                else "FAIL"
            ),
            "claim_scope": "EXPLORATORY_QUANTIZED_DISCOVERY_REQUIRES_QUALITY_EVALUATION",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "binary": {
                "path": str(server_path),
                "sha256": sha256(server_path),
                "version": (version.stdout + version.stderr).strip(),
            },
            "model": {
                "path": str(model_path),
                "sha256": sha256(model_path),
                "size_bytes": model_path.stat().st_size,
            },
            "controls": {
                "gpu_layers": "all",
                "flash_attention": True,
                "context_size": 4096,
                "parallel_slots": 1,
                "threads": args.threads,
                "reasoning": False,
                "prompt_cache": False,
                "spec_type": args.spec_type,
                "spec_draft_n_max": (
                    args.spec_draft_n_max if args.spec_type != "none" else 0
                ),
                "warmups": args.warmups,
                "trials": args.trials,
                "case_order": "rotated per iteration",
                "sampling": (
                    f"greedy temperature=0, ignore_eos=True, exact {args.new_tokens} tokens"
                ),
                "server_initialization_seconds": init_seconds,
                "command": command,
            },
            "aggregate": {
                "weighted_median_end_to_end_ms": weighted_e2e_ms,
                "weighted_median_output_tokens_per_second": weighted_decode_tps,
            },
            "gpu_telemetry": gpu_samples,
            "cases": summaries,
            "raw_samples": samples,
        }
    finally:
        gpu_stop.set()
        if gpu_thread is not None:
            gpu_thread.join(timeout=3)
        process.terminate()
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=10)
        server_log = output or ""

    payload["server_log_tail"] = server_log[-12000:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "aggregate": payload["aggregate"], "cases": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
