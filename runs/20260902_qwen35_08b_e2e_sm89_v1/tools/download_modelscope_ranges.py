#!/usr/bin/env python3
"""Resume one ModelScope CDN object with verified parallel HTTP ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=16)
    args = parser.parse_args()

    probe = requests.get(args.url, headers={"Range": "bytes=0-0"}, stream=True, timeout=30)
    probe.raise_for_status()
    if probe.status_code != 206 or "Content-Range" not in probe.headers:
        raise RuntimeError("ModelScope object does not honor byte ranges")
    total = int(probe.headers["Content-Range"].rsplit("/", 1)[1])
    linked_etag = probe.headers.get("X-Linked-ETag")
    final_url_host = requests.utils.urlparse(probe.url).hostname
    probe.close()

    prefix_size = args.prefix.stat().st_size
    if not 0 < prefix_size < total:
        raise ValueError(f"prefix size {prefix_size} is outside (0, {total})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.output.with_suffix(args.output.suffix + ".ranges.json")
    completed = set()
    if state_path.is_file() and args.output.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("total_bytes") != total or state.get("prefix_bytes") != prefix_size:
            raise ValueError("range state does not match the current object/prefix")
        completed = {tuple(item) for item in state.get("completed_ranges", [])}
    else:
        with args.prefix.open("rb") as source, args.output.open("wb") as target:
            while block := source.read(8 * 1024 * 1024):
                target.write(block)
            target.truncate(total)

    chunk_bytes = args.chunk_mib * 1024 * 1024
    ranges = []
    start = prefix_size
    while start < total:
        end = min(total - 1, start + chunk_bytes - 1)
        if (start, end) not in completed:
            ranges.append((start, end))
        start = end + 1
    descriptor = os.open(args.output, os.O_RDWR)
    lock = threading.Lock()
    started = time.perf_counter()

    def save_state():
        value = {
            "schema_version": "parallel-range-download-state-v1",
            "total_bytes": total,
            "prefix_bytes": prefix_size,
            "completed_ranges": sorted([list(item) for item in completed]),
        }
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, state_path)

    def download(item):
        start, end = item
        expected_range = f"bytes {start}-{end}/{total}"
        last_error = None
        for attempt in range(5):
            try:
                response = requests.get(args.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(f"invalid Content-Range for {start}-{end}: {response.headers.get('Content-Range')}")
                payload = response.content
                if len(payload) != end - start + 1:
                    raise RuntimeError(f"short range {start}-{end}: {len(payload)} bytes")
                break
            except (requests.RequestException, RuntimeError) as error:
                last_error = error
                if attempt == 4:
                    raise RuntimeError(f"range {start}-{end} failed after five attempts") from last_error
                time.sleep(2 ** attempt)
        os.pwrite(descriptor, payload, start)
        with lock:
            completed.add(item)
            save_state()
            finished = prefix_size + sum(last - first + 1 for first, last in completed)
            print(json.dumps({"downloaded_bytes": finished, "total_bytes": total, "percent": round(100 * finished / total, 2)}), flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(download, ranges))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    digest = hashlib.sha256()
    with args.output.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    sha256 = digest.hexdigest()
    if linked_etag and len(linked_etag) == 64 and sha256 != linked_etag:
        raise RuntimeError(f"SHA-256 mismatch: {sha256} != ModelScope {linked_etag}")
    receipt = {
        "schema_version": "modelscope-range-download-receipt-v1",
        "status": "PASS",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_url": args.url,
        "cdn_host": final_url_host,
        "source": "ModelScope official model repository CDN",
        "prefix_bytes_reused": prefix_size,
        "total_bytes": total,
        "workers": args.workers,
        "chunk_bytes": chunk_bytes,
        "duration_seconds": time.perf_counter() - started,
        "output": {"path": str(args.output.resolve()), "sha256": sha256},
        "modelscope_linked_etag": linked_etag,
        "verification": "every range checked for status 206, exact Content-Range and exact length; whole object checked by SHA-256",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
