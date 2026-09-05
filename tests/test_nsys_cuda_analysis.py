#!/usr/bin/env python3
"""Dependency-free checks for CUDA timeline opportunity and reachability analysis."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_nsys_cuda import analyze


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "trace.sqlite"
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
            insert into StringIds values(1, '_candidate_kernel');
            insert into StringIds values(2, 'internal::gemvx::kernel');
            insert into StringIds values(3, 'cutlass::flash_fwd_splitkv_kernel');
            insert into CUPTI_ACTIVITY_KIND_KERNEL values
                (0, 1000, 1, 4, 1, 1, 32, 1, 1),
                (1000, 3000, 1, 4, 1, 1, 32, 1, 1),
                (3000, 6000, 2, 8, 1, 1, 64, 1, 1),
                (6000, 10000, 3, 1, 1, 1, 128, 1, 1);
            """
        )
        connection.commit()
        connection.close()

        result = analyze(path, 1, [("_candidate_kernel", 2)], top=10)
        assert result["status"] == "PASS"
        assert result["kernel_instances"] == 4
        cublas_family = next(
            item for item in result["families"] if item["family"] == "cublas_gemvx"
        )
        assert cublas_family["instances"] == 1
        assert any(item["family"] == "flash_attention" for item in result["families"])
        assert result["execution_proofs"][0]["observed_count"] == 2
        assert result["execution_proofs"][0]["evidence_index"] == 0

        failed = analyze(path, 1, [("_candidate_kernel", 3)], top=10)
        assert failed["status"] == "FAIL"

    print("nsys CUDA analysis test: PASS")


if __name__ == "__main__":
    main()
