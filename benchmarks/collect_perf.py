#!/usr/bin/env python3
"""Collect benchmark results into commit-indexed history."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import socket
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"
CSV_PATH = BENCH_DIR / "perf_history.csv"
META_PATH = BENCH_DIR / "perf_run_meta.json"


def run(cmd: list[str]) -> str:
    out = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def maybe_run_benchmarks(skip_run: bool) -> None:
    if skip_run:
        return
    run(["pixi", "run", "-e", "bench", "bench_mojo_l1"])
    run(["pixi", "run", "-e", "bench", "bench_mojo_l2"])
    run(["pixi", "run", "-e", "bench", "bench_mojo_l3"])


def op_flops(op: str, n: int) -> float:
    op = op.lower()
    if op in {"axpy", "dot"}:
        return float(2 * n)
    if op in {"nrm2", "sum", "scal", "copy", "swap", "rot", "rotm"}:
        return float(n)
    if op in {"rotg", "rotmg"}:
        return 1.0

    if op in {"gemv", "gemv_trans", "symv"}:
        return float(2 * n * n)
    if op in {"syr", "trmv", "trsv", "spr", "tbmv", "tbsv", "tpmv", "tpsv"}:
        return float(n * n)
    if op in {"syr2", "spr2", "spmv", "sbmv", "gbmv"}:
        return float(2 * n * n)

    if op in {"gemm", "symm"}:
        return float(2 * n * n * n)
    if op in {"syrk", "trmm", "trsm"}:
        return float(n * n * n)
    if op in {"syr2k"}:
        return float(2 * n * n * n)

    return 0.0


def load_results() -> list[dict]:
    rows: list[dict] = []
    for path in (
        BENCH_DIR / "mojo_l1_results.json",
        BENCH_DIR / "mojo_l2_results.json",
        BENCH_DIR / "mojo_l3_results.json",
    ):
        data = json.loads(path.read_text())
        for r in data["results"]:
            n = int(r["n"])
            avg_ns = float(r["avg_ns"])
            flops = op_flops(str(r["op"]), n)
            gflops = (flops / (avg_ns * 1e-9)) / 1e9 if avg_ns > 0 else 0.0
            rows.append(
                {
                    "kernel": str(r["op"]),
                    "size": n,
                    "avg_ns": avg_ns,
                    "gflops": gflops,
                }
            )
    return rows


def baseline_map() -> dict[tuple[str, int], float]:
    if not CSV_PATH.exists():
        return {}
    out: dict[tuple[str, int], float] = {}
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("phase") == "phase0_baseline":
                key = (r["kernel"], int(r["size"]))
                out[key] = float(r["avg_ns"])
    return out


def append_rows(
    rows: list[dict], phase: str, commit_idx: int, sha: str, title: str
) -> None:
    headers = [
        "commit_sha",
        "commit_title",
        "commit_idx",
        "phase",
        "kernel",
        "size",
        "avg_ns",
        "gflops",
        "speedup_vs_baseline",
        "timestamp",
    ]
    write_header = not CSV_PATH.exists()
    baseline = baseline_map()
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        for r in rows:
            key = (r["kernel"], int(r["size"]))
            b = baseline.get(key, r["avg_ns"])
            speedup = (b / r["avg_ns"]) if r["avg_ns"] > 0 else 0.0
            writer.writerow(
                {
                    "commit_sha": sha,
                    "commit_title": title,
                    "commit_idx": commit_idx,
                    "phase": phase,
                    "kernel": r["kernel"],
                    "size": r["size"],
                    "avg_ns": f"{r['avg_ns']:.6f}",
                    "gflops": f"{r['gflops']:.6f}",
                    "speedup_vs_baseline": f"{speedup:.6f}",
                    "timestamp": ts,
                }
            )


def update_meta(
    phase: str, commit_idx: int, sha: str, title: str, skip_run: bool
) -> None:
    payload = {
        "commit_sha": sha,
        "commit_title": title,
        "commit_idx": commit_idx,
        "phase": phase,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "skip_run": skip_run,
    }
    history: list[dict] = []
    if META_PATH.exists():
        history = json.loads(META_PATH.read_text())
    history.append(payload)
    META_PATH.write_text(json.dumps(history, indent=2) + "\n")


def git_info() -> tuple[str, str]:
    sha = run(["git", "rev-parse", "--short", "HEAD"])
    title = run(["git", "show", "-s", "--format=%s", "HEAD"])
    return sha, title


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True)
    p.add_argument("--commit-idx", type=int, required=True)
    p.add_argument("--skip-run", action="store_true")
    args = p.parse_args()

    maybe_run_benchmarks(args.skip_run)
    sha, title = git_info()
    rows = load_results()
    append_rows(rows, args.phase, args.commit_idx, sha, title)
    update_meta(args.phase, args.commit_idx, sha, title, args.skip_run)
    print(
        f"appended {len(rows)} rows for {sha} phase={args.phase} idx={args.commit_idx}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

