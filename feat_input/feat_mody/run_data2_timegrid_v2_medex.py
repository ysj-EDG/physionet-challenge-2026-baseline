#!/usr/bin/env python
"""Medex entry point for the restartable full data2 timegrid_v2 cache."""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_CANDIDATES = [ROOT / "data2_source", ROOT / "data2_source" / "data2", Path("/database2/physionet2026_kaggle/data2")]
DATA = next((path for path in DATA_CANDIDATES if (path / "demographics.csv").is_file()), DATA_CANDIDATES[0])
OUT = ROOT / "npz_data2"
LOG = OUT / "medex_full.log"


def available_gib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0


def main():
    if not (DATA / "demographics.csv").is_file():
        raise SystemExit(f"BLOCKED: exact data2 root is unavailable: {DATA}")
    if len(list((DATA / "physiological_data").glob("*/*.edf"))) != 6600:
        raise SystemExit("BLOCKED: data2 physiological EDF count is not 6600")
    mem = available_gib()
    cpus = os.cpu_count() or 1
    # Smoke peak was 3.12 GiB/worker. Reserve substantial headroom and cap I/O
    # concurrency at 16; no BLAS worker may create its own thread pool.
    workers = max(1, min(16, cpus, int(mem // 5) if mem else 8))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    identity = {
        "started_unix": time.time(), "hostname": os.uname().nodename,
        "data_root": str(DATA), "data_edf_count": 6600,
        "output_root": str(OUT), "workers": workers,
        "cpu_count": cpus, "mem_available_gib": mem,
        "blas_threads": 1, "extraction_version": "timegrid_v2.0.0",
        "python": sys.executable,
    }
    (OUT / "medex_launch.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n")
    cmd = [sys.executable, "-u", str(ROOT / "extract_data2_timegrid_v2.py"),
           "--data-root", str(DATA), "--output-root", str(OUT),
           "--workers", str(workers), "--manifest", str(OUT / "manifest.jsonl")]
    with LOG.open("a", buffering=1) as log:
        log.write("COMMAND " + " ".join(cmd) + "\n")
        process = subprocess.Popen(cmd, cwd=ROOT, env=os.environ.copy(),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line)
        code = process.wait()
    identity["finished_unix"] = time.time(); identity["exit_code"] = code
    (OUT / "medex_launch.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n")
    raise SystemExit(code)

if __name__ == "__main__": main()
