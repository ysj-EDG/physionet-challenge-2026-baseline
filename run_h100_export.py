#!/usr/bin/env python3
"""Medex wrapper for the unchanged formal EMA-teacher embedding export."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PREPARED = ROOT / "prepared"
RESULTS = ROOT / "h100_export_results"
MANIFEST = RESULTS / "training_manifest.parquet"
INDEX = RESULTS / "candidate_index.pkl.gz"
CHECKPOINT = ROOT / "export_input" / "checkpoint_epoch_15.pt"
EXPORT_DIR = RESULTS / "frozen" / "e1_seed20260804_epoch15"
LOG = RESULTS / "formal_export.log"


def build_remote_manifest() -> None:
    source = PREPARED / "training_manifest.parquet"
    frame = pd.read_parquet(source).copy()
    record_ids = frame["record_id"].astype(str)
    frame["canonical_path"] = [
        str(PREPARED / "canonical" / "records" / f"{record_id}.h5")
        for record_id in record_ids
    ]
    frame["annotation_path"] = [
        str(PREPARED / "annotations" / "records" / f"{record_id}.h5")
        for record_id in record_ids
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(MANIFEST, index=False)


def main() -> int:
    os.environ["PYTHONNOUSERSITE"] = "1"
    build_remote_manifest()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "export_full_embeddings.py"),
        "--manifest", str(MANIFEST),
        "--annotation-cache-dir", str(PREPARED / "annotations"),
        "--candidate-index", str(INDEX),
        "--checkpoint", str(CHECKPOINT),
        "--state", "teacher",
        "--output-dir", str(EXPORT_DIR),
        "--records-per-shard", "100",
        "--batch-size", "64",
        "--device", "cuda",
    ]
    with LOG.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
