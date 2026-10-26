#!/usr/bin/env python3
"""Medex wrapper: remap the prepared-cache manifest and run the unchanged E1 script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PREPARED = ROOT / "prepared"
RESULTS = ROOT / "h100_e1_results"
MANIFEST = RESULTS / "training_manifest.parquet"
INDEX = RESULTS / "candidate_index.pkl.gz"
E1_DIR = RESULTS / "e1_seed20260804_epoch15"
LOG = RESULTS / "formal_e1.log"


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
    (RESULTS / "path_remap.json").write_text(
        json.dumps(
            {
                "source_manifest": str(source),
                "remote_prepared_root": str(PREPARED),
                "records": int(len(frame)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    os.environ["PYTHONNOUSERSITE"] = "1"
    build_remote_manifest()
    command = [
        sys.executable,
        str(ROOT / "train_domain_robust_encoder_full.py"),
        "--manifest",
        str(MANIFEST),
        "--annotation-cache-dir",
        str(PREPARED / "annotations"),
        "--candidate-index",
        str(INDEX),
        "--output-dir",
        str(E1_DIR),
        "--variant",
        "e1",
        "--seed",
        "20260804",
        "--epochs",
        "15",
        "--windows-per-record",
        "128",
        "--natural-windows",
        "64",
        "--eval-windows-per-record",
        "64",
        "--batch-size",
        "16",
        "--records-per-batch",
        "1",
        "--num-workers",
        "4",
        "--learning-rate",
        "6e-4",
        "--weight-decay",
        "1e-4",
        "--ema-decay",
        "0.996",
        "--device",
        "cuda",
        "--overwrite",
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
