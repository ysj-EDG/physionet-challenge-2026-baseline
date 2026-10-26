#!/usr/bin/env python3
"""Run the existing fixed-epoch final sequence trainer for all V18 seeds."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "final_input" / "training_manifest.parquet"
EMBED_ROOT = ROOT / "frozen_input" / "e1_seed20260804_epoch15"
EMBED_MANIFEST = ROOT / "final_input" / "embedding_manifest.parquet"
RESULTS = ROOT / "h100_final_results"
LOG = RESULTS / "formal_final_sequence.log"
SEEDS = {20260806: 2, 20260807: 4, 20260808: 1}


def build_embedding_manifest() -> None:
    frame = pd.read_parquet(EMBED_ROOT / "embedding_manifest.parquet").copy()
    if len(frame) != 6600 or frame["record_id"].duplicated().any():
        raise RuntimeError(f"Invalid frozen embedding manifest: {len(frame)} records")
    frame["shard_path"] = [
        str(EMBED_ROOT / "shards" / f"shard_{int(index):04d}.h5")
        for index in frame["shard_index"]
    ]
    frame.to_parquet(EMBED_MANIFEST, index=False)


def run_stage(command: list[str], log) -> None:
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
    status = process.wait()
    if status != 0:
        raise SystemExit(status)


def main() -> int:
    os.environ["PYTHONNOUSERSITE"] = "1"
    RESULTS.mkdir(parents=True, exist_ok=True)
    build_embedding_manifest()
    with LOG.open("w", encoding="utf-8") as log:
        for seed, epochs in SEEDS.items():
            run_stage(
                [
                    sys.executable,
                    str(ROOT / "train_domain_adversarial_final.py"),
                    "--manifest", str(MANIFEST),
                    "--embedding-manifest", str(EMBED_MANIFEST),
                    "--output-dir", str(RESULTS / f"seed_{seed}"),
                    "--batch-size", "16",
                    "--num-workers", "4",
                    "--learning-rate", "1e-4",
                    "--weight-decay", "1e-4",
                    "--fixed-epochs", str(epochs),
                    "--seed", str(seed),
                    "--device", "cuda",
                ],
                log,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
