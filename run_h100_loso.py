#!/usr/bin/env python3
"""Run the existing cache-only LOSO entrypoints for all fixed V18 seeds."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PREPARED = ROOT / "prepared"
EMBED_ROOT = ROOT / "frozen_input" / "e1_seed20260804_epoch15"
EMBED_MANIFEST = ROOT / "loso_input_embedding_manifest.parquet"
RESULTS = ROOT / "h100_loso_results"
LOG = RESULTS / "formal_loso.log"
SEEDS = (20260806, 20260807, 20260808)


def build_embedding_manifest() -> None:
    source = pd.read_parquet(EMBED_ROOT / "embedding_manifest.parquet").copy()
    if len(source) != 6600 or source["record_id"].duplicated().any():
        raise RuntimeError(f"Invalid frozen embedding manifest: {len(source)} records")
    source["shard_path"] = [
        str(EMBED_ROOT / "shards" / f"shard_{int(index):04d}.h5")
        for index in source["shard_index"]
    ]
    source.to_parquet(EMBED_MANIFEST, index=False)


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
    manifest = PREPARED / "training_manifest.parquet"
    common = [
        "--manifest", str(manifest),
        "--embedding-manifest", str(EMBED_MANIFEST),
        "--batch-size", "16",
        "--num-workers", "4",
        "--learning-rate", "1e-4",
        "--weight-decay", "1e-4",
        "--max-epochs", "20",
        "--minimum-epochs", "5",
        "--patience", "4",
        "--inner-validation-fraction", "0.15",
        "--device", "cuda",
    ]
    with LOG.open("w", encoding="utf-8") as log:
        for seed in SEEDS:
            seed_root = RESULTS / f"seed_{seed}"
            baseline = seed_root / "baseline"
            candidate = seed_root / "domain_adversarial"
            run_stage(
                [
                    sys.executable,
                    str(ROOT / "train_sequence_ci_loso.py"),
                    *common,
                    "--output-dir", str(baseline),
                    "--models", "raw_sequence",
                    "--seed", str(seed),
                    "--pair-scope", "same_site",
                    "--selection-metric", "age_conditioned_auroc",
                ],
                log,
            )
            run_stage(
                [
                    sys.executable,
                    str(ROOT / "train_domain_adversarial_loso.py"),
                    *common,
                    "--baseline-root", str(baseline),
                    "--output-dir", str(candidate),
                    "--seed", str(seed),
                ],
                log,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
