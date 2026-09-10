#!/usr/bin/env python
"""Run unchanged official inference/scoring entry points for P3 pooled models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
OUTPUT = ROOT / "output" / "p3_v2" / "final"
CACHE = ROOT / "npz_new"
SIDECAR = ROOT / "output" / "p3_v2" / "stage_sidecar"
PREVALENCE = Path(
    "/database/home/gaohaojie/workspace/challenge2026/output/input/prevalence.csv"
)
ARMS = (
    "P3_demo10",
    "P3_compact30",
    "P3_compact35",
    "P3_global59",
    "P3_stage155",
)
SPLITS = {
    "train": (
        ROOT / "output/p2_model_controls/seed7/eval_inputs/train",
        ROOT / "output/p2_model_controls/seed7/eval_inputs/train/demographics.csv",
        733,
    ),
    "val": (
        ROOT / "output/p2_model_controls/seed7/eval_inputs/val",
        ROOT / "output/p2_model_controls/seed7/eval_inputs/val/demographics.csv",
        158,
    ),
    "test": (
        Path("/database/home/gaohaojie/workspace/challenge2026/output/input/test"),
        Path("/database/home/gaohaojie/workspace/challenge2026/output/input/test/labels.csv"),
        158,
    ),
    "external": (
        Path("/database/home/gaohaojie/workspace/challenge2026/output/input/external"),
        Path("/database/home/gaohaojie/workspace/challenge2026/output/input/external/labels.csv"),
        54,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def cache_path_for(row: pd.Series) -> Path:
    session = row["SessionID"]
    if isinstance(session, (float, np.floating)) and float(session).is_integer():
        session = int(session)
    name = f"{row['BidsFolder']}_ses-{session}.npz"
    candidates = [CACHE / name] + [
        CACHE / split / name for split in ("train", "val", "test", "external")
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def validate_output(source_csv: Path, output_csv: Path, sidecar: Path, expected: int) -> None:
    source = pd.read_csv(source_csv)
    prediction = pd.read_csv(output_csv)
    decision = pd.read_csv(sidecar)
    if not (len(source) == len(prediction) == len(decision) == expected):
        raise RuntimeError(
            f"Coverage mismatch: source={len(source)} prediction={len(prediction)} "
            f"decision={len(decision)} expected={expected}"
        )
    keys = ["BidsFolder", "SessionID"]
    if not source[keys].astype(str).reset_index(drop=True).equals(
        prediction[keys].astype(str).reset_index(drop=True)
    ):
        raise RuntimeError("Official input and output identity/order differ")
    if not source[keys].astype(str).reset_index(drop=True).equals(
        decision[keys].astype(str).reset_index(drop=True)
    ):
        raise RuntimeError("Decision sidecar identity/order differs from official input")
    official = pd.to_numeric(
        prediction["Cognitive_Impairment_Probability"], errors="raise"
    ).to_numpy(float)
    saved = pd.to_numeric(
        decision["calibrated_probability"], errors="raise"
    ).to_numpy(float)
    logits = pd.to_numeric(decision["decision_logit"], errors="raise").to_numpy(float)
    if not np.isfinite(official).all() or not np.isfinite(logits).all():
        raise RuntimeError("Non-finite official probability or decision logit")
    if not np.allclose(official, saved, rtol=0, atol=1e-15):
        raise RuntimeError("Decision sidecar and official probabilities differ")


def main() -> None:
    if len(list(CACHE.rglob("*.npz"))) != 1103:
        raise RuntimeError(f"Frozen cache count differs from 1103: {CACHE}")
    if len(list(SIDECAR.glob("*.npz"))) != 1103:
        raise RuntimeError(f"P3 sidecar count differs from 1103: {SIDECAR}")
    for required in (PREVALENCE, ROOT / "run_model.py", ROOT / "evaluate_model.py"):
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest = {
        "python": str(PYTHON),
        "cache": str(CACHE),
        "sidecar": str(SIDECAR),
        "prevalence": str(PREVALENCE),
        "prevalence_sha256": sha256(PREVALENCE),
        "run_model_sha256": sha256(ROOT / "run_model.py"),
        "evaluate_model_sha256": sha256(ROOT / "evaluate_model.py"),
        "arms": {},
        "splits": {},
    }
    for split_name, (data_folder, labels, expected) in SPLITS.items():
        demographics = data_folder / "demographics.csv"
        for path in (demographics, labels):
            if not path.is_file():
                raise FileNotFoundError(path)
        frame = pd.read_csv(demographics)
        if len(frame) != expected:
            raise RuntimeError(f"{split_name}: {len(frame)} records, expected {expected}")
        missing_cache = [
            str(cache_path_for(row))
            for _, row in frame.iterrows()
            if not cache_path_for(row).is_file()
        ]
        missing_sidecar = []
        for _, row in frame.iterrows():
            cache = cache_path_for(row)
            target = SIDECAR / cache.name
            if not target.is_file():
                missing_sidecar.append(str(target))
        if missing_cache or missing_sidecar:
            raise RuntimeError(
                f"{split_name}: missing cache={len(missing_cache)}, "
                f"sidecar={len(missing_sidecar)}"
            )
        manifest["splits"][split_name] = {
            "data_folder": str(data_folder),
            "demographics_sha256": sha256(demographics),
            "labels": str(labels),
            "labels_sha256": sha256(labels),
            "expected_records": expected,
        }

    common_env = dict(os.environ)
    common_env["LSTM_NPZ_CACHE"] = str(CACHE)
    common_env["P3_STAGE_SIDECAR"] = str(SIDECAR)
    for arm in ARMS:
        model_folder = OUTPUT / arm
        checkpoint = model_folder / "lstm_model.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(checkpoint)
        manifest["arms"][arm] = {"checkpoint_sha256": sha256(checkpoint)}
        for split_name, (data_folder, labels, expected) in SPLITS.items():
            split_output = model_folder / f"official_{split_name}"
            if split_output.exists():
                raise RuntimeError(f"Refusing to overwrite: {split_output}")
            decision = split_output / "decision_outputs.csv"
            env = dict(common_env)
            env["LSTM_DECISION_OUTPUT"] = str(decision)
            run(
                [str(PYTHON), "-u", str(ROOT / "run_model.py"), "-d", str(data_folder),
                 "-m", str(model_folder), "-o", str(split_output)], env
            )
            output_csv = split_output / "demographics.csv"
            validate_output(data_folder / "demographics.csv", output_csv, decision, expected)
            run(
                [str(PYTHON), "-u", str(ROOT / "evaluate_model.py"), "-d", str(labels),
                 "-o", str(output_csv), "-p", str(PREVALENCE),
                 "-s", str(split_output / "scores.csv"),
                 "-t", str(split_output / "table.csv")], common_env
            )
    (OUTPUT.parent / "official_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
