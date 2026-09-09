#!/usr/bin/env python
"""Run the unchanged official inference and scoring entry points for P2."""

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
OUTPUT = ROOT / "output" / "p2_model_controls" / "seed7"
CACHE = ROOT / "npz_new"
PREVALENCE = Path(
    "/database/home/gaohaojie/workspace/challenge2026/output/input/prevalence.csv"
)
ARMS = (
    "G_B_sameenv",
    "H_single_balance",
    "I_LR_demo10",
    "J_LR_compact30",
    "K_LR_static196",
)
SPLITS = {
    "train": (
        OUTPUT / "eval_inputs" / "train",
        OUTPUT / "eval_inputs" / "train" / "demographics.csv",
        733,
    ),
    "val": (
        OUTPUT / "eval_inputs" / "val",
        OUTPUT / "eval_inputs" / "val" / "demographics.csv",
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


def validate_output(input_csv: Path, output_csv: Path, sidecar: Path, expected: int) -> None:
    source = pd.read_csv(input_csv)
    prediction = pd.read_csv(output_csv)
    decision = pd.read_csv(sidecar)
    if not (len(source) == len(prediction) == len(decision) == expected):
        raise RuntimeError(
            f"Coverage mismatch: source={len(source)} prediction={len(prediction)} "
            f"decision={len(decision)} expected={expected}"
        )
    source_keys = source[["BidsFolder", "SessionID"]].astype(str)
    prediction_keys = prediction[["BidsFolder", "SessionID"]].astype(str)
    decision_ids = decision[["BidsFolder", "SessionID"]].astype(str)
    if not source_keys.reset_index(drop=True).equals(prediction_keys.reset_index(drop=True)):
        raise RuntimeError("Official input and output identity/order differ")
    if not source_keys.reset_index(drop=True).equals(decision_ids.reset_index(drop=True)):
        raise RuntimeError("Decision sidecar identity/order differs from official input")
    official_probability = pd.to_numeric(
        prediction["Cognitive_Impairment_Probability"], errors="raise"
    ).to_numpy(float)
    sidecar_probability = pd.to_numeric(
        decision["calibrated_probability"], errors="raise"
    ).to_numpy(float)
    logits = pd.to_numeric(decision["decision_logit"], errors="raise").to_numpy(float)
    if not np.isfinite(official_probability).all() or not np.isfinite(logits).all():
        raise RuntimeError("Non-finite official probability or decision logit")
    if not np.allclose(official_probability, sidecar_probability, rtol=0, atol=1e-15):
        raise RuntimeError("Sidecar calibrated probabilities differ from official output")


def cache_path_for(row: pd.Series) -> Path:
    session = row["SessionID"]
    if isinstance(session, (float, np.floating)) and float(session).is_integer():
        session = int(session)
    name = f"{row['BidsFolder']}_ses-{session}.npz"
    candidates = [CACHE / name] + [
        CACHE / split_name / name for split_name in ("train", "val", "test", "external")
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def main() -> None:
    if not CACHE.is_dir() or len(list(CACHE.rglob("*.npz"))) != 1103:
        raise RuntimeError(f"Frozen cache is missing or has unexpected count: {CACHE}")
    if not PREVALENCE.is_file():
        raise FileNotFoundError(PREVALENCE)
    for arm in ARMS:
        checkpoint = OUTPUT / arm / "lstm_model.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(checkpoint)
        for split_name in SPLITS:
            split_output = OUTPUT / arm / split_name
            if split_output.exists():
                raise RuntimeError(f"Refusing to overwrite: {split_output}")

    manifest = {
        "python": str(PYTHON),
        "cache": str(CACHE),
        "cache_count": 1103,
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
            raise RuntimeError(
                f"{split_name}: input count {len(frame)} != expected {expected}"
            )
        missing_cache = [
            str(cache_path_for(row))
            for _, row in frame.iterrows()
            if not cache_path_for(row).is_file()
        ]
        if missing_cache:
            raise RuntimeError(
                f"{split_name}: {len(missing_cache)} frozen NPZ files missing; "
                f"first={missing_cache[0]}"
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
    for arm in ARMS:
        model_folder = OUTPUT / arm
        manifest["arms"][arm] = {
            "checkpoint_sha256": sha256(model_folder / "lstm_model.pt")
        }
        for split_name, (data_folder, labels, expected) in SPLITS.items():
            split_output = model_folder / split_name
            sidecar = split_output / "decision_outputs.csv"
            env = dict(common_env)
            env["LSTM_DECISION_OUTPUT"] = str(sidecar)
            run(
                [
                    str(PYTHON), "-u", str(ROOT / "run_model.py"),
                    "-d", str(data_folder), "-m", str(model_folder),
                    "-o", str(split_output),
                ],
                env,
            )
            output_csv = split_output / "demographics.csv"
            validate_output(
                data_folder / "demographics.csv", output_csv, sidecar, expected
            )
            run(
                [
                    str(PYTHON), "-u", str(ROOT / "evaluate_model.py"),
                    "-d", str(labels), "-o", str(output_csv),
                    "-p", str(PREVALENCE),
                    "-s", str(split_output / "scores.csv"),
                    "-t", str(split_output / "table.csv"),
                ],
                common_env,
            )
    (OUTPUT / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
