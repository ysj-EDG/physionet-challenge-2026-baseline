#!/usr/bin/env python
"""Run the fixed seed-7 D/E/F typed-v1 ablations sequentially on H100."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "h100_results" / "seed7"
SOURCE_CHECKPOINT = ROOT / "p1_inputs" / "B_typed_v1" / "lstm_model.pt"
ARMS = (
    ("D_typed_no_age", None, {"x_static_zero_indices": [0]}),
    ("E_typed_no_bmi", None, {"x_static_zero_indices": [9]}),
    ("F_typed_no_coherence", None, {"x_seq_zero_ranges": [[54, 414]]}),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_arm(name: str, clip_z: float | None, mask_config: dict, common_env: dict) -> None:
    arm_dir = OUTPUT / name
    arm_dir.mkdir()
    env = dict(common_env)
    env["LSTM_MODEL_DIR"] = str(arm_dir)
    env["LSTM_INPUT_CLIP_Z"] = "" if clip_z is None else str(clip_z)
    env["LSTM_INPUT_MASK_CONFIG"] = json.dumps(mask_config, separators=(",", ":"))
    command = [sys.executable, "-u", str(ROOT / "train_lstm.py")]
    with (arm_dir / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
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
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{name} training failed with exit code {return_code}")
    checkpoint = arm_dir / "lstm_model.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f"{name} did not produce a non-empty checkpoint")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Medex job")
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name.upper():
        raise RuntimeError(f"Expected H100, got {gpu_name}")
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    for required in (ROOT / "npz_new", ROOT / "split", ROOT / "feat_input" / "feature_rules_v1.json"):
        if not required.exists():
            raise FileNotFoundError(required)
    for name, _, _ in ARMS:
        if (OUTPUT / name).exists():
            raise FileExistsError(f"Refusing to overwrite: {OUTPUT / name}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = {
        "gpu": gpu_name,
        "python": sys.executable,
        "seed": 7,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
        "npz_cache": str((ROOT / "npz_new").resolve()),
        "split": str((ROOT / "split").resolve()),
        "arms": [
            {"name": name, "clip_z": clip_z, "mask_config": mask}
            for name, clip_z, mask in ARMS
        ],
    }
    (OUTPUT / "h100_run_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    common_env = dict(os.environ)
    common_env.update({
        "LSTM_CACHE_DIR": str(ROOT / "npz_new"),
        "LSTM_SPLITS_DIR": str(ROOT / "split"),
        "LSTM_FEATURE_RULES": str(ROOT / "feat_input" / "feature_rules_v1.json"),
        "LSTM_SEED": "7",
        "PYTHONHASHSEED": "7",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LSTM_NPZ_CACHE": str(ROOT / "npz_new"),
        "LSTM_INPUT_PREPROCESSING": "typed_v1",
        "LSTM_PREPROCESSOR_STATE_CHECKPOINT": str(SOURCE_CHECKPOINT),
    })
    for arm in ARMS:
        run_arm(*arm, common_env)


if __name__ == "__main__":
    main()
