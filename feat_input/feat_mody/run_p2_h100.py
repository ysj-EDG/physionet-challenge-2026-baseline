#!/usr/bin/env python
"""Run fixed P2 experiments G-K sequentially in one H100 Medex job."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import sklearn
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_lstm


OUTPUT = ROOT / "p2_h100_results" / "seed7"
SOURCE = ROOT / "p2_inputs" / "B_typed_v1" / "lstm_model.pt"
ARMS = [
    "G_B_sameenv",
    "H_single_balance",
    "I_LR_demo10",
    "J_LR_compact30",
    "K_LR_static196",
]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(command: list[str], env: dict, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
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
        raise RuntimeError(f"Command failed ({return_code}): {' '.join(command)}")


def run_lstm(name: str, mode: str, common_env: dict) -> None:
    arm_dir = OUTPUT / name
    arm_dir.mkdir()
    env = dict(common_env)
    env["LSTM_MODEL_DIR"] = str(arm_dir)
    env["LSTM_POS_WEIGHT_MODE"] = mode
    run(
        [sys.executable, "-u", str(ROOT / "train_lstm.py")],
        env,
        arm_dir / "train.log",
    )
    path = arm_dir / "lstm_model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("pos_weight_mode") != mode:
        raise RuntimeError(f"{name}: incorrect saved pos_weight_mode")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    gpu = torch.cuda.get_device_name(0)
    if "H100" not in gpu.upper():
        raise RuntimeError(f"Expected H100, got {gpu}")
    for path in (
        SOURCE,
        ROOT / "npz_new",
        ROOT / "split",
        ROOT / "feat_input" / "feature_rules_v1.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for arm in ARMS:
        if (OUTPUT / arm).exists():
            raise FileExistsError(f"Refusing to overwrite: {OUTPUT / arm}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    train_lstm.seed_everything(7)
    model_g = train_lstm.LSTMModel()
    train_lstm.seed_everything(7)
    model_h = train_lstm.LSTMModel()
    hash_g, hash_h = state_hash(model_g), state_hash(model_h)
    if hash_g != hash_h:
        raise RuntimeError("G/H initial model hashes differ")
    logits = torch.tensor([0.5, -0.25, 1.0, -1.5])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    weights = {
        "empirical": train_lstm.resolve_pos_weight("empirical", 53, 680),
        "unit": train_lstm.resolve_pos_weight("unit", 53, 680),
    }
    losses = {
        name: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))(
            logits, labels
        )
        for name, weight in weights.items()
    }
    if not all(torch.isfinite(value) for value in losses.values()):
        raise RuntimeError("Non-finite loss precheck")

    identity = {
        "base_commit": "e048052f033a62e95ee911fab151faa773cfffde",
        "gpu": gpu,
        "python": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "seed": 7,
        "source_checkpoint_sha256": file_hash(SOURCE),
        "rules_sha256": file_hash(ROOT / "feat_input" / "feature_rules_v1.json"),
        "split_sha256": {
            name: file_hash(ROOT / "split" / f"{name}_records.json")
            for name in ("train", "val", "test", "external")
        },
        "source_files_sha256": {
            name: file_hash(ROOT / name)
            for name in (
                "train_lstm.py",
                "train_static_lr.py",
                "team_code.py",
                "static_logistic.py",
            )
        },
        "g_initial_model_hash": hash_g,
        "h_initial_model_hash": hash_h,
        "pos_weights": weights,
        "losses": {name: float(value) for name, value in losses.items()},
    }
    (OUTPUT / "h100_run_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.update(
        {
            "LSTM_CACHE_DIR": str(ROOT / "npz_new"),
            "LSTM_SPLITS_DIR": str(ROOT / "split"),
            "LSTM_FEATURE_RULES": str(ROOT / "feat_input" / "feature_rules_v1.json"),
            "LSTM_PREPROCESSOR_STATE_CHECKPOINT": str(SOURCE),
            "LSTM_INPUT_PREPROCESSING": "typed_v1",
            "LSTM_INPUT_CLIP_Z": "",
            "LSTM_INPUT_MASK_CONFIG": "{}",
            "LSTM_SEED": "7",
            "PYTHONHASHSEED": "7",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "LSTM_NPZ_CACHE": str(ROOT / "npz_new"),
        }
    )
    run_lstm("G_B_sameenv", "empirical", env)
    run_lstm("H_single_balance", "unit", env)
    env["P2_OUTPUT_DIR"] = str(OUTPUT)
    run(
        [sys.executable, "-u", str(ROOT / "train_static_lr.py")],
        env,
        OUTPUT / "static_lr_train.log",
    )
    for arm in ARMS:
        if not (OUTPUT / arm / "lstm_model.pt").is_file():
            raise RuntimeError(f"Missing checkpoint: {arm}")


if __name__ == "__main__":
    main()
