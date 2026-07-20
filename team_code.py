#!/usr/bin/env python
"""Challenge entry points backed by the current 12-D ECG LSTM pipeline.

The official scripts call only ``train_model``, ``load_model`` and ``run_model``.
Local experiments keep using the unchanged cache-backed scripts directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import infer_lstm
from helper_code import DEMOGRAPHICS_FILE, HEADERS, load_label


SEQ_FEATURE_DIM = infer_lstm.SEQ_DIM
ECG_DIM = infer_lstm.ECG_DIM
STATIC_DIM = infer_lstm.STATIC_DIM


ROOT = Path(__file__).resolve().parent
SPLIT_NAMES = ("train", "val", "test", "external")


def _record_key(record):
    return f"{record[HEADERS['bids_folder']]}_ses-{record[HEADERS['session_id']]}"


def _record_identifiers(record):
    return {
        HEADERS["bids_folder"]: record[HEADERS["bids_folder"]],
        HEADERS["site_id"]: record[HEADERS["site_id"]],
        HEADERS["session_id"]: record[HEADERS["session_id"]],
    }


def _normalise_arrays(X_seq, X_ecg, x_static, mask):
    X_seq = np.asarray(X_seq, dtype=np.float32)
    if X_seq.ndim != 2 or len(X_seq) == 0 or X_seq.shape[1] != SEQ_FEATURE_DIM:
        raise ValueError(f"invalid X_seq shape {X_seq.shape}")

    if X_ecg is None:
        X_ecg = np.zeros((0, ECG_DIM), dtype=np.float32)
    else:
        X_ecg = np.asarray(X_ecg, dtype=np.float32)
    if X_ecg.ndim != 2 or X_ecg.shape[1] != ECG_DIM:
        raise ValueError(f"invalid X_ecg shape {X_ecg.shape}")

    x_static = np.asarray(x_static, dtype=np.float32)
    if x_static.shape != (STATIC_DIM,):
        raise ValueError(f"invalid x_static shape {x_static.shape}")

    if mask is None:
        mask = np.zeros(len(X_seq), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(X_seq),):
        raise ValueError(f"invalid mask shape {mask.shape}")

    return (
        np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(X_ecg, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(x_static, nan=0.0, posinf=0.0, neginf=0.0),
        mask,
    )


@contextmanager
def _label_optional_extraction():
    """Make feature extraction work when official holdout labels are hidden."""
    import per_epoch_features.per_epoch_extractor as extractor_module

    original = extractor_module.load_diagnoses
    extractor_module.load_diagnoses = lambda *_args, **_kwargs: -1
    try:
        yield
    finally:
        extractor_module.load_diagnoses = original


def _extract_features(record, data_folder):
    from per_epoch_features.per_epoch_extractor import PerEpochExtractor

    with _label_optional_extraction():
        X_seq, X_ecg, x_static, _unused_y, mask = PerEpochExtractor().extract_all(
            record, str(data_folder)
        )
    return _normalise_arrays(X_seq, X_ecg, x_static, mask)


def _has_fixed_local_cache(data_folder):
    try:
        is_local_data = Path(data_folder).resolve() == (ROOT / "data").resolve()
    except OSError:
        return False
    return is_local_data and all(
        (ROOT / "npz_full" / split).is_dir() and (ROOT / "split" / f"{split}_records.json").is_file()
        for split in SPLIT_NAMES
    )


def _stable_train_val_split(frame, validation_fraction=0.20):
    """Deterministic label-stratified split for official training data."""
    train_indices, val_indices = [], []
    labels = frame.apply(lambda row: load_label(row.to_dict()), axis=1)
    for label in sorted(labels.unique()):
        indices = list(frame.index[labels == label])
        indices.sort(
            key=lambda idx: hashlib.sha256(
                _record_key(frame.loc[idx].to_dict()).encode("utf-8")
            ).hexdigest()
        )
        n_val = max(1, int(round(len(indices) * validation_fraction)))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    return frame.loc[train_indices], frame.loc[val_indices]


def _write_records(frame, path):
    records = [_record_identifiers(row) for row in frame.to_dict("records")]
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def _cache_records(records, rows_by_key, split, cache_root, data_folder):
    split_dir = cache_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        key = _record_key(record)
        output_path = split_dir / f"{key}.npz"
        row = rows_by_key[key]
        X_seq, X_ecg, x_static, mask = _extract_features(record, data_folder)
        np.savez_compressed(
            output_path,
            X_seq=X_seq,
            X_ecg=X_ecg,
            x_static=x_static,
            y=np.asarray(load_label(row), dtype=np.int64),
            mask=mask,
        )


def _link_validation_as_test(cache_root, validation_records):
    test_dir = cache_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for record in validation_records:
        name = f"{_record_key(record)}.npz"
        source, target = cache_root / "val" / name, test_dir / name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def _run_unchanged_trainer(data_folder, model_folder, splits_dir, cache_dir, verbose):
    env = os.environ.copy()
    env.update(
        {
            "LSTM_DATA_FOLDER": str(data_folder),
            "LSTM_SPLITS_DIR": str(splits_dir),
            "LSTM_CACHE_DIR": str(cache_dir),
            "LSTM_MODEL_DIR": str(model_folder),
            "LSTM_SEED": env.get("LSTM_SEED", "7"),
            "PYTHONHASHSEED": env.get("LSTM_SEED", "7"),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    command = [sys.executable, str(ROOT / "train_lstm.py")]
    if verbose:
        print("Running unchanged trainer:", " ".join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def train_model(data_folder, model_folder, verbose):
    """Train through the official API while preserving the current LSTM code."""
    data_folder = Path(data_folder).resolve()
    model_folder = Path(model_folder).resolve()
    model_folder.mkdir(parents=True, exist_ok=True)

    if _has_fixed_local_cache(data_folder):
        _run_unchanged_trainer(data_folder, model_folder, ROOT / "split", ROOT / "npz_full", verbose)
        return

    demographics_path = data_folder / DEMOGRAPHICS_FILE
    frame = pd.read_csv(demographics_path)
    train_frame, val_frame = _stable_train_val_split(frame)
    rows_by_key = {_record_key(row): row for row in frame.to_dict("records")}

    with tempfile.TemporaryDirectory(prefix="challenge2026_train_") as temp_name:
        temp_root = Path(temp_name)
        splits_dir, cache_root = temp_root / "split", temp_root / "npz"
        splits_dir.mkdir(parents=True)
        train_records = _write_records(train_frame, splits_dir / "train_records.json")
        val_records = _write_records(val_frame, splits_dir / "val_records.json")
        (splits_dir / "test_records.json").write_text(
            (splits_dir / "val_records.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        _cache_records(train_records, rows_by_key, "train", cache_root, data_folder)
        _cache_records(val_records, rows_by_key, "val", cache_root, data_folder)
        _link_validation_as_test(cache_root, val_records)
        _run_unchanged_trainer(data_folder, model_folder, splits_dir, cache_root, verbose)


def load_model(model_folder, verbose):
    model_folder = Path(model_folder).resolve()
    model_path = model_folder / "lstm_model.pt"
    checkpoint = torch.load(model_path, map_location=infer_lstm.device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint format: {model_path}")

    saved_dims = checkpoint.get("feature_dims")
    expected_dims = {"X_seq": SEQ_FEATURE_DIM, "X_ecg": ECG_DIM, "x_static": STATIC_DIM}
    if saved_dims is not None and saved_dims != expected_dims:
        raise RuntimeError(f"Checkpoint feature dimensions {saved_dims} do not match {expected_dims}")

    network = infer_lstm.LSTMModel().to(infer_lstm.device)
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()

    cache_roots = []
    configured_cache = os.environ.get("LSTM_NPZ_CACHE")
    if configured_cache:
        cache_roots.append(Path(configured_cache))
    if (ROOT / "npz_full").is_dir():
        cache_roots.append(ROOT / "npz_full")
    cache_roots.append(model_folder / "inference_cache")
    if verbose:
        print(f"Loaded {model_path}; cache search roots={cache_roots}")
    return {
        "network": network,
        "cache_roots": cache_roots,
        "calibrator": checkpoint.get("calibrator"),
        "threshold": float(checkpoint.get("threshold", 0.5)),
    }


def _cached_sample(record, cache_roots):
    name = f"{_record_key(record)}.npz"
    for root in cache_roots:
        candidates = [root / name] + [root / split / name for split in SPLIT_NAMES]
        for path in candidates:
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as data:
                X_seq, X_ecg, x_static, mask = _normalise_arrays(
                    data["X_seq"], data["X_ecg"], data["x_static"], data["mask"]
                )
            return X_seq, X_ecg, x_static, mask
    return None


def _sample(record, data_folder, cache_roots):
    arrays = _cached_sample(record, cache_roots)
    if arrays is None:
        arrays = _extract_features(record, data_folder)
        target = cache_roots[-1] / f"{_record_key(record)}.npz"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                X_seq=arrays[0], X_ecg=arrays[1], x_static=arrays[2],
                y=np.asarray(-1, dtype=np.int64), mask=arrays[3],
            )
        except OSError:
            pass
    X_seq, X_ecg, x_static, mask = arrays
    return {
        "X_seq": np.clip(X_seq, -50.0, 50.0),
        "X_ecg": np.clip(X_ecg, -50.0, 50.0),
        "x_static": np.clip(x_static, -50.0, 50.0),
        "y": -1,
        "mask": mask,
        "length": len(X_seq),
        "rec_key": _record_key(record),
    }


def run_model(model, record, data_folder, verbose):
    """Run one official record; failures are raised instead of hidden as 0.5."""
    sample = _sample(record, data_folder, model["cache_roots"])
    _labels, logits, _keys = infer_lstm.infer_batched(
        model["network"], [sample], batch_size=1
    )
    probability = float(infer_lstm.apply_calibrator(logits, model["calibrator"])[0])
    if not np.isfinite(probability):
        raise RuntimeError(f"Non-finite prediction for {_record_key(record)}")
    probability = float(np.clip(probability, 0.0, 1.0))
    return bool(probability >= model["threshold"]), probability
