#!/usr/bin/env python
"""Official Challenge entry point for on-demand feature extraction and LSTM inference."""

from pathlib import Path

import numpy as np
import torch

import infer_lstm
import train_lstm
from helper_code import HEADERS
from per_epoch_features.per_epoch_extractor import ECG_DIM, SEQ_FEATURE_DIM, STATIC_DIM, PerEpochExtractor


def _record_key(record):
    return f"{record[HEADERS['bids_folder']]}_ses-{record[HEADERS['session_id']]}"


def _cache_root(model_folder):
    return Path(model_folder) / "lstm_cache"


def _normalise_feature_arrays(X_seq, X_ecg, x_static, mask):
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
    return X_seq, X_ecg, x_static, mask


def train_model(data_folder, model_folder, verbose):
    """Official training: extract/cache features from the supplied Challenge data."""
    train_lstm.main(
        data_folder=data_folder,
        model_dir=model_folder,
        cache_dir=_cache_root(model_folder),
        verbose=verbose,
    )


def load_model(model_folder, verbose):
    model_path = Path(model_folder) / "lstm_model.pt"
    checkpoint = torch.load(model_path, map_location=infer_lstm.device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported LSTM checkpoint format: {model_path}")

    expected_dims = {"X_seq": SEQ_FEATURE_DIM, "X_ecg": ECG_DIM, "x_static": STATIC_DIM}
    saved_dims = checkpoint.get("feature_dims")
    if saved_dims is not None and saved_dims != expected_dims:
        raise RuntimeError(f"Checkpoint feature dimensions {saved_dims} do not match {expected_dims}")

    network = infer_lstm.LSTMModel().to(infer_lstm.device)
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    return {
        "network": network,
        "cache_dir": str(_cache_root(model_folder)),
        "calibrator": checkpoint.get("calibrator"),
        "threshold": float(checkpoint.get("threshold", 0.5)),
    }


def _load_or_extract_sample(record, data_folder, cache_dir):
    rec_key = _record_key(record)
    path = Path(cache_dir) / "holdout" / f"{rec_key}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                X_seq, X_ecg, x_static, mask = _normalise_feature_arrays(
                    data["X_seq"], data["X_ecg"], data["x_static"], data["mask"]
                )
                y = int(data["y"])
        except Exception:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        else:
            return {
                "X_seq": np.clip(np.nan_to_num(X_seq), -50.0, 50.0),
                "X_ecg": np.clip(np.nan_to_num(X_ecg), -50.0, 50.0),
                "x_static": np.clip(np.nan_to_num(x_static), -50.0, 50.0),
                "y": y, "mask": mask, "length": len(X_seq), "rec_key": rec_key,
            }

    X_seq, X_ecg, x_static, y, mask = PerEpochExtractor().extract_all(record, str(data_folder))
    X_seq, X_ecg, x_static, mask = _normalise_feature_arrays(X_seq, X_ecg, x_static, mask)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X_seq=X_seq, X_ecg=X_ecg, x_static=x_static,
                            y=-1 if y is None else int(y), mask=mask)
    except OSError:
        pass
    return {
        "X_seq": np.clip(np.nan_to_num(X_seq), -50.0, 50.0),
        "X_ecg": np.clip(np.nan_to_num(X_ecg), -50.0, 50.0),
        "x_static": np.clip(np.nan_to_num(x_static), -50.0, 50.0),
        "y": -1 if y is None else int(y), "mask": mask, "length": len(X_seq), "rec_key": rec_key,
    }


def run_model(model, record, data_folder, verbose):
    """Official holdout inference for one record."""
    rec_key = _record_key(record)
    try:
        sample = _load_or_extract_sample(record, data_folder, model["cache_dir"])
        _y, logits, _keys = infer_lstm.infer_batched(model["network"], [sample], batch_size=1)
        probability = float(infer_lstm.apply_calibrator(logits, model["calibrator"])[0])
        probability = float(np.clip(np.nan_to_num(probability, nan=0.5), 0.0, 1.0))
        return bool(probability >= model["threshold"]), probability
    except Exception as exc:
        if verbose:
            print(f"Feature extraction or inference failed for {rec_key}: {exc}")
        return False, 0.5
