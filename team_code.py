#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

import infer_lstm
import train_lstm
from helper_code import HEADERS

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LSTM_DATA_FOLDER = Path(os.environ.get("LSTM_DATA_FOLDER", SCRIPT_DIR))
DEFAULT_LSTM_CACHE_DIR = os.environ.get("LSTM_CACHE_DIR")


def _cache_root_for_model(model_folder):
    model_dir = Path(model_folder)
    return Path(DEFAULT_LSTM_CACHE_DIR) if DEFAULT_LSTM_CACHE_DIR else model_dir / "lstm_cache"


def _configure_lstm_paths(data_folder, model_folder):
    data_dir = Path(data_folder)
    model_dir = Path(model_folder)
    cache_dir = _cache_root_for_model(model_dir)

    train_lstm.DATA_FOLDER = str(data_dir)
    train_lstm.SPLITS_DIR = str(data_dir / "splits")
    train_lstm.CACHE_DIR = str(cache_dir)
    train_lstm.MODEL_DIR = str(model_dir)

    return data_dir, cache_dir, model_dir


def train_model(data_folder, model_folder, verbose):
    _configure_lstm_paths(data_folder, model_folder)
    train_lstm.main()


def load_model(model_folder, verbose):
    model_path = Path(model_folder) / "lstm_model.pt"
    model = infer_lstm.LSTMModel().to(infer_lstm.device)
    checkpoint = torch.load(model_path, map_location=infer_lstm.device, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return {"model": model, "model_path": str(model_path), "cache_dir": str(_cache_root_for_model(model_folder))}


def _record_key(record):
    patient_id = record[HEADERS["bids_folder"]]
    session_id = record[HEADERS["session_id"]]
    return f"{patient_id}_ses-{session_id}"


def _find_record_cache_dir(cache_root, rec_key):
    cache_root = Path(cache_root)
    for split_name in ("test", "external", "val", "train"):
        candidate_dir = cache_root / split_name
        if (candidate_dir / f"{rec_key}.npz").exists():
            return candidate_dir
    for candidate_dir in cache_root.iterdir() if cache_root.exists() else []:
        if candidate_dir.is_dir() and (candidate_dir / f"{rec_key}.npz").exists():
            return candidate_dir
    raise FileNotFoundError(f"No cached LSTM npz found for {rec_key} under {cache_root}")


def run_model(model, record, data_folder, verbose):
    rec_key = _record_key(record)
    try:
        cache_dir = _find_record_cache_dir(model["cache_dir"], rec_key)
        samples = infer_lstm.preload_all(str(cache_dir), [record])
    except FileNotFoundError:
        samples = []
    if not samples:
        try:
            sample = _extract_record_sample(record, data_folder, model["cache_dir"], rec_key)
        except Exception as exc:
            if verbose:
                print(f"Feature extraction failed for {rec_key}: {exc}")
            sample = None
        samples = [sample] if sample is not None else []
    if not samples:
        return False, 0.5
    _y_true, y_prob, _keys = infer_lstm.infer_batched(model["model"], samples, batch_size=1)
    prob = float(np.clip(np.nan_to_num(y_prob[0], nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0))
    return bool(prob >= 0.5), prob


def _extract_record_sample(record, data_folder, cache_root, rec_key):
    from per_epoch_extractor import PerEpochExtractor

    extractor = PerEpochExtractor()
    X_seq, X_ecg, x_static, y, mask = extractor.extract_all(record, data_folder)
    if X_seq is None or len(X_seq) == 0:
        return None

    if X_ecg is None:
        X_ecg = np.zeros((0, infer_lstm.ECG_DIM), dtype=np.float32)
    if mask is None:
        mask = np.zeros(len(X_seq), dtype=bool)

    cache_dir = Path(cache_root) / "holdout"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_dir / f"{rec_key}.npz",
            X_seq=X_seq,
            X_ecg=X_ecg,
            x_static=x_static,
            y=y,
            mask=mask,
        )
    except Exception:
        pass

    return {
        "X_seq": np.clip(np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0),
        "X_ecg": np.clip(np.nan_to_num(X_ecg, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0),
        "x_static": np.clip(np.nan_to_num(x_static, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0),
        "y": int(y) if y is not None else -1,
        "mask": mask,
        "length": len(X_seq),
        "rec_key": rec_key,
    }

def _run_infer_cli(data_folder, model_folder, output_dir, cache_dir, split, batch_size, nested=False):
    split_output_dir = Path(output_dir) / split if nested else Path(output_dir)
    argv = [
        "infer_lstm.py",
        "--data_folder", str(data_folder),
        "--model", str(Path(model_folder) / "lstm_model.pt"),
        "--output_dir", str(split_output_dir),
        "--cache_dir", str(Path(cache_dir) / split),
        "--split", split,
        "--batch_size", str(batch_size),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        infer_lstm.main()
    finally:
        sys.argv = old_argv


def main(argv=None):
    global DEFAULT_LSTM_DATA_FOLDER, DEFAULT_LSTM_CACHE_DIR
    parser = argparse.ArgumentParser(description="Official team_code.py bridge for the challenge2026 LSTM route.")
    parser.add_argument("--mode", choices=["train", "infer", "all"], default="train")
    parser.add_argument("--data_folder", default=str(DEFAULT_LSTM_DATA_FOLDER))
    parser.add_argument("--model_folder", default=str(SCRIPT_DIR / "lstm_model_kaggle"))
    parser.add_argument("--cache_dir", default=DEFAULT_LSTM_CACHE_DIR)
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "lstm_results_kaggle"))
    parser.add_argument("--split", choices=["train", "val", "test", "external"], default="test")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    DEFAULT_LSTM_DATA_FOLDER = Path(args.data_folder)
    DEFAULT_LSTM_CACHE_DIR = args.cache_dir

    if args.mode in ("train", "all"):
        train_model(args.data_folder, args.model_folder, args.verbose)

    if args.mode == "infer":
        _run_infer_cli(args.data_folder, args.model_folder, args.output_dir, args.cache_dir, args.split, args.batch_size)
    elif args.mode == "all":
        _run_infer_cli(args.data_folder, args.model_folder, args.output_dir, args.cache_dir, "test", args.batch_size)
        _run_infer_cli(args.data_folder, args.model_folder, args.output_dir, args.cache_dir, "external", args.batch_size, nested=True)


if __name__ == "__main__":
    main()
