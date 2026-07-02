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
DEFAULT_LSTM_CACHE_DIR = Path(os.environ.get("LSTM_CACHE_DIR", SCRIPT_DIR / "lstm_cache_kaggle"))


def _configure_lstm_paths(model_folder):
    data_folder = DEFAULT_LSTM_DATA_FOLDER
    cache_dir = DEFAULT_LSTM_CACHE_DIR
    model_dir = Path(model_folder)

    train_lstm.DATA_FOLDER = str(data_folder)
    train_lstm.SPLITS_DIR = str(data_folder / "splits")
    train_lstm.CACHE_DIR = str(cache_dir)
    train_lstm.MODEL_DIR = str(model_dir)

    return data_folder, cache_dir, model_dir


def train_model(data_folder, model_folder, verbose):
    _configure_lstm_paths(model_folder)
    train_lstm.main()


def load_model(model_folder, verbose):
    model_path = Path(model_folder) / "lstm_model.pt"
    model = infer_lstm.LSTMModel().to(infer_lstm.device)
    checkpoint = torch.load(model_path, map_location=infer_lstm.device, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return {"model": model, "model_path": str(model_path), "cache_dir": str(DEFAULT_LSTM_CACHE_DIR)}


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
    cache_dir = _find_record_cache_dir(model["cache_dir"], rec_key)
    samples = infer_lstm.preload_all(str(cache_dir), [record])
    if not samples:
        raise FileNotFoundError(f"No cached LSTM sample loaded for {rec_key}")
    _y_true, y_prob, _keys = infer_lstm.infer_batched(model["model"], samples, batch_size=1)
    prob = float(np.clip(np.nan_to_num(y_prob[0], nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0))
    return bool(prob >= 0.5), prob

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
    parser.add_argument("--cache_dir", default=str(DEFAULT_LSTM_CACHE_DIR))
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "lstm_results_kaggle"))
    parser.add_argument("--split", choices=["train", "val", "test", "external"], default="test")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    DEFAULT_LSTM_DATA_FOLDER = Path(args.data_folder)
    DEFAULT_LSTM_CACHE_DIR = Path(args.cache_dir)

    if args.mode in ("train", "all"):
        train_model(args.data_folder, args.model_folder, args.verbose)

    if args.mode == "infer":
        _run_infer_cli(args.data_folder, args.model_folder, args.output_dir, args.cache_dir, args.split, args.batch_size)
    elif args.mode == "all":
        _run_infer_cli(args.data_folder, args.model_folder, args.output_dir, args.cache_dir, "test", args.batch_size)
        _run_infer_cli(args.data_folder, args.model_folder, args.output_dir, args.cache_dir, "external", args.batch_size, nested=True)


if __name__ == "__main__":
    main()
