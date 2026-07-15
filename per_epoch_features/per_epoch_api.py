#!/usr/bin/env python
"""Public API for extracting per-epoch model features from one PSG record.

Typical Python usage::

    from per_epoch_features import extract_features

    features = extract_features(
        data_folder="/path/to/training_set",
        bids_folder="sub-I0002150005420",
        site_id="I0002",
        session_id=1,
    )
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from .per_epoch_extractor import (
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
    DEMOGRAPHICS_FILE,
    ECG_DIM,
    PHYSIOLOGICAL_DATA_SUBFOLDER,
    SEQ_FEATURE_DIM,
    STATIC_DIM,
    PerEpochExtractor,
)


PathLike = Union[str, os.PathLike]
FeatureDict = Dict[str, np.ndarray]

REQUIRED_FEATURE_KEYS = ("X_seq", "X_ecg", "x_static", "mask")


class FeatureExtractionError(RuntimeError):
    """Raised when a record cannot be converted into model-ready features."""


def _record_key(record: Mapping[str, Any]) -> str:
    return f"{record['BidsFolder']}_ses-{record['SessionID']}"


def _validate_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    required = ("BidsFolder", "SiteID", "SessionID")
    missing = [key for key in required if record.get(key) is None]
    if missing:
        raise ValueError(
            "record is missing required field(s): " + ", ".join(missing)
        )
    return {key: record[key] for key in required}


def _expected_paths(data_folder: Path, record: Mapping[str, Any]) -> Dict[str, Path]:
    rec_key = _record_key(record)
    site_id = str(record["SiteID"])
    return {
        "demographics": data_folder / DEMOGRAPHICS_FILE,
        "physiological": (
            data_folder / PHYSIOLOGICAL_DATA_SUBFOLDER / site_id
            / f"{rec_key}.edf"
        ),
        "algorithmic": (
            data_folder / ALGORITHMIC_ANNOTATIONS_SUBFOLDER / site_id
            / f"{rec_key}_caisr_annotations.edf"
        ),
    }


def _normalise_features(
    X_seq: Any,
    X_ecg: Any,
    x_static: Any,
    mask: Any,
) -> FeatureDict:
    if X_seq is None:
        raise FeatureExtractionError("per-epoch sequence features were not produced")

    sequence = np.asarray(X_seq, dtype=np.float32)
    static = np.asarray(x_static, dtype=np.float32)
    if X_ecg is None:
        ecg = np.zeros((0, ECG_DIM), dtype=np.float32)
    else:
        ecg = np.asarray(X_ecg, dtype=np.float32)
    if mask is None:
        ecg_mask = np.zeros(sequence.shape[0], dtype=bool)
    else:
        ecg_mask = np.asarray(mask, dtype=bool)

    if sequence.ndim != 2 or sequence.shape[1] != SEQ_FEATURE_DIM:
        raise FeatureExtractionError(
            f"invalid X_seq shape {sequence.shape}; expected (N, {SEQ_FEATURE_DIM})"
        )
    if ecg.ndim != 2 or ecg.shape[1] != ECG_DIM:
        raise FeatureExtractionError(
            f"invalid X_ecg shape {ecg.shape}; expected (M, {ECG_DIM})"
        )
    if static.shape != (STATIC_DIM,):
        raise FeatureExtractionError(
            f"invalid x_static shape {static.shape}; expected ({STATIC_DIM},)"
        )
    if ecg_mask.shape != (sequence.shape[0],):
        raise FeatureExtractionError(
            f"invalid mask shape {ecg_mask.shape}; expected ({sequence.shape[0]},)"
        )

    return {
        "X_seq": np.ascontiguousarray(sequence),
        "X_ecg": np.ascontiguousarray(ecg),
        "x_static": np.ascontiguousarray(static),
        "mask": np.ascontiguousarray(ecg_mask),
    }


def extract_features_from_record(
    record: Mapping[str, Any],
    data_folder: PathLike,
    *,
    channel_table_path: Optional[PathLike] = None,
) -> FeatureDict:
    """Extract model-ready features for one record.

    Parameters
    ----------
    record
        Mapping containing ``BidsFolder``, ``SiteID`` and ``SessionID``.
    data_folder
        Dataset root containing ``demographics.csv``, ``physiological_data``
        and ``algorithmic_annotations``.
    channel_table_path
        Optional custom channel alias table. The project table is used by
        default.

    Returns
    -------
    dict
        ``X_seq`` (N, 483), ``X_ecg`` (M, 37), ``x_static`` (196,) and
        ``mask`` (N,). Missing ECG is represented by an empty (0, 37) array.
    """
    clean_record = _validate_record(record)
    root = Path(data_folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data_folder does not exist: {root}")

    paths = _expected_paths(root, clean_record)
    if not paths["demographics"].is_file():
        raise FileNotFoundError(
            f"demographics file does not exist: {paths['demographics']}"
        )

    extractor = PerEpochExtractor(
        csv_path=str(channel_table_path) if channel_table_path is not None else None
    )
    try:
        X_seq, X_ecg, x_static, _label, mask = extractor.extract_all(
            clean_record, str(root),
        )
        return _normalise_features(X_seq, X_ecg, x_static, mask)
    except FeatureExtractionError as exc:
        missing_files = [
            str(path) for name, path in paths.items()
            if name != "demographics" and not path.is_file()
        ]
        detail = (
            " Missing expected file(s): " + ", ".join(missing_files)
            if missing_files else
            " Check that the EDF files contain stage, EEG, EMG and respiratory channels."
        )
        raise FeatureExtractionError(
            f"failed to extract {_record_key(clean_record)}.{detail}"
        ) from exc
    except Exception as exc:
        raise FeatureExtractionError(
            f"failed to extract {_record_key(clean_record)}: {exc}"
        ) from exc


def extract_features(
    data_folder: PathLike,
    bids_folder: str,
    site_id: str,
    session_id: Any,
    *,
    channel_table_path: Optional[PathLike] = None,
) -> FeatureDict:
    """Convenience wrapper accepting record identifiers as arguments."""
    return extract_features_from_record(
        {
            "BidsFolder": bids_folder,
            "SiteID": site_id,
            "SessionID": session_id,
        },
        data_folder,
        channel_table_path=channel_table_path,
    )


def save_features(features: Mapping[str, Any], output_path: PathLike) -> Path:
    """Validate and save a feature dictionary as compressed NPZ."""
    missing = [key for key in REQUIRED_FEATURE_KEYS if key not in features]
    if missing:
        raise ValueError("features missing key(s): " + ", ".join(missing))
    normalised = _normalise_features(
        features["X_seq"], features["X_ecg"],
        features["x_static"], features["mask"],
    )
    path = Path(output_path).expanduser().resolve()
    if path.suffix.lower() != ".npz":
        path = Path(str(path) + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **normalised)
    return path


def load_features(input_path: PathLike) -> FeatureDict:
    """Load and validate a feature dictionary saved by :func:`save_features`."""
    path = Path(input_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_FEATURE_KEYS if key not in data]
        if missing:
            raise ValueError("feature file missing key(s): " + ", ".join(missing))
        return _normalise_features(
            data["X_seq"], data["X_ecg"], data["x_static"], data["mask"],
        )


def _coerce_session_id(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        return value


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-epoch PSG features and save them as NPZ.",
    )
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--bids-folder", required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--session-id", required=True, type=_coerce_session_id)
    parser.add_argument("--output", required=True)
    parser.add_argument("--channel-table", default=None)
    args = parser.parse_args(argv)

    features = extract_features(
        data_folder=args.data_folder,
        bids_folder=args.bids_folder,
        site_id=args.site_id,
        session_id=args.session_id,
        channel_table_path=args.channel_table,
    )
    output = save_features(features, args.output)
    summary = {
        "output": str(output),
        "shapes": {key: list(value.shape) for key, value in features.items()},
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FeatureDict",
    "FeatureExtractionError",
    "extract_features",
    "extract_features_from_record",
    "load_features",
    "save_features",
]
