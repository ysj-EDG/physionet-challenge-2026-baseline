"""I/O and schema guards for the bounded G2-A smoke workflow."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXPECTED_MANIFEST_SHA256 = "7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a"
EXPECTED_EXTRACTION = "timegrid_v2.1.0"
EXPECTED_VALIDITY = "validity_v0.2"
VIEW_SIZES = {"V1": 113, "V2": 129, "V3": 349, "V4": 113, "V5": 123}
REQUIRED_MANIFEST_FIELDS = {
    "view_id", "view_position", "unique_feature_id", "family", "source_npz_field",
    "source_index", "source_indices", "aggregation", "stage_group",
    "validity_expression", "min_valid_observations", "missing_semantics",
}
REQUIRED_NPZ_FIELDS = {
    "X_seq", "X_ecg", "x_static", "extraction_version", "validity_schema_version",
    "stage_code_aligned", "stage_valid", "eeg_channel_available",
    "eeg_clean_subsegment_count", "emg_channel_available", "emg_epoch_success",
    "resp_feature_valid", "hrv_success", "hrv_feature_valid", "ecg_alignment_valid",
    "arousal_valid", "resp_event_valid", "limb_event_valid", "annotation_status",
    "stage_probability", "stage_probability_valid", "stage_probability_channels_present",
    "annotation_signal_lengths_json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class FrozenManifest:
    rows: list[dict[str, str]]
    unique_ids: list[str]
    view_sizes: dict[str, int]
    sha256: str


def load_manifest(path: Path, expected_sha256: str = EXPECTED_MANIFEST_SHA256) -> FrozenManifest:
    path = Path(path)
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"Frozen Manifest SHA256 mismatch: {digest}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_MANIFEST_FIELDS.issubset(reader.fieldnames or ()):
            raise ValueError("Frozen Manifest required fields missing")
        rows = list(reader)
    if len(rows) != 827:
        raise ValueError("Frozen Manifest must contain 827 rows")
    sizes = {v: sum(r["view_id"] == v for r in rows) for v in VIEW_SIZES}
    if sizes != VIEW_SIZES:
        raise ValueError(f"Frozen Manifest view sizes invalid: {sizes}")
    for view, size in sizes.items():
        positions = [int(r["view_position"]) for r in rows if r["view_id"] == view]
        if positions != list(range(size)):
            raise ValueError(f"Non-contiguous positions in {view}")
    unique_ids = list(dict.fromkeys(r["unique_feature_id"] for r in rows))
    if len(unique_ids) != 807:
        raise ValueError("Frozen Manifest must contain 807 unique IDs")
    return FrozenManifest(rows, unique_ids, sizes, digest)


def _scalar(z, key: str) -> str:
    value = np.asarray(z[key])
    if value.shape != ():
        raise ValueError(f"{key} must be scalar")
    return str(value.item())


def load_npz_record(path: Path) -> dict:
    """Load approved fields only; deliberately never accesses the label key `y`."""
    with np.load(Path(path), allow_pickle=False) as z:
        missing = REQUIRED_NPZ_FIELDS.difference(z.files)
        if missing:
            raise ValueError(f"NPZ required fields missing: {sorted(missing)}")
        if _scalar(z, "extraction_version") != EXPECTED_EXTRACTION:
            raise ValueError("Unexpected extraction_version")
        if _scalar(z, "validity_schema_version") != EXPECTED_VALIDITY:
            raise ValueError("Unexpected validity_schema_version")
        record = {key: np.asarray(z[key]).copy() for key in REQUIRED_NPZ_FIELDS
                  if key not in {"extraction_version", "validity_schema_version",
                                 "annotation_status", "annotation_signal_lengths_json"}}
        record["extraction_version"] = EXPECTED_EXTRACTION
        record["validity_schema_version"] = EXPECTED_VALIDITY
        record["annotation_status"] = _scalar(z, "annotation_status")
        try:
            lengths = json.loads(_scalar(z, "annotation_signal_lengths_json"))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Invalid annotation_signal_lengths_json") from exc
        if not isinstance(lengths, dict):
            raise ValueError("annotation_signal_lengths_json must decode to an object")
        record["annotation_signal_lengths"] = lengths
    t = record["X_seq"].shape[0] if record["X_seq"].ndim == 2 else -1
    w = record["X_ecg"].shape[0] if record["X_ecg"].ndim == 2 else -1
    shapes = {
        "X_seq": (t, 483), "X_ecg": (w, 12), "x_static": (196,),
        "stage_code_aligned": (t,), "stage_valid": (t,),
        "eeg_channel_available": (6,), "eeg_clean_subsegment_count": (t, 6),
        "emg_channel_available": (3,), "emg_epoch_success": (t, 3),
        "resp_feature_valid": (t, 14), "hrv_success": (w,),
        "hrv_feature_valid": (w, 11), "ecg_alignment_valid": (t,),
        "arousal_valid": (t,), "resp_event_valid": (t,), "limb_event_valid": (t,),
        "stage_probability": (t, 5), "stage_probability_valid": (t,),
        "stage_probability_channels_present": (5,),
    }
    for key, shape in shapes.items():
        if record[key].shape != shape:
            raise ValueError(f"Invalid {key} shape {record[key].shape}, expected {shape}")
    if t < 1 or w < 0:
        raise ValueError("Invalid time axes")
    return record


def read_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        return {key: np.asarray(z[key]).copy() for key in z.files}


def _same_array(left, right):
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _verify_roundtrip(payload, loaded):
    if set(loaded) != set(payload):
        raise ValueError("Cache round-trip keys mismatch")
    for key in payload:
        if not _same_array(payload[key], loaded[key]):
            raise ValueError(f"Cache round-trip mismatch: {key}")


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_cache_atomic(path: Path, arrays: dict[str, object], payload_validator=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype == object for value in payload.values()):
        raise ValueError("Object arrays are forbidden")
    if payload_validator is not None:
        payload_validator(payload)
    temporary = None
    linked = False; completed = False
    try:
        with tempfile.NamedTemporaryFile("wb", suffix=".npz.tmp", prefix=path.stem + ".",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush(); os.fsync(handle.fileno())
        loaded = read_cache(temporary)
        _verify_roundtrip(payload, loaded)
        if payload_validator is not None:
            payload_validator(loaded)
        os.link(temporary, path)
        linked = True
        _verify_roundtrip(payload, read_cache(path))
        _fsync_directory(path.parent)
        temporary.unlink()
        temporary = None
        _fsync_directory(path.parent)
        completed = True
    except Exception:
        if linked and path.exists():
            path.unlink()
            _fsync_directory(path.parent)
        raise
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
            _fsync_directory(path.parent)
    if not completed:
        raise RuntimeError("Cache publication did not complete")
