"""Frozen DATA2 loading and CI-blind coherence sampling for Physio Fusion V1."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SEQ_DIM = 483
ECG_DIM = 12
SPECTRAL = slice(0, 54)
COHERENCE = slice(54, 414)
BSR = slice(414, 432)
EMG = slice(432, 456)
RESP = slice(456, 470)
STAGE_EVENT = slice(470, 483)

EXPECTED_SITE_COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}
EXPECTED_EXTRACTION_VERSION = "timegrid_v2.1.0"
EXPECTED_VALIDITY_SCHEMA = "validity_v0.2"
EEG_PAIRS = tuple(combinations(range(6), 2))
MODEL_FAMILIES = {
    "record_id", "site", "y", "age",
    "spectral", "spectral_valid", "coherence", "coherence_pair_valid",
    "bsr", "bsr_valid", "emg", "emg_valid", "resp", "resp_valid",
    "hrv", "hrv_valid",
}


@dataclass(frozen=True)
class RecordInfo:
    path: Path
    record_id: str
    site: str
    y: int


@dataclass(frozen=True)
class RobustScaler24:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return ((values - self.center) / self.scale).astype(np.float32, copy=False)


def _scalar(z: np.lib.npyio.NpzFile, key: str):
    return np.asarray(z[key]).item()


def stable_uint64(*parts: object) -> int:
    text = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little", signed=False)


def discover_records(npz_root: Path, validate_schema: bool = True) -> list[RecordInfo]:
    paths = sorted(Path(npz_root).rglob("*.npz"))
    records: list[RecordInfo] = []
    seen: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            record_id = str(_scalar(z, "record_id"))
            site = str(_scalar(z, "site_id"))
            y = int(_scalar(z, "y"))
            if record_id in seen:
                raise ValueError(f"duplicate record_id: {record_id}")
            seen.add(record_id)
            if path.parent.name != site:
                raise ValueError(f"path/metadata site mismatch for {record_id}")
            if y not in (0, 1):
                raise ValueError(f"non-binary label for {record_id}: {y}")
            if validate_schema:
                if str(_scalar(z, "extraction_version")) != EXPECTED_EXTRACTION_VERSION:
                    raise ValueError(f"wrong extraction version: {record_id}")
                if str(_scalar(z, "validity_schema_version")) != EXPECTED_VALIDITY_SCHEMA:
                    raise ValueError(f"wrong validity schema: {record_id}")
                if np.asarray(z["X_seq"]).ndim != 2 or np.asarray(z["X_seq"]).shape[1] != SEQ_DIM:
                    raise ValueError(f"wrong X_seq shape: {record_id}")
                if np.asarray(z["X_ecg"]).ndim != 2 or np.asarray(z["X_ecg"]).shape[1] != ECG_DIM:
                    raise ValueError(f"wrong X_ecg shape: {record_id}")
            records.append(RecordInfo(path=path, record_id=record_id, site=site, y=y))
    observed = {site: sum(record.site == site for record in records) for site in EXPECTED_SITE_COUNTS}
    if len(records) != 6600 or observed != EXPECTED_SITE_COUNTS:
        raise ValueError(f"unexpected DATA2 inventory: total={len(records)}, sites={observed}")
    return records


def _coherence_pair_valid(z: np.lib.npyio.NpzFile, length: int) -> np.ndarray:
    available = np.asarray(z["eeg_channel_available"], dtype=bool)
    common_clean = np.asarray(z["eeg_common_clean_subsegment_count"], dtype=np.int32)
    if available.shape != (6,) or common_clean.shape != (length,):
        raise ValueError("invalid coherence metadata shape")
    endpoints = np.array(
        [available[a] and available[b] for a, b in EEG_PAIRS], dtype=bool
    )
    return (common_clean[:, None] > 0) & endpoints[None, :]


def load_record(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as z:
        X_seq = np.asarray(z["X_seq"], dtype=np.float32)
        X_ecg = np.asarray(z["X_ecg"], dtype=np.float32)
        if X_seq.ndim != 2 or X_seq.shape[1] != SEQ_DIM:
            raise ValueError(f"invalid X_seq shape in {path}")
        if X_ecg.ndim != 2 or X_ecg.shape[1] != ECG_DIM:
            raise ValueError(f"invalid X_ecg shape in {path}")
        T = len(X_seq)
        M = len(X_ecg)

        eeg_available = np.asarray(z["eeg_channel_available"], dtype=bool)
        clean = np.asarray(z["eeg_clean_subsegment_count"], dtype=np.int32)
        if eeg_available.shape != (6,) or clean.shape != (T, 6):
            raise ValueError(f"invalid EEG metadata shape in {path}")
        spectral = X_seq[:, SPECTRAL].reshape(T, 6, 9).reshape(T, 54)
        spectral_channel_valid = eeg_available[None, :] & (clean > 0)
        spectral_valid = np.repeat(spectral_channel_valid, 9, axis=1)

        coherence = X_seq[:, COHERENCE].reshape(T, 15, 24)
        coherence_pair_valid = _coherence_pair_valid(z, T)

        bsr = X_seq[:, BSR]
        bsr_valid = np.broadcast_to(np.tile(eeg_available, 3), (T, 18)).copy()

        emg = X_seq[:, EMG]
        emg_available = np.asarray(z["emg_channel_available"], dtype=bool)
        emg_preprocess = np.asarray(z["emg_preprocessing_success"], dtype=bool)
        emg_epoch = np.asarray(z["emg_epoch_success"], dtype=bool)
        if emg_available.shape != (3,) or emg_preprocess.shape != (3,) or emg_epoch.shape != (T, 3):
            raise ValueError(f"invalid EMG metadata shape in {path}")
        emg_channel_valid = emg_epoch & (emg_available & emg_preprocess)[None, :]
        emg_valid = np.repeat(emg_channel_valid, 8, axis=1)

        resp = X_seq[:, RESP]
        resp_valid = np.asarray(z["resp_feature_valid"], dtype=bool)
        if resp_valid.shape != (T, 14):
            raise ValueError(f"invalid Resp metadata shape in {path}")

        hrv = np.zeros((T, 11), dtype=np.float32)
        hrv_valid = np.zeros((T, 11), dtype=bool)
        alignment = np.asarray(z["ecg_alignment_valid"], dtype=bool)
        success = np.asarray(z["hrv_success"], dtype=bool)
        feature_valid = np.asarray(z["hrv_feature_valid"], dtype=bool)
        if alignment.shape != (T,) or success.shape != (M,) or feature_valid.shape != (M, 11):
            raise ValueError(f"invalid HRV metadata shape in {path}")
        rows = np.arange(M, dtype=np.int64)
        epochs = rows + 10
        keep = epochs < T
        rows = rows[keep]
        epochs = epochs[keep]
        if len(rows):
            hrv[epochs] = X_ecg[rows, :11]
            hrv_valid[epochs] = (
                alignment[epochs, None]
                & success[rows, None]
                & feature_valid[rows]
            )
            hrv[~hrv_valid] = 0.0

        return {
            "record_id": str(_scalar(z, "record_id")),
            "site": str(_scalar(z, "site_id")),
            "y": int(_scalar(z, "y")),
            "age": float(np.asarray(z["x_static"], dtype=np.float32)[0]),
            "spectral": spectral.copy(),
            "spectral_valid": spectral_valid,
            "coherence": coherence.copy(),
            "coherence_pair_valid": coherence_pair_valid,
            "bsr": bsr.copy(),
            "bsr_valid": bsr_valid,
            "emg": emg.copy(),
            "emg_valid": emg_valid,
            "resp": resp.copy(),
            "resp_valid": resp_valid.copy(),
            "hrv": hrv,
            "hrv_valid": hrv_valid,
        }


def outer_loso_split(records: Sequence[RecordInfo], holdout_site: str) -> tuple[list[RecordInfo], list[RecordInfo]]:
    if holdout_site not in EXPECTED_SITE_COUNTS:
        raise ValueError(f"unknown holdout site: {holdout_site}")
    train = [record for record in records if record.site != holdout_site]
    test = [record for record in records if record.site == holdout_site]
    if not train or not test:
        raise ValueError(f"empty outer split for {holdout_site}")
    return train, test


def inner_split_by_site(records: Sequence[RecordInfo], seed: int = 7) -> tuple[list[RecordInfo], list[RecordInfo]]:
    sites = sorted({record.site for record in records})
    inner_train: list[RecordInfo] = []
    inner_val: list[RecordInfo] = []
    for site in sites:
        site_records = [record for record in records if record.site == site]
        ordered = sorted(
            site_records,
            key=lambda record: (stable_uint64("inner", seed, site, record.record_id), record.record_id),
        )
        n_val = max(1, min(len(ordered) - 1, int(round(0.2 * len(ordered)))))
        inner_val.extend(ordered[:n_val])
        inner_train.extend(ordered[n_val:])
    return sorted(inner_train, key=lambda r: (r.site, r.record_id)), sorted(inner_val, key=lambda r: (r.site, r.record_id))


def fit_robust_scaler24(values: np.ndarray) -> RobustScaler24:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 24 or not len(values):
        raise ValueError(f"expected nonempty [N,24] scaler input, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite coherence vector in scaler input")
    q25, center, q75 = np.percentile(values, [25, 50, 75], axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return RobustScaler24(center=center.astype(np.float32), scale=scale.astype(np.float32))


def _record_valid_coherence(record: RecordInfo) -> tuple[np.ndarray, np.ndarray]:
    with np.load(record.path, allow_pickle=False) as z:
        X_seq = np.asarray(z["X_seq"], dtype=np.float32)
        coherence = X_seq[:, COHERENCE].reshape(len(X_seq), 15, 24)
        valid = _coherence_pair_valid(z, len(X_seq))
    return coherence.reshape(-1, 24), np.flatnonzero(valid.reshape(-1))


def _stable_choice(indices: np.ndarray, count: int, *seed_parts: object) -> np.ndarray:
    if count >= len(indices):
        return indices.copy()
    rng = np.random.default_rng(stable_uint64(*seed_parts))
    return np.sort(rng.choice(indices, size=count, replace=False))


def sample_training_pairs(
    records: Sequence[RecordInfo], target_per_site: int = 200_000, seed: int = 7,
) -> tuple[np.ndarray, dict[str, int]]:
    sites = sorted({record.site for record in records})
    site_arrays: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for site in sites:
        site_records = sorted(
            (record for record in records if record.site == site),
            key=lambda record: (stable_uint64("train-order", seed, site, record.record_id), record.record_id),
        )
        selected: list[np.ndarray] = []
        remaining = target_per_site
        for position, record in enumerate(site_records):
            if not remaining:
                break
            vectors, valid_indices = _record_valid_coherence(record)
            records_left = len(site_records) - position
            quota = math.ceil(remaining / records_left) if remaining else 0
            take = min(quota, len(valid_indices))
            if take:
                chosen = _stable_choice(valid_indices, take, "train", seed, site, record.record_id)
                selected.append(vectors[chosen])
                remaining -= take
        if remaining:
            raise RuntimeError(
                f"site {site} supplied only {target_per_site - remaining:,} of "
                f"{target_per_site:,} requested valid coherence pairs"
            )
        site_values = np.concatenate(selected).astype(np.float32, copy=False)
        site_arrays.append(site_values)
        counts[site] = len(site_values)
    return np.concatenate(site_arrays).astype(np.float32, copy=False), counts


def sample_validation_pairs(
    records: Sequence[RecordInfo], max_per_record: int = 128, seed: int = 7,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    result: dict[str, list[tuple[str, np.ndarray]]] = {
        site: [] for site in sorted({record.site for record in records})
    }
    for record in sorted(records, key=lambda r: (r.site, r.record_id)):
        vectors, valid_indices = _record_valid_coherence(record)
        chosen = _stable_choice(
            valid_indices, min(max_per_record, len(valid_indices)),
            "validation", seed, record.site, record.record_id,
        )
        if len(chosen):
            result[record.site].append((record.record_id, vectors[chosen].astype(np.float32, copy=False)))
    for site, samples in result.items():
        if not samples:
            raise RuntimeError(f"no valid validation coherence pairs for site {site}")
    return result
