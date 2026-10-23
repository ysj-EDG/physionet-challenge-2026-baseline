"""Frozen V5 family leave-one-out contracts.

This module only selects columns from the already validated V5 Rank256 view.
It never rebuilds ranks or reads raw signals.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


CONFIGS = (
    "LOO_NO_COHERENCE",
    "LOO_NO_CAISR",
    "LOO_NO_EMG",
    "LOO_NO_RESP",
)
REMOVED_BLOCK = {
    "LOO_NO_COHERENCE": "eeg_coherence_stage",
    "LOO_NO_CAISR": "caisr_stage",
    "LOO_NO_EMG": "emg_stage",
    "LOO_NO_RESP": "resp_stage",
}
EXPECTED_WIDTH = {
    "LOO_NO_COHERENCE": 286,
    "LOO_NO_CAISR": 341,
    "LOO_NO_EMG": 304,
    "LOO_NO_RESP": 320,
}
EXPECTED_BLOCK_WIDTH = {
    "old807": 123,
    "eeg_coherence_stage": 90,
    "caisr_stage": 35,
    "emg_stage": 72,
    "resp_stage": 56,
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class V5ManifestRow:
    view_position: int
    feature_id: str
    unique_column_index: int
    source_block: str


@dataclass(frozen=True)
class V5Manifest:
    source_path: str
    source_sha256: str
    v5_rows: tuple[V5ManifestRow, ...]

    def columns(self, config_id: str) -> tuple[V5ManifestRow, ...]:
        return select_config(self, config_id)


def load_v5_manifest(path: Path) -> V5Manifest:
    path = Path(path)
    try:
        raw = path.read_bytes()
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"view_id", "view_position", "feature_id", "unique_column_index", "source_block"}
            if not required.issubset(set(reader.fieldnames or ())):
                raise ValueError("view manifest fields incomplete")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise ValueError(f"unable to read view manifest: {path}") from exc
    v5 = []
    for row in rows:
        if row.get("view_id") != "V5":
            continue
        try:
            parsed = V5ManifestRow(
                view_position=int(row["view_position"]),
                feature_id=str(row["feature_id"]),
                unique_column_index=int(row["unique_column_index"]),
                source_block=str(row["source_block"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid V5 manifest row") from exc
        if not parsed.feature_id or parsed.unique_column_index < 0:
            raise ValueError("invalid V5 manifest identity")
        v5.append(parsed)
    v5.sort(key=lambda item: item.view_position)
    if len(rows) != 1464 or len(v5) != 376:
        raise ValueError("frozen manifest size mismatch")
    if [row.view_position for row in v5] != list(range(376)):
        raise ValueError("V5 positions are not contiguous")
    if len({row.feature_id for row in v5}) != 376 or len({row.unique_column_index for row in v5}) != 376:
        raise ValueError("V5 feature identity is not unique")
    blocks = {block: sum(row.source_block == block for row in v5) for block in EXPECTED_BLOCK_WIDTH}
    if blocks != EXPECTED_BLOCK_WIDTH:
        raise ValueError(f"V5 source-block counts mismatch: {blocks}")
    if any(row.unique_column_index >= 1444 for row in v5):
        raise ValueError("V5 unique-column index out of range")
    return V5Manifest(str(path.resolve()), hashlib.sha256(raw).hexdigest(), tuple(v5))


def select_config(manifest: V5Manifest, config_id: str) -> tuple[V5ManifestRow, ...]:
    if config_id not in CONFIGS:
        raise ValueError(f"unknown LOO configuration: {config_id}")
    removed = REMOVED_BLOCK[config_id]
    selected = tuple(row for row in manifest.v5_rows if row.source_block != removed)
    if len(selected) != EXPECTED_WIDTH[config_id]:
        raise ValueError("LOO width mismatch")
    if [row.view_position for row in selected] != sorted(row.view_position for row in selected):
        raise ValueError("V5 order is not stable")
    return selected


def project_matrix(values: np.ndarray, valid_mask: np.ndarray,
                   rows: Sequence[V5ManifestRow]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    mask = np.asarray(valid_mask)
    if values.ndim != 2 or mask.shape != values.shape or values.dtype != np.float64 or mask.dtype != bool:
        raise ValueError("Rank256 matrix contract mismatch")
    indices = np.asarray([row.view_position for row in rows], dtype=np.int64)
    if len(indices) != len(set(indices.tolist())) or np.any(indices < 0) or np.any(indices >= values.shape[1]):
        raise ValueError("invalid V5 projection positions")
    projected_values = np.array(values[:, indices], dtype=np.float64, copy=True)
    projected_mask = np.array(mask[:, indices], dtype=bool, copy=True)
    if np.any(projected_mask & ~np.isfinite(projected_values)) or np.any((~projected_mask) & ~np.isnan(projected_values)):
        raise ValueError("Rank256 validity/NaN contract mismatch")
    return projected_values, projected_mask


def config_manifest(manifest: V5Manifest, config_ids: Iterable[str] = CONFIGS) -> dict[str, dict[str, object]]:
    ids = tuple(sorted(set(config_ids)))
    if ids != tuple(sorted(CONFIGS)) and any(item not in CONFIGS for item in ids):
        raise ValueError("unknown configuration")
    out: dict[str, dict[str, object]] = {}
    for config_id in ids:
        rows = select_config(manifest, config_id)
        out[config_id] = {
            "config_id": config_id,
            "removed_family": REMOVED_BLOCK[config_id],
            "view_id": "V5",
            "feature_count": len(rows),
            "feature_ids": [row.feature_id for row in rows],
            "unique_column_indices": [row.unique_column_index for row in rows],
            "feature_ids_sha256": _sha([row.feature_id for row in rows]),
            "source_columns_sha256": _sha([row.unique_column_index for row in rows]),
        }
    return out


def write_public_manifest(path: Path, manifest: V5Manifest) -> dict[str, object]:
    payload = {
        "schema": "featurelab_v2_v5_family_loo_manifest_v1",
        "source_manifest_sha256": manifest.source_sha256,
        "view_id": "V5",
        "full_feature_count": 376,
        "old_v5_prefix_count": 123,
        "configs": config_manifest(manifest),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload))
    return payload
