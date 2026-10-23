"""Frozen CORE_COH_EMG feature selection from the V5 manifest.

This module only selects and describes columns already present in the frozen
V5 Rank256 view. It never recomputes ranks or reads raw signals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .v5_family_loo import V5Manifest, V5ManifestRow, _canonical, _sha

CORE_CONFIG_ID = "CORE_COH_EMG"
CORE_KEEP_BLOCKS = ("old807", "eeg_coherence_stage", "emg_stage")
CORE_REMOVE_BLOCKS = ("caisr_stage", "resp_stage")
CORE_BLOCK_COUNTS = {
    "old807": 123,
    "eeg_coherence_stage": 90,
    "emg_stage": 72,
}
CORE_REMOVED_COUNTS = {"caisr_stage": 35, "resp_stage": 56}
CORE_WIDTH = 285


def select_core_config(manifest: V5Manifest, config_id: str = CORE_CONFIG_ID) -> tuple[V5ManifestRow, ...]:
    if config_id != CORE_CONFIG_ID:
        raise ValueError(f"unknown CORE configuration: {config_id}")
    selected = tuple(row for row in manifest.v5_rows if row.source_block in CORE_KEEP_BLOCKS)
    counts = {block: sum(row.source_block == block for row in selected) for block in CORE_KEEP_BLOCKS}
    if counts != CORE_BLOCK_COUNTS or len(selected) != CORE_WIDTH:
        raise ValueError(f"CORE source-block counts mismatch: {counts}")
    if [row.view_position for row in selected] != sorted(row.view_position for row in selected):
        raise ValueError("CORE feature order is not stable")
    if len({row.feature_id for row in selected}) != CORE_WIDTH:
        raise ValueError("CORE feature IDs are not unique")
    if len({row.unique_column_index for row in selected}) != CORE_WIDTH:
        raise ValueError("CORE unique-column identity is not unique")
    return selected


def load_core_manifest_payload(manifest: V5Manifest) -> dict[str, object]:
    rows = select_core_config(manifest)
    return {
        "schema": "featurelab_v2_core_coh_emg_manifest_v1",
        "config_id": CORE_CONFIG_ID,
        "view_id": "V5",
        "feature_count": CORE_WIDTH,
        "ordered_feature_ids": [row.feature_id for row in rows],
        "feature_ids_sha256": _sha([row.feature_id for row in rows]),
        "unique_column_indices": [row.unique_column_index for row in rows],
        "source_columns_sha256": _sha([row.unique_column_index for row in rows]),
        "source_manifest_sha256": manifest.source_sha256,
        "retained_family_counts": dict(CORE_BLOCK_COUNTS),
        "removed_family_counts": dict(CORE_REMOVED_COUNTS),
    }


def write_core_manifest(path: Path, manifest: V5Manifest) -> dict[str, object]:
    payload = load_core_manifest_payload(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload))
    return payload


__all__ = [
    "CORE_CONFIG_ID", "CORE_KEEP_BLOCKS", "CORE_REMOVE_BLOCKS",
    "CORE_BLOCK_COUNTS", "CORE_REMOVED_COUNTS", "CORE_WIDTH",
    "select_core_config", "load_core_manifest_payload", "write_core_manifest",
]
