"""Exact Feature ID subset contract for CORE_NO_CAISR20.

This module only projects the frozen CORE V5 Rank256 matrix; it never recomputes
Rank256 values or reads raw signals. Training/prediction runners consume the
returned ordered rows and reuse the existing V5 model contract.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .caisr20_core_protocol import CAISR20_FEATURE_IDS
from .core_coh_emg import CORE_WIDTH, select_core_config
from .v5_family_loo import V5Manifest, V5ManifestRow, _canonical, _sha

ROUTE_ID = "CORE_NO_CAISR20"
V5_OLD_DIM = CORE_WIDTH
V5_NEW_DIM = 265
EXPECTED_FAMILY_COUNTS = {"HRV": 22, "EMG": 48, "Resp": 28, "Demo": 5, "Coherence": 90, "StageEMG": 72}


class CoreNoCAISRStop(ValueError):
    """Frozen CORE/CAISR subset identity or shape failed."""


def load_caisr20_feature_ids(path: Path) -> tuple[str, ...]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "feature_name" not in (reader.fieldnames or ()): 
            raise CoreNoCAISRStop("CAISR20 manifest lacks feature_name")
        ids = tuple(str(row["feature_name"]) for row in reader)
    if ids != tuple(CAISR20_FEATURE_IDS) or len(set(ids)) != 20:
        raise CoreNoCAISRStop("CAISR20 canonical feature order mismatch")
    return ids


def build_no_caisr20_rows(core_rows: Sequence[V5ManifestRow], caisr_ids: Sequence[str] = CAISR20_FEATURE_IDS) -> tuple[V5ManifestRow, ...]:
    rows = tuple(core_rows); ids = tuple(str(item) for item in caisr_ids)
    if len(rows) != V5_OLD_DIM or len(set(row.feature_id for row in rows)) != V5_OLD_DIM:
        raise CoreNoCAISRStop("CORE V5 row contract must contain 285 unique features")
    if ids != tuple(CAISR20_FEATURE_IDS) or len(set(ids)) != 20:
        raise CoreNoCAISRStop("CAISR20 deletion set is not the frozen 20 IDs")
    prefix = tuple(row.feature_id for row in rows[:20])
    if prefix != ids:
        raise CoreNoCAISRStop("CORE V5 first 20 IDs are not the canonical CAISR20 prefix")
    removed = tuple(row for row in rows if row.feature_id in set(ids))
    if tuple(row.feature_id for row in removed) != ids:
        raise CoreNoCAISRStop("CORE V5 CAISR20 IDs are not in exact canonical order")
    retained = tuple(row for row in rows if row.feature_id not in set(ids))
    if len(retained) != V5_NEW_DIM or any(a.view_position >= b.view_position for a, b in zip(retained, retained[1:])):
        raise CoreNoCAISRStop("retained CORE V5 order/width mismatch")
    return retained


def project_rank_subset(values: np.ndarray, retained_positions: Sequence[int]) -> np.ndarray:
    array = np.asarray(values)
    positions = np.asarray(retained_positions, dtype=np.int64)
    if array.ndim != 2 or positions.ndim != 1 or len(positions) != V5_NEW_DIM or len(np.unique(positions)) != V5_NEW_DIM or np.any(positions < 0) or np.any(positions >= V5_OLD_DIM):
        raise CoreNoCAISRStop("rank subset position contract mismatch")
    return np.array(array[:, positions], dtype=np.float64, copy=True)


def feature_id_disjoint(left: Iterable[str], right: Iterable[str]) -> bool:
    return set(map(str, left)).isdisjoint(set(map(str, right)))


def _family(row: V5ManifestRow) -> str:
    fid = str(row.feature_id)
    if row.source_block == "eeg_coherence_stage": return "Coherence"
    if row.source_block == "emg_stage": return "StageEMG"
    if fid.startswith("hrv_model_"): return "HRV"
    if fid.startswith("emg_"): return "EMG"
    if fid.startswith("resp_"): return "Resp"
    if fid.startswith("demo_"): return "Demo"
    return "other"


def family_counts(rows: Sequence[V5ManifestRow]) -> dict[str, int]:
    counts = {key: 0 for key in EXPECTED_FAMILY_COUNTS}
    counts["other"] = 0
    for row in rows: counts[_family(row)] += 1
    return counts


def build_manifest_payload(core_rows: Sequence[V5ManifestRow], retained_rows: Sequence[V5ManifestRow], caisr_ids: Sequence[str] = CAISR20_FEATURE_IDS) -> dict[str, object]:
    core = tuple(core_rows); retained = tuple(retained_rows); ids = tuple(map(str, caisr_ids))
    if len(retained) != V5_NEW_DIM or not feature_id_disjoint([r.feature_id for r in retained], ids):
        raise CoreNoCAISRStop("invalid retained manifest")
    counts = family_counts(retained)
    expected = dict(EXPECTED_FAMILY_COUNTS); expected["other"] = 0
    if counts != expected:
        raise CoreNoCAISRStop(f"retained family counts mismatch: {counts}")
    return {"schema":"featurelab_core_no_caisr20_manifest_v1","route_id":ROUTE_ID,"view_id":"V5","old_v5_feature_count":V5_OLD_DIM,"new_v5_feature_count":V5_NEW_DIM,"removed_feature_count":20,"removed_feature_ids":list(ids),"retained_feature_ids":[r.feature_id for r in retained],"retained_view_positions":[int(r.view_position) for r in retained],"retained_unique_column_indices":[int(r.unique_column_index) for r in retained],"retained_feature_ids_sha256":_sha([r.feature_id for r in retained]),"retained_source_columns_sha256":_sha([r.unique_column_index for r in retained]),"family_counts":{k:int(v) for k,v in counts.items() if k != "other"},"caisr20_remaining_in_v5":0,"source_core_feature_ids_sha256":_sha([r.feature_id for r in core]),"rank256_rebuild_count":0,"standalone_caisr_feature_id_disjoint":True}


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(_canonical(payload))


def load_core_rows_from_manifest(manifest: V5Manifest) -> tuple[V5ManifestRow, ...]:
    return select_core_config(manifest)

__all__ = ["CAISR20_FEATURE_IDS", "CoreNoCAISRStop", "EXPECTED_FAMILY_COUNTS", "ROUTE_ID", "V5_NEW_DIM", "V5_OLD_DIM", "build_manifest_payload", "build_no_caisr20_rows", "family_counts", "feature_id_disjoint", "load_caisr20_feature_ids", "load_core_rows_from_manifest", "project_rank_subset", "write_manifest"]
