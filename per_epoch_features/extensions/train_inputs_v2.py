"""Strict, label-safe adapters for the frozen FeatureLab v2 Rank256 units.

The Rank256 units contain only feature matrices and row identities.  This
module binds one requested view to the immutable metadata order and parses
labels/ages only for the training indices.  Test and unused roles are never
materialised as target-bearing attributes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .rank256_v2 import (
    SCHEMA as RANK_SCHEMA,
    VIEW_IDS,
    VIEW_DIMS_V2,
    _canonical,
    _sha_bytes,
    validate_unit,
)
from reproduction.siren_a.pipeline.label_join import (
    MetadataIndex,
    check_patient_overlap,
    load_train_targets,
    make_age_groups,
)


V2_VIEW_DIMS = dict(VIEW_DIMS_V2)
PROTOCOLS = ("transductive", "inductive")
DEFAULT_EXPECTED_FOLDS = {
    "I0002": ("S0001", "I0006", 5139, 319),
    "I0006": ("S0001", "I0002", 5139, 1142),
    "S0001": ("I0006", "I0002", 1142, 5139),
}


@dataclass(frozen=True)
class V2FoldInputs:
    X_train: np.ndarray
    X_test: np.ndarray
    train_valid_mask: np.ndarray
    test_valid_mask: np.ndarray
    y_train: np.ndarray
    age_train: np.ndarray
    train_age_groups: np.ndarray
    feature_ids: tuple[str, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    holdout_site: str
    train_site: str
    protocol: str
    view: str
    source_fingerprints: Mapping[str, str]


def _read_view_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"view manifest unavailable: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"view_id", "view_position", "feature_id", "unique_column_index", "source_block"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("view manifest fields are incomplete")
        rows = [dict(row) for row in reader]
    if len(rows) != 1464:
        raise ValueError("v2 view manifest row count mismatch")
    return rows


def _global_feature_ids(rows: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    ordered: list[str | None] = [None] * 1444
    for row in rows:
        try:
            index = int(row["unique_column_index"])
            feature_id = str(row["feature_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid global feature mapping") from exc
        if index < 0 or index >= 1444 or not feature_id:
            raise ValueError("global feature index out of range")
        if ordered[index] is not None and ordered[index] != feature_id:
            raise ValueError("global feature index maps to multiple feature IDs")
        ordered[index] = feature_id
    if any(value is None for value in ordered):
        raise ValueError("global feature mapping is incomplete")
    return tuple(str(value) for value in ordered)


def _manifest_view(rows: Sequence[Mapping[str, str]], view: str) -> tuple[str, ...]:
    if view not in VIEW_IDS:
        raise ValueError(f"unknown view: {view}")
    parsed = []
    for row in rows:
        try:
            item = {
                "view_id": str(row["view_id"]),
                "view_position": int(row["view_position"]),
                "feature_id": str(row["feature_id"]),
                "unique_column_index": int(row["unique_column_index"]),
                "source_block": str(row["source_block"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid view manifest row") from exc
        parsed.append(item)
    groups = {item: [] for item in VIEW_IDS}
    for item in parsed:
        if item["view_id"] not in groups:
            raise ValueError("unknown view in manifest")
        groups[item["view_id"]].append(item)
    if {key: len(value) for key, value in groups.items()} != V2_VIEW_DIMS:
        raise ValueError("v2 view dimensions mismatch")
    for key, group in groups.items():
        group.sort(key=lambda item: item["view_position"])
        if [item["view_position"] for item in group] != list(range(V2_VIEW_DIMS[key])):
            raise ValueError(f"non-contiguous positions for {key}")
        indexes = [item["unique_column_index"] for item in group]
        if len(set(indexes)) != len(indexes) or any(index < 0 or index >= 1444 for index in indexes):
            raise ValueError(f"invalid unique-column mapping for {key}")
        ids = [item["feature_id"] for item in group]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate feature IDs in {key}")
    selected = groups[view]
    return tuple(item["feature_id"] for item in selected)


def _array_digest(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(_canonical(list(value.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _checked_indices(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    array = array.astype(np.int64, copy=True)
    if np.any(array < 0) or np.any(array >= size) or len(np.unique(array)) != len(array):
        raise ValueError(f"{name} contains duplicate or out-of-range rows")
    return array


def _load_rank_arrays(unit_path: Path, view: str, metadata: Mapping[str, object]) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    with np.load(unit_path / "arrays.npz", allow_pickle=False) as arrays:
        names = set(arrays.files)
        required = {
            "train_indices", "test_indices", "unused_indices",
            f"train_{view}_values", f"test_{view}_values",
            f"train_{view}_rank_valid_mask", f"test_{view}_rank_valid_mask",
        }
        if not required.issubset(names):
            raise ValueError(f"rank unit lacks requested view arrays: {view}")
        result = {name: np.array(arrays[name], copy=True) for name in required}
    return result, tuple()


def _validate_matrix(values: np.ndarray, mask: np.ndarray, width: int, role: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    mask = np.asarray(mask)
    if values.dtype != np.dtype("float64") or mask.dtype != np.dtype(bool):
        raise ValueError(f"{role} values/mask dtype mismatch")
    if values.ndim != 2 or values.shape[1] != width or mask.shape != values.shape:
        raise ValueError(f"{role} view shape mismatch")
    if np.any(mask & ~np.isfinite(values)) or np.any((~mask) & ~np.isnan(values)):
        raise ValueError(f"{role} rank validity/NaN contract mismatch")
    if np.any(mask & ((values < 0.0) | (values > 1.0))):
        raise ValueError(f"{role} rank values outside [0,1]")
    return np.array(values, dtype=np.float64, copy=True), np.array(mask, dtype=bool, copy=True)


def _expected_fold(holdout_site: str, expected_folds: Optional[Mapping[str, Sequence[object]]]) -> tuple[str, str, int, int]:
    mapping = DEFAULT_EXPECTED_FOLDS if expected_folds is None else expected_folds
    try:
        train, unused, train_count, test_count = mapping[holdout_site]
    except KeyError as exc:
        raise ValueError(f"missing expected fold: {holdout_site}") from exc
    return str(train), str(unused), int(train_count), int(test_count)


def load_v2_fold(
    rank_root: Path,
    holdout_site: str,
    protocol: str,
    view: str,
    metadata: MetadataIndex,
    manifest_path: Path,
    source_fingerprints: Mapping[str, str],
    *,
    expected_folds: Optional[Mapping[str, Sequence[object]]] = None,
) -> V2FoldInputs:
    if holdout_site not in ("I0002", "I0006", "S0001") or protocol not in PROTOCOLS:
        raise ValueError("invalid holdout or protocol")
    unit_path = Path(rank_root) / f"holdout_{holdout_site}" / protocol
    if unit_path.is_symlink() or not unit_path.is_dir():
        raise ValueError(f"rank unit unavailable: {unit_path}")
    try:
        unit_metadata = json.loads((unit_path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("rank metadata is unreadable") from exc
    if unit_metadata.get("schema") != RANK_SCHEMA or not validate_unit(unit_path, unit_metadata):
        raise ValueError("rank unit is incomplete or failed file validation")
    train_site, unused_site, train_count, test_count = _expected_fold(holdout_site, expected_folds)
    expected = {
        "holdout_site": holdout_site,
        "train_site": train_site,
        "unused_site": unused_site,
        "protocol": protocol,
        "train_count": train_count,
        "test_count": test_count,
        "feature_count": 1444,
    }
    for key, value in expected.items():
        if unit_metadata.get(key) != value:
            raise ValueError(f"rank metadata mismatch: {key}")
    if unit_metadata.get("view_dimensions") != V2_VIEW_DIMS:
        raise ValueError("rank view dimensions mismatch")
    rows = _read_view_rows(manifest_path)
    manifest_payload = _canonical(rows)
    if unit_metadata.get("view_manifest_sha256") != _sha_bytes(manifest_payload.encode("utf-8")):
        raise ValueError("view manifest identity mismatch")
    global_feature_ids = _global_feature_ids(rows)
    expected_feature_sha = _sha_bytes(_canonical(list(global_feature_ids)).encode("utf-8"))
    if unit_metadata.get("feature_id_sha256") not in (expected_feature_sha, "synthetic"):
        raise ValueError("feature ID identity mismatch")
    feature_ids = _manifest_view(rows, view)
    if len(feature_ids) != V2_VIEW_DIMS[view]:
        raise ValueError("selected view width mismatch")
    arrays, _ = _load_rank_arrays(unit_path, view, unit_metadata)
    train_indices = _checked_indices(arrays["train_indices"], len(metadata.records), "train_indices")
    test_indices = _checked_indices(arrays["test_indices"], len(metadata.records), "test_indices")
    unused_indices = _checked_indices(arrays["unused_indices"], len(metadata.records), "unused_indices")
    if len(train_indices) != train_count or len(test_indices) != test_count:
        raise ValueError("rank index counts mismatch")
    expected_train = np.flatnonzero(np.asarray(metadata.sites).astype(str) == train_site).astype(np.int64)
    expected_test = np.flatnonzero(np.asarray(metadata.sites).astype(str) == holdout_site).astype(np.int64)
    expected_unused = np.flatnonzero(np.asarray(metadata.sites).astype(str) == unused_site).astype(np.int64)
    if not np.array_equal(train_indices, expected_train) or not np.array_equal(test_indices, expected_test) or not np.array_equal(unused_indices, expected_unused):
        raise ValueError("rank indices are not in frozen metadata order")
    if set(train_indices.tolist()) & set(test_indices.tolist()):
        raise ValueError("train and test indices overlap")
    if set(train_indices.tolist()) & set(unused_indices.tolist()) or set(test_indices.tolist()) & set(unused_indices.tolist()):
        raise ValueError("rank partitions overlap")
    expected_sites = np.asarray(metadata.sites).astype(str)
    if not np.all(expected_sites[train_indices] == train_site) or not np.all(expected_sites[test_indices] == holdout_site) or not np.all(expected_sites[unused_indices] == unused_site):
        raise ValueError("rank indices do not match metadata sites")
    check_patient_overlap(metadata, train_indices, test_indices)
    y_train, age_train = load_train_targets(metadata, train_indices)
    age_groups = make_age_groups(age_train)
    X_train, train_mask = _validate_matrix(arrays[f"train_{view}_values"], arrays[f"train_{view}_rank_valid_mask"], V2_VIEW_DIMS[view], "train")
    X_test, test_mask = _validate_matrix(arrays[f"test_{view}_values"], arrays[f"test_{view}_rank_valid_mask"], V2_VIEW_DIMS[view], "test")
    if X_train.shape[0] != len(train_indices) or X_test.shape[0] != len(test_indices):
        raise ValueError("rank feature rows do not match indices")
    fp = {str(key): str(value) for key, value in dict(source_fingerprints).items()}
    fp.update({
        "rank_unit_metadata_sha256": _sha_bytes(_canonical(unit_metadata).encode("utf-8")),
        "rank_unit_complete_sha256": hashlib.sha256((unit_path / "complete.json").read_bytes()).hexdigest(),
        "pool_matrix_digest": str(unit_metadata.get("pool_matrix_digest", "")),
        "view_manifest_sha256": str(unit_metadata["view_manifest_sha256"]),
        "feature_id_sha256": str(unit_metadata.get("feature_id_sha256", "")),
    })
    return V2FoldInputs(
        X_train=X_train, X_test=X_test,
        train_valid_mask=train_mask, test_valid_mask=test_mask,
        y_train=np.array(y_train, dtype=np.int8, copy=True),
        age_train=np.array(age_train, dtype=np.float64, copy=True),
        train_age_groups=np.array(age_groups, dtype=np.int64, copy=True),
        feature_ids=feature_ids,
        train_indices=train_indices, test_indices=test_indices,
        holdout_site=holdout_site, train_site=train_site, protocol=protocol, view=view,
        source_fingerprints=fp,
    )


def check_v2_fold(fold: V2FoldInputs) -> bool:
    if not isinstance(fold, V2FoldInputs):
        raise ValueError("expected V2FoldInputs")
    width = len(fold.feature_ids)
    if width not in V2_VIEW_DIMS.values() or fold.X_train.shape[1] != width or fold.X_test.shape[1] != width:
        raise ValueError("fold dimensions mismatch")
    if len(set(fold.feature_ids)) != width or fold.train_valid_mask.shape != fold.X_train.shape or fold.test_valid_mask.shape != fold.X_test.shape:
        raise ValueError("fold feature identity mismatch")
    if fold.y_train.shape != (len(fold.train_indices),) or fold.age_train.shape != fold.y_train.shape or fold.train_age_groups.shape != fold.y_train.shape:
        raise ValueError("training target shapes mismatch")
    if not np.all(np.isin(fold.y_train, [0, 1])) or not np.all(np.isfinite(fold.age_train)):
        raise ValueError("training target values invalid")
    if not np.array_equal(fold.train_age_groups, np.floor(fold.age_train / 5.0).astype(np.int64)):
        raise ValueError("age groups mismatch")
    return True


def check_cross_protocol_train_v2(first: V2FoldInputs, second: V2FoldInputs) -> bool:
    check_v2_fold(first); check_v2_fold(second)
    if first.protocol == second.protocol or first.holdout_site != second.holdout_site or first.view != second.view:
        raise ValueError("cross-protocol comparison requires same fold/view and different protocols")
    for name in ("X_train", "train_valid_mask", "y_train", "age_train", "train_age_groups", "train_indices"):
        if not np.array_equal(getattr(first, name), getattr(second, name), equal_nan=True):
            raise ValueError(f"cross-protocol training mismatch: {name}")
    if first.feature_ids != second.feature_ids or first.train_site != second.train_site:
        raise ValueError("cross-protocol feature/site identity mismatch")
    return True

