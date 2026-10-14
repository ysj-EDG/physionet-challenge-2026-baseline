"""Read-only G3-A five-view inputs with train-only targets."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .label_join import MetadataIndex, check_patient_overlap, load_train_targets, make_age_groups


EXPECTED_FOLDS = {
    "I0002": ("S0001", "I0006", 5139, 319),
    "I0006": ("S0001", "I0002", 5139, 1142),
    "S0001": ("I0006", "I0002", 1142, 5139),
}
VIEW_SIZES = {"V1": 113, "V2": 129, "V3": 349, "V4": 113, "V5": 123}
PROTOCOLS = {"transductive", "inductive"}
FORBIDDEN_FEATURE_TOKENS = {
    "age", "cognitive_impairment", "time_to_event", "last_known_visit_date",
    "diagnosis", "siteid", "bdsppatientid", "bidsfolder", "sessionid",
}


@dataclass(frozen=True)
class FoldInputs:
    X_train: np.ndarray
    X_test: np.ndarray
    train_valid_mask: np.ndarray
    test_valid_mask: np.ndarray
    y_train: np.ndarray
    age_train: np.ndarray
    train_age_groups: np.ndarray
    feature_ids: Tuple[str, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    holdout_site: str
    train_site: str
    protocol: str
    view: str
    source_fingerprints: Mapping[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _verify_complete(unit: Path) -> Mapping[str, str]:
    if unit.is_symlink() or not unit.is_dir():
        raise ValueError("G3-A unit must be a normal directory")
    if any(path.name.endswith(".building") for path in unit.iterdir()):
        raise ValueError("incomplete G3-A build evidence present")
    complete_path = unit / "complete.json"
    if not complete_path.is_file() or complete_path.is_symlink():
        raise ValueError("G3-A completion marker unavailable")
    complete = _read_json(complete_path)
    if not isinstance(complete, dict) or complete.get("schema") != "g3a_rank256_v1":
        raise ValueError("G3-A completion schema mismatch")
    files = complete.get("files")
    if not isinstance(files, dict) or not {"indices.npz", "views.npz", "metadata.json"}.issubset(files):
        raise ValueError("G3-A completion file list incomplete")
    for name, expected in files.items():
        path = unit / name
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise ValueError(f"G3-A artifact hash mismatch: {name}")
    return files


def get_view_feature_ids(manifest_path: Path, view: str, *, expected_size: Optional[int] = None) -> Tuple[str, ...]:
    if view not in VIEW_SIZES:
        raise ValueError(f"unknown view: {view}")
    size = VIEW_SIZES[view] if expected_size is None else int(expected_size)
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"view_id", "view_position", "unique_feature_id", "feature_name"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Frozen Manifest fields missing")
        rows = [row for row in reader if row["view_id"] == view]
    try:
        rows.sort(key=lambda row: int(row["view_position"]))
        positions = [int(row["view_position"]) for row in rows]
    except ValueError as exc:
        raise ValueError("non-integer view position") from exc
    if positions != list(range(size)):
        raise ValueError(f"{view} positions are not contiguous 0..{size - 1}")
    ids = tuple(row["unique_feature_id"] for row in rows)
    if len(ids) != size or len(set(ids)) != size:
        raise ValueError(f"{view} feature IDs are missing or duplicated")
    for row in rows:
        candidates = {row["unique_feature_id"].strip().lower(), row["feature_name"].strip().lower()}
        if candidates & FORBIDDEN_FEATURE_TOKENS:
            raise ValueError("forbidden metadata field appears in model feature manifest")
    return ids


def _validate_array_pair(values: np.ndarray, mask: np.ndarray, expected_shape: Tuple[int, int], role: str) -> None:
    if values.shape != expected_shape or values.dtype != np.float64:
        raise ValueError(f"{role} values schema mismatch")
    if mask.shape != expected_shape or mask.dtype != np.bool_:
        raise ValueError(f"{role} mask schema mismatch")
    if np.isinf(values).any() or not np.isfinite(values[mask]).all() or not np.isnan(values[~mask]).all():
        raise ValueError(f"{role} values/mask semantics mismatch")
    if np.any(values[mask] < 0.0) or np.any(values[mask] > 1.0):
        raise ValueError(f"{role} Rank256 value outside [0,1]")


def load_fold(
    g3_root: Path,
    holdout_site: str,
    protocol: str,
    view: str,
    metadata: MetadataIndex,
    manifest_path: Path,
    source_fingerprints: Mapping[str, str],
    *,
    expected_folds: Optional[Mapping[str, Sequence[object]]] = None,
) -> FoldInputs:
    folds = EXPECTED_FOLDS if expected_folds is None else expected_folds
    if holdout_site not in folds or protocol not in PROTOCOLS or view not in VIEW_SIZES:
        raise ValueError("unknown fold, protocol, or view")
    train_site, unused_site, train_count, test_count = folds[holdout_site]
    unit = Path(g3_root) / f"holdout_{holdout_site}" / protocol
    _verify_complete(unit)
    unit_metadata = _read_json(unit / "metadata.json")
    expected_meta = {
        "schema": "g3a_rank256_v1", "holdout_site": holdout_site,
        "train_site": train_site, "unused_site": unused_site,
        "protocol": protocol, "train_count": int(train_count), "test_count": int(test_count),
    }
    if not isinstance(unit_metadata, dict) or any(unit_metadata.get(key) != value for key, value in expected_meta.items()):
        raise ValueError("G3-A unit metadata mismatch")

    with np.load(unit / "indices.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"train_indices", "test_indices", "unused_indices"}:
            raise ValueError("G3-A index keys mismatch")
        train_indices = np.asarray(archive["train_indices"])
        test_indices = np.asarray(archive["test_indices"])
        unused_indices = np.asarray(archive["unused_indices"])
    for name, array in (("train", train_indices), ("test", test_indices), ("unused", unused_indices)):
        if array.ndim != 1 or array.dtype != np.int64:
            raise ValueError(f"{name} indices schema mismatch")
    expected_train = np.flatnonzero(metadata.sites == train_site).astype(np.int64)
    expected_test = np.flatnonzero(metadata.sites == holdout_site).astype(np.int64)
    expected_unused = np.flatnonzero(metadata.sites == unused_site).astype(np.int64)
    if not np.array_equal(train_indices, expected_train) or not np.array_equal(test_indices, expected_test) or not np.array_equal(unused_indices, expected_unused):
        raise ValueError("G3-A indices differ from exact frozen metadata order")
    if len(train_indices) != train_count or len(test_indices) != test_count:
        raise ValueError("G3-A fold counts mismatch")
    if len(train_indices) + len(test_indices) + len(unused_indices) != len(metadata.records):
        raise ValueError("G3-A indices do not cover the metadata index")
    check_patient_overlap(metadata, train_indices, test_indices)

    feature_ids = get_view_feature_ids(manifest_path, view)
    with np.load(unit / "views.npz", allow_pickle=False) as archive:
        keys = (
            f"train_{view}_values", f"train_{view}_rank_valid_mask",
            f"test_{view}_values", f"test_{view}_rank_valid_mask",
        )
        if not set(keys).issubset(archive.files):
            raise ValueError(f"G3-A {view} arrays unavailable")
        X_train = np.asarray(archive[keys[0]])
        train_mask = np.asarray(archive[keys[1]])
        X_test = np.asarray(archive[keys[2]])
        test_mask = np.asarray(archive[keys[3]])
    feature_count = len(feature_ids)
    _validate_array_pair(X_train, train_mask, (int(train_count), feature_count), "train")
    _validate_array_pair(X_test, test_mask, (int(test_count), feature_count), "test")

    y_train, age_train = load_train_targets(metadata, train_indices)
    groups = make_age_groups(age_train)
    fingerprints = dict(source_fingerprints)
    fingerprints.update({
        "demographics_sha256": metadata.demographics_sha256,
        "metadata_join_sha256": metadata.join_fingerprint,
        "unit_metadata_sha256": _sha256(unit / "metadata.json"),
        "unit_complete_sha256": _sha256(unit / "complete.json"),
        "feature_ids_sha256": hashlib.sha256("\n".join(feature_ids).encode("utf-8")).hexdigest(),
    })
    result = FoldInputs(
        _readonly(X_train), _readonly(X_test), _readonly(train_mask), _readonly(test_mask),
        _readonly(y_train), _readonly(age_train), _readonly(groups), feature_ids,
        _readonly(train_indices), _readonly(test_indices), holdout_site, str(train_site), protocol, view,
        MappingProxyType(fingerprints),
    )
    check_fold_inputs(result)
    return result


def check_fold_inputs(fold: FoldInputs) -> bool:
    if fold.view not in VIEW_SIZES or fold.protocol not in PROTOCOLS:
        raise ValueError("FoldInputs identity invalid")
    train_count, feature_count = fold.X_train.shape
    test_count = fold.X_test.shape[0]
    if feature_count != VIEW_SIZES[fold.view] or fold.X_test.shape != (test_count, feature_count):
        raise ValueError("FoldInputs feature dimensions invalid")
    _validate_array_pair(fold.X_train, fold.train_valid_mask, fold.X_train.shape, "train")
    _validate_array_pair(fold.X_test, fold.test_valid_mask, fold.X_test.shape, "test")
    if fold.y_train.shape != (train_count,) or fold.y_train.dtype != np.int8 or not np.all(np.isin(fold.y_train, [0, 1])):
        raise ValueError("training labels invalid")
    if fold.age_train.shape != (train_count,) or fold.age_train.dtype != np.float64 or not np.isfinite(fold.age_train).all():
        raise ValueError("training ages invalid")
    if fold.train_age_groups.shape != (train_count,) or fold.train_age_groups.dtype != np.int64:
        raise ValueError("training age groups invalid")
    if not np.array_equal(fold.train_age_groups, np.floor(fold.age_train / 5.0).astype(np.int64)):
        raise ValueError("training age groups inconsistent")
    if len(fold.feature_ids) != feature_count or len(set(fold.feature_ids)) != feature_count:
        raise ValueError("feature IDs invalid")
    if fold.train_indices.shape != (train_count,) or fold.test_indices.shape != (test_count,):
        raise ValueError("fold indices invalid")
    return True


def check_cross_protocol_train(first: FoldInputs, second: FoldInputs) -> bool:
    identity = (first.holdout_site, first.train_site, first.view)
    if identity != (second.holdout_site, second.train_site, second.view):
        raise ValueError("cross-protocol fold identity mismatch")
    for name in ("X_train", "train_valid_mask", "y_train", "age_train", "train_age_groups", "train_indices"):
        if not np.array_equal(getattr(first, name), getattr(second, name), equal_nan=True):
            raise ValueError(f"cross-protocol training mismatch: {name}")
    if first.feature_ids != second.feature_ids:
        raise ValueError("cross-protocol feature order mismatch")
    return True
