"""Independent mathematical and artifact verification for G4-B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from reproduction.siren_a.g4.verify_g4a import sha256_file, verify_unit
from reproduction.siren_a.pipeline.adapters import NEUTRAL_RANK256


QUERY_KEYS = (
    "permutation",
    "inverse_permutation",
    "sorted_local_positions",
    "sorted_global_indices",
    "sorted_group_ids",
    "sorted_labels",
    "group_ids",
    "group_sizes",
    "group_positive",
    "group_negative",
)
ADAPTER_KEYS = (
    "model_type",
    "feature_ids",
    "feature_ids_sha256",
    "fit_indices_sha256",
    "fit_scope",
    "finite_counts",
    "fill_values",
    "all_missing",
)


def _canonical_sha(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _feature_sha(feature_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in feature_ids).encode("utf-8")).hexdigest()


def _indices_sha(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _scalar(saved: Mapping[str, np.ndarray], key: str) -> str:
    value = np.asarray(saved[key])
    if value.shape != ():
        raise ValueError(f"adapter scalar has wrong shape: {key}")
    return str(value.item())


def rebuild_query_expected(
    ages: np.ndarray,
    labels: np.ndarray,
    local_positions: np.ndarray,
    global_indices: np.ndarray,
) -> Mapping[str, object]:
    """Rebuild query semantics without calling the production query builder."""

    ages = np.asarray(ages)
    labels = np.asarray(labels)
    local = np.asarray(local_positions)
    global_idx = np.asarray(global_indices)
    if ages.ndim != 1 or ages.dtype != np.float64:
        raise ValueError("query ages must be one-dimensional float64")
    if labels.shape != ages.shape or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("query labels invalid")
    if local.shape != ages.shape or global_idx.shape != ages.shape:
        raise ValueError("query indices are not aligned")
    if local.dtype != np.int64 or global_idx.dtype != np.int64:
        raise ValueError("query indices must be int64")
    if not np.isfinite(ages).all() or len(np.unique(global_idx)) != len(global_idx):
        raise ValueError("query ages or global indices invalid")

    groups = np.floor(ages / 5.0).astype(np.int64)
    order = sorted(range(len(groups)), key=lambda index: int(groups[index]))
    permutation = np.asarray(order, dtype=np.int64)
    inverse = np.empty(len(order), dtype=np.int64)
    for output_position, source_position in enumerate(order):
        inverse[source_position] = output_position
    sorted_groups = groups[permutation]
    sorted_labels = labels[permutation].astype(np.int8, copy=False)

    group_ids_list = []
    group_sizes_list = []
    group_positive_list = []
    start = 0
    while start < len(sorted_groups):
        group = int(sorted_groups[start])
        end = start + 1
        while end < len(sorted_groups) and int(sorted_groups[end]) == group:
            end += 1
        group_labels = sorted_labels[start:end]
        positive = int(np.sum(group_labels == 1))
        negative = int(np.sum(group_labels == 0))
        if positive == 0 or negative == 0:
            raise ValueError("each independent query must contain both classes")
        group_ids_list.append(group)
        group_sizes_list.append(end - start)
        group_positive_list.append(positive)
        start = end

    group_ids = np.asarray(group_ids_list, dtype=np.int64)
    group_sizes = np.asarray(group_sizes_list, dtype=np.int64)
    group_positive = np.asarray(group_positive_list, dtype=np.int64)
    group_negative = group_sizes - group_positive
    arrays = {
        "permutation": permutation,
        "inverse_permutation": inverse,
        "sorted_local_positions": local[permutation].astype(np.int64, copy=False),
        "sorted_global_indices": global_idx[permutation].astype(np.int64, copy=False),
        "sorted_group_ids": sorted_groups.astype(np.int64, copy=False),
        "sorted_labels": sorted_labels,
        "group_ids": group_ids,
        "group_sizes": group_sizes,
        "group_positive": group_positive,
        "group_negative": group_negative,
    }
    fingerprint_payload = {
        "global": arrays["sorted_global_indices"].tolist(),
        "groups": group_ids.tolist(),
        "sizes": group_sizes.tolist(),
        "labels": sorted_labels.tolist(),
    }
    positive_positions = np.flatnonzero(labels == 1)
    negative_positions = np.flatnonzero(labels == 0)
    valid_pairs = np.abs(ages[positive_positions, None] - ages[negative_positions][None, :]) <= 2.0
    all_pairs = int(valid_pairs.sum())
    if all_pairs == 0:
        raise ValueError("no valid opposite-label age pairs")
    same_group = groups[positive_positions, None] == groups[negative_positions][None, :]
    within = int((valid_pairs & same_group).sum())
    maximum = int(group_sizes.max())
    audit = {
        "query_count": int(len(group_sizes)),
        "maximum_group_size": maximum,
        "truncation_level": maximum,
        "query_fingerprint": _canonical_sha(fingerprint_payload),
        "all_valid_pairs": all_pairs,
        "within_group_valid_pairs": within,
        "pair_coverage": within / all_pairs,
    }
    return {"arrays": arrays, "audit": audit}


def verify_query_arrays(
    saved_arrays: Mapping[str, np.ndarray],
    saved_audit: Mapping[str, object],
    ages: np.ndarray,
    labels: np.ndarray,
    local_positions: np.ndarray,
    global_indices: np.ndarray,
    *,
    sample_sha256: str,
) -> Mapping[str, object]:
    expected = rebuild_query_expected(ages, labels, local_positions, global_indices)
    if set(saved_arrays) != set(QUERY_KEYS):
        raise ValueError("query array keys mismatch")
    for key in QUERY_KEYS:
        observed = np.asarray(saved_arrays[key])
        wanted = expected["arrays"][key]
        if observed.dtype != wanted.dtype or observed.shape != wanted.shape:
            raise ValueError(f"query array schema mismatch: {key}")
        if not np.array_equal(observed, wanted):
            raise ValueError(f"query array value mismatch: {key}")
    wanted_audit = dict(expected["audit"])
    wanted_audit["sample_sha256"] = str(sample_sha256)
    for key, value in wanted_audit.items():
        if key == "pair_coverage":
            if not isinstance(saved_audit.get(key), (int, float)) or float(saved_audit[key]) != float(value):
                raise ValueError("query pair coverage mismatch")
        elif saved_audit.get(key) != value:
            raise ValueError(f"query audit mismatch: {key}")
    return expected


def load_npz_mapping(path: Path) -> Dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def verify_query_artifact(
    unit: Path,
    schema: str,
    identity: Mapping[str, object],
    ages: np.ndarray,
    labels: np.ndarray,
    local_positions: np.ndarray,
    global_indices: np.ndarray,
    *,
    sample_sha256: str,
) -> Mapping[str, object]:
    verify_unit(unit, schema, identity)
    arrays = load_npz_mapping(Path(unit) / "query.npz")
    audit = json.loads((Path(unit) / "query_audit.json").read_text(encoding="utf-8"))
    if "sample_sha256" not in audit:
        audit["sample_sha256"] = identity.get("sample_sha256")
    return verify_query_arrays(arrays, audit, ages, labels, local_positions, global_indices,
                               sample_sha256=sample_sha256)


def verify_adapter_arrays(
    saved: Mapping[str, np.ndarray],
    model_type: str,
    X_selected: np.ndarray,
    feature_ids: Sequence[str],
    selected_global_indices: np.ndarray,
) -> Mapping[str, np.ndarray]:
    if set(saved) != set(ADAPTER_KEYS):
        raise ValueError("adapter keys mismatch")
    values = np.asarray(X_selected)
    indices = np.asarray(selected_global_indices)
    ids = tuple(str(value) for value in feature_ids)
    if model_type not in {"lr", "hgb", "lambdarank"}:
        raise ValueError("unknown adapter type")
    if values.ndim != 2 or values.dtype != np.float64 or values.shape[1] != len(ids):
        raise ValueError("adapter source matrix invalid")
    if indices.shape != (len(values),) or indices.dtype != np.int64:
        raise ValueError("adapter fit indices invalid")
    if _scalar(saved, "model_type") != model_type:
        raise ValueError("adapter model type mismatch")
    saved_ids = tuple(str(value) for value in np.asarray(saved["feature_ids"]).tolist())
    if saved_ids != ids:
        raise ValueError("adapter feature order mismatch")
    if _scalar(saved, "feature_ids_sha256") != _feature_sha(ids):
        raise ValueError("adapter feature fingerprint mismatch")
    if _scalar(saved, "fit_indices_sha256") != _indices_sha(indices):
        raise ValueError("adapter fit-index fingerprint mismatch")
    if _scalar(saved, "fit_scope") != "seed_subset":
        raise ValueError("adapter fit scope mismatch")

    finite = np.isfinite(values)
    counts = finite.sum(axis=0, dtype=np.int64)
    all_missing = counts == 0
    if np.asarray(saved["finite_counts"]).dtype != np.int64 or not np.array_equal(saved["finite_counts"], counts):
        raise ValueError("adapter finite counts mismatch")
    if np.asarray(saved["all_missing"]).dtype != np.bool_ or not np.array_equal(saved["all_missing"], all_missing):
        raise ValueError("adapter all-missing flags mismatch")
    fill_values = np.asarray(saved["fill_values"])
    if fill_values.shape != (values.shape[1],) or fill_values.dtype != np.float64:
        raise ValueError("adapter fill-value schema mismatch")
    if model_type == "lr":
        expected_fill = np.empty(values.shape[1], dtype=np.float64)
        for column in range(values.shape[1]):
            expected_fill[column] = (
                np.median(values[finite[:, column], column])
                if counts[column]
                else NEUTRAL_RANK256
            )
        if not np.array_equal(fill_values, expected_fill):
            raise ValueError("LR independent median mismatch")
        transformed = np.array(values, copy=True)
        missing = np.isnan(transformed)
        transformed[missing] = np.broadcast_to(expected_fill, transformed.shape)[missing]
        if not np.isfinite(transformed).all() or not np.array_equal(transformed[finite], values[finite]):
            raise ValueError("LR transformed matrix semantics mismatch")
    else:
        if not np.isnan(fill_values).all():
            raise ValueError("native-NaN adapter unexpectedly has fill values")
        transformed = np.array(values, copy=True)
        if not np.array_equal(transformed, values, equal_nan=True):
            raise ValueError("native-NaN adapter changed source values")
    return {
        "finite_counts": counts,
        "fill_values": fill_values,
        "all_missing": all_missing,
        "transformed": transformed,
    }


def verify_adapter_artifact(
    path: Path,
    model_type: str,
    X_selected: np.ndarray,
    feature_ids: Sequence[str],
    selected_global_indices: np.ndarray,
) -> Mapping[str, np.ndarray]:
    return verify_adapter_arrays(
        load_npz_mapping(path), model_type, X_selected, feature_ids, selected_global_indices
    )


def verify_file_hash(path: Path, expected: str) -> bool:
    if not Path(path).is_file() or Path(path).is_symlink() or sha256_file(path) != expected:
        raise ValueError("artifact file hash mismatch")
    return True
