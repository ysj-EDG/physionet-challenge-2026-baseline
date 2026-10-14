"""Five-year LambdaRank query construction and diagnostics."""

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class QueryResult:
    permutation: np.ndarray
    inverse_permutation: np.ndarray
    sorted_local_positions: np.ndarray
    sorted_global_indices: np.ndarray
    sorted_group_ids: np.ndarray
    sorted_labels: np.ndarray
    group_ids: np.ndarray
    group_sizes: np.ndarray
    group_positive: np.ndarray
    group_negative: np.ndarray
    truncation_level: int
    fingerprint: str


def build_queries(ages: np.ndarray, labels: np.ndarray, local_positions: np.ndarray,
                  global_indices: np.ndarray) -> QueryResult:
    ages = np.asarray(ages, dtype=np.float64)
    y = np.asarray(labels)
    local = np.asarray(local_positions)
    global_idx = np.asarray(global_indices)
    if ages.ndim != 1 or y.shape != ages.shape or local.shape != ages.shape or global_idx.shape != ages.shape:
        raise ValueError("query inputs must be aligned")
    if not np.isfinite(ages).all() or not np.all(np.isin(y, [0, 1])):
        raise ValueError("query ages must be finite and labels binary")
    groups = np.floor(ages / 5.0).astype(np.int64)
    permutation = np.argsort(groups, kind="stable").astype(np.int64)
    sorted_groups = groups[permutation]
    sorted_labels = y[permutation].astype(np.int8)
    group_ids, first, sizes = np.unique(sorted_groups, return_index=True, return_counts=True)
    positives = np.asarray([np.sum(sorted_labels[index:index + size] == 1)
                            for index, size in zip(first, sizes)], dtype=np.int64)
    negatives = sizes.astype(np.int64) - positives
    if np.any(positives == 0) or np.any(negatives == 0):
        raise ValueError("each LambdaRank query must contain both classes")
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation), dtype=np.int64)
    payload = {"global": global_idx[permutation].astype(np.int64).tolist(),
               "groups": group_ids.tolist(), "sizes": sizes.tolist(),
               "labels": sorted_labels.tolist()}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                            separators=(",", ":")).encode("utf-8")).hexdigest()
    return QueryResult(permutation, inverse, local[permutation].astype(np.int64),
                       global_idx[permutation].astype(np.int64), sorted_groups,
                       sorted_labels, group_ids.astype(np.int64), sizes.astype(np.int64),
                       positives, negatives, int(sizes.max()), fingerprint)


def audit_ac_pair_coverage(ages: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> Mapping[str, object]:
    ages = np.asarray(ages, dtype=np.float64)
    y = np.asarray(labels)
    groups = np.asarray(groups)
    if ages.ndim != 1 or y.shape != ages.shape or groups.shape != ages.shape:
        raise ValueError("AC pair inputs invalid")
    if not np.isfinite(ages).all() or not np.all(np.isin(y, [0, 1])):
        raise ValueError("AC pair inputs invalid")
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    valid = np.abs(ages[positive, None] - ages[negative][None, :]) <= 2.0
    total = int(valid.sum())
    if total == 0:
        raise ValueError("no valid opposite-label age pairs")
    same_group = groups[positive, None] == groups[negative][None, :]
    within = int((valid & same_group).sum())
    return {"all_valid_pairs": total, "within_group_valid_pairs": within,
            "pair_coverage": within / total}
