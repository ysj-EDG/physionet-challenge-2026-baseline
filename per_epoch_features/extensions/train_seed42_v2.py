"""Seed-42 training-role preparation over frozen FeatureLab v2 inputs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from reproduction.siren_a.g4.queries import QueryResult, build_queries
from reproduction.siren_a.g4.sampler import SampleResult, sample_training_rows
from reproduction.siren_a.pipeline.adapters import AdapterState, check_adapter, fit_adapter, transform_adapter

from .train_inputs_v2 import V2FoldInputs


SEED = 42
MODEL_TYPES = ("lr", "hgb", "lambdarank")


@dataclass(frozen=True)
class Seed42Training:
    fold: V2FoldInputs
    sample: SampleResult
    query: QueryResult

    @property
    def X_sample(self) -> np.ndarray:
        return self.fold.X_train[self.sample.local_positions]

    @property
    def y_sample(self) -> np.ndarray:
        return self.fold.y_train[self.sample.local_positions]

    @property
    def age_sample(self) -> np.ndarray:
        return self.fold.age_train[self.sample.local_positions]


def prepare_seed42_training(fold: V2FoldInputs) -> Seed42Training:
    """Apply exactly the frozen sampler and query builder to one training role."""
    sample = sample_training_rows(
        fold.train_site, SEED, fold.train_age_groups, fold.y_train, fold.train_indices
    )
    query = build_queries(
        fold.age_train[sample.local_positions],
        fold.y_train[sample.local_positions],
        sample.local_positions,
        sample.global_indices,
    )
    return Seed42Training(fold=fold, sample=sample, query=query)


def compare_training_protocols_v2(first: Seed42Training, second: Seed42Training) -> bool:
    """Prove that transductive/inductive protocols share the same training role."""
    if first.fold.protocol == second.fold.protocol:
        raise ValueError("protocol comparison requires distinct protocol names")
    for name in ("X_train", "train_valid_mask", "y_train", "age_train", "train_age_groups", "train_indices"):
        left = getattr(first.fold, name)
        right = getattr(second.fold, name)
        if not np.array_equal(left, right, equal_nan=True):
            raise ValueError(f"protocol training mismatch: {name}")
    if first.fold.train_site != second.fold.train_site or first.fold.feature_ids != second.fold.feature_ids:
        raise ValueError("protocol training identity mismatch")
    if not np.array_equal(first.sample.local_positions, second.sample.local_positions):
        raise ValueError("protocol sample local positions mismatch")
    if not np.array_equal(first.sample.global_indices, second.sample.global_indices):
        raise ValueError("protocol sample global indices mismatch")
    if first.query.fingerprint != second.query.fingerprint:
        raise ValueError("protocol query fingerprint mismatch")
    return True


def fit_seed42_adapter_v2(
    fold: V2FoldInputs, prepared: Seed42Training, model_type: str
) -> tuple[AdapterState, np.ndarray]:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model type: {model_type}")
    if prepared.fold is not fold:
        raise ValueError("prepared training inputs do not belong to fold")
    X = prepared.X_sample
    state = fit_adapter(
        X,
        model_type,
        fold.feature_ids,
        fit_indices=prepared.sample.global_indices,
        fit_scope="seed_subset",
    )
    transformed = transform_adapter(X, state)
    check_adapter(X, transformed, state)
    return state, transformed


def compare_old_sample(prepared: Seed42Training, old_root: Path) -> bool:
    """Read-only exact bridge to one immutable G4-A sample unit."""
    path = Path(old_root) / "samples" / f"{prepared.fold.train_site}_seed42" / "selected_indices.npz"
    if not path.is_file():
        raise ValueError(f"old sample unavailable: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["local_positions"], prepared.sample.local_positions)
        np.testing.assert_array_equal(arrays["global_indices"], prepared.sample.global_indices)
    return True


def compare_old_query(prepared: Seed42Training, old_root: Path) -> bool:
    """Read-only exact bridge to one immutable G4-A query unit."""
    path = Path(old_root) / "queries" / f"{prepared.fold.train_site}_seed42" / "query.npz"
    if not path.is_file():
        raise ValueError(f"old query unavailable: {path}")
    expected = prepared.query
    keys = (
        "permutation", "inverse_permutation", "sorted_local_positions",
        "sorted_global_indices", "sorted_group_ids", "sorted_labels",
        "group_ids", "group_sizes", "group_positive", "group_negative",
    )
    with np.load(path, allow_pickle=False) as arrays:
        for key in keys:
            np.testing.assert_array_equal(arrays[key], getattr(expected, key))
    audit_path = path.parent / "query_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if str(audit.get("query_fingerprint", "")) != expected.fingerprint:
        raise ValueError("old query fingerprint mismatch")
    return True

