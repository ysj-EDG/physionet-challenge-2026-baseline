"""Deterministic, model-free input adapters for G3-B."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


MODEL_TYPES = {"lr", "hgb", "lambdarank"}
NEUTRAL_RANK256 = 128.0 / 255.0


@dataclass(frozen=True)
class AdapterState:
    model_type: str
    feature_ids: Tuple[str, ...]
    feature_ids_sha256: str
    fit_indices_sha256: str
    fit_scope: str
    finite_counts: np.ndarray
    fill_values: np.ndarray
    all_missing: np.ndarray


def _feature_sha(feature_ids: Tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(feature_ids).encode("utf-8")).hexdigest()


def _indices_sha(indices: np.ndarray) -> str:
    payload = np.asarray(indices, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(payload.shape).encode("ascii"))
    digest.update(payload.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _check_matrix(X: np.ndarray, feature_count: int) -> np.ndarray:
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[1] != feature_count or values.dtype != np.float64:
        raise ValueError("adapter input must be a float64 two-dimensional matrix in frozen feature order")
    if np.isinf(values).any():
        raise ValueError("adapter input contains Inf")
    return values


def fit_adapter(
    X_fit: np.ndarray,
    model_type: str,
    feature_ids: Iterable[str],
    *,
    fit_indices: np.ndarray = None,
    fit_scope: str = "full_train_preview",
) -> AdapterState:
    """Fit only deterministic input state; no estimator is constructed."""

    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model type: {model_type}")
    ids = tuple(str(value) for value in feature_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("feature IDs must be non-empty and unique")
    values = _check_matrix(X_fit, len(ids))
    if fit_indices is None:
        indices = np.arange(values.shape[0], dtype=np.int64)
    else:
        indices = np.asarray(fit_indices)
        if indices.shape != (values.shape[0],) or indices.dtype.kind not in "iu":
            raise ValueError("fit_indices must identify every supplied fit row")
        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
            raise ValueError("fit_indices must be unique and non-negative")
    if not fit_scope:
        raise ValueError("fit_scope is required")

    finite = np.isfinite(values)
    counts = finite.sum(axis=0, dtype=np.int64)
    all_missing = counts == 0
    fill_values = np.full(len(ids), np.nan, dtype=np.float64)
    if model_type == "lr":
        for column in range(len(ids)):
            if counts[column]:
                fill_values[column] = np.median(values[finite[:, column], column])
            else:
                fill_values[column] = NEUTRAL_RANK256
        if not np.isfinite(fill_values).all():
            raise ValueError("LR fill values must be finite")

    return AdapterState(
        model_type=model_type,
        feature_ids=ids,
        feature_ids_sha256=_feature_sha(ids),
        fit_indices_sha256=_indices_sha(indices),
        fit_scope=fit_scope,
        finite_counts=_readonly(counts.astype(np.int64)),
        fill_values=_readonly(fill_values),
        all_missing=_readonly(all_missing.astype(bool)),
    )


def transform_adapter(X: np.ndarray, state: AdapterState) -> np.ndarray:
    values = _check_matrix(X, len(state.feature_ids))
    if _feature_sha(state.feature_ids) != state.feature_ids_sha256:
        raise ValueError("adapter feature-order fingerprint mismatch")
    output = np.array(values, dtype=np.float64, copy=True)
    if state.model_type == "lr":
        if state.fill_values.shape != (output.shape[1],) or state.fill_values.dtype != np.float64 or not np.isfinite(state.fill_values).all():
            raise ValueError("LR adapter state invalid")
        missing = np.isnan(output)
        if missing.any():
            output[missing] = np.broadcast_to(state.fill_values, output.shape)[missing]
    elif state.model_type not in {"hgb", "lambdarank"}:
        raise ValueError("adapter model type invalid")
    return output


def check_adapter(source: np.ndarray, transformed: np.ndarray, state: AdapterState) -> bool:
    original = _check_matrix(source, len(state.feature_ids))
    output = np.asarray(transformed)
    if _feature_sha(state.feature_ids) != state.feature_ids_sha256:
        raise ValueError("adapter feature-order fingerprint mismatch")
    width = original.shape[1]
    if output.shape != original.shape or output.dtype != np.float64:
        raise ValueError("adapter output shape or dtype changed")
    if state.finite_counts.shape != (width,) or state.finite_counts.dtype != np.int64 or np.any(state.finite_counts < 0):
        raise ValueError("adapter finite counts invalid")
    if state.all_missing.shape != (width,) or state.all_missing.dtype != np.bool_:
        raise ValueError("adapter all-missing flags invalid")
    if not np.array_equal(state.all_missing, state.finite_counts == 0):
        raise ValueError("adapter all-missing flags inconsistent")
    finite = np.isfinite(original)
    if not np.array_equal(output[finite], original[finite]):
        raise ValueError("adapter changed observed Rank256 values")
    if state.model_type == "lr":
        if np.isnan(output).any() or np.isinf(output).any():
            raise ValueError("LR adapter left a non-finite value")
        expected = np.array(original, copy=True)
        missing = np.isnan(expected)
        expected[missing] = np.broadcast_to(state.fill_values, expected.shape)[missing]
        if not np.array_equal(output, expected):
            raise ValueError("LR transform does not match frozen fill values")
    elif state.model_type in {"hgb", "lambdarank"}:
        if not np.array_equal(output, original, equal_nan=True):
            raise ValueError("native-NaN adapter changed its input")
    else:
        raise ValueError("adapter model type invalid")
    return True
