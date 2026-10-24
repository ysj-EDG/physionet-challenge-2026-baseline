"""Frozen CAISR20 route contract and small, testable numerical helpers.

This module deliberately contains no data loading, labels, scoring targets, or
filesystem side effects.  The route runner keeps those concerns private and
uses this module only for the frozen feature/model/preprocessing contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

import feature_scaling as _feature_scaling


ROUTE_ID = "CAISR20_CORE_PROTOCOL"
CAISR20_FEATURE_IDS = (
    "caisr_sleep_tst_sec",
    "caisr_sleep_se",
    "caisr_sleep_sol_sec",
    "caisr_sleep_rem_latency_sec",
    "caisr_sleep_waso_sec",
    "caisr_sleep_n1_pct",
    "caisr_sleep_n2_pct",
    "caisr_sleep_n3_pct",
    "caisr_sleep_rem_pct",
    "caisr_sleep_transition_rate",
    "caisr_sleep_short_bout_ratio",
    "caisr_sleep_mean_stage_entropy",
    "caisr_arousal_arousal_index",
    "caisr_arousal_arousal_burden_ratio",
    "caisr_arousal_mean_arousal_duration",
    "caisr_respiratory_respiratory_event_index",
    "caisr_respiratory_respiratory_burden_ratio",
    "caisr_respiratory_mean_resp_duration",
    "caisr_limb_limb_movement_index",
    "caisr_limb_plmi",
)
CAISR20_X_STATIC_INDICES = (11, 12, 13, 14, 16, 17, 18, 19, 20, 32,
                            34, 87, 96, 97, 98, 126, 127, 141, 168, 173)
SEEDS = tuple(range(42, 57))
SITES = ("I0002", "I0006", "S0001")
HOLDOUT_TO_TRAIN = {"I0002": "S0001", "I0006": "S0001", "S0001": "I0006"}
EXPECTED_SITE_COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}
EXPECTED_SITE_POSITIVES = {"I0002": 52, "I0006": 112, "S0001": 334}


def model_config(seed: int) -> dict[str, object]:
    """Return the historical CAISR20 LR config with only the seed varied."""
    if isinstance(seed, bool) or int(seed) not in SEEDS:
        raise ValueError("CAISR20 CORE seeds are exactly 42..56")
    return {
        "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.4,
        "C": 0.03, "class_weight": "balanced", "fit_intercept": True,
        "max_iter": 100000, "tol": 0.0001, "random_state": int(seed),
    }


@dataclass(frozen=True)
class SelectedPreprocessor:
    median: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    all_missing: np.ndarray

    def __post_init__(self):
        arrays = (self.median, self.center, self.scale, self.all_missing)
        if self.median.ndim != 1 or self.center.shape != self.median.shape or self.scale.shape != self.median.shape or self.all_missing.shape != self.median.shape:
            raise ValueError("preprocessor state shape mismatch")
        if not np.isfinite(self.median).all() or not np.isfinite(self.center).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("preprocessor state must be finite with positive scales")
        if self.all_missing.dtype != np.dtype(bool):
            raise ValueError("all_missing must be boolean")


def fit_selected_preprocessor(values: np.ndarray) -> SelectedPreprocessor:
    """Fit historical median/linear scaling on selected training rows only."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != len(CAISR20_FEATURE_IDS):
        raise ValueError("selected CAISR20 values must be N x 20 and non-empty")
    median = np.zeros(x.shape[1], dtype=np.float64)
    all_missing = np.zeros(x.shape[1], dtype=bool)
    for column in range(x.shape[1]):
        finite = np.isfinite(x[:, column])
        all_missing[column] = not bool(finite.any())
        median[column] = float(np.median(x[finite, column])) if finite.any() else 0.0
    filled = np.where(np.isfinite(x), x, median)
    center = np.asarray(filled.mean(axis=0), dtype=np.float64)
    scale = np.asarray(filled.std(axis=0), dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0).astype(np.float64)
    return SelectedPreprocessor(median, center, scale, all_missing)


def apply_selected_preprocessor(values: np.ndarray, state: SelectedPreprocessor) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(CAISR20_FEATURE_IDS):
        raise ValueError("CAISR20 values must be N x 20")
    return ((np.where(np.isfinite(x), x, state.median) - state.center) / state.scale).astype(np.float64)


@dataclass(frozen=True)
class HistoricalTypedState:
    """Selected-row typed_v1 parameters for the frozen CAISR20 columns."""
    center: np.ndarray
    scale: np.ndarray
    scale_method: tuple[str, ...]
    fit_valid_counts: np.ndarray
    all_missing: np.ndarray

    def __post_init__(self):
        arrays = (self.center, self.scale, self.fit_valid_counts, self.all_missing)
        if any(np.asarray(item).shape != (len(CAISR20_FEATURE_IDS),) for item in arrays):
            raise ValueError("historical typed state must contain 20 values per field")
        if not np.isfinite(self.center).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("historical typed state must be finite with positive scales")
        if np.asarray(self.fit_valid_counts).dtype.kind not in "iu":
            raise ValueError("fit_valid_counts must be integer")
        if np.asarray(self.all_missing).dtype != np.dtype(bool):
            raise ValueError("all_missing must be boolean")
        if len(self.scale_method) != len(CAISR20_FEATURE_IDS):
            raise ValueError("scale_method must contain 20 values")


def _caisr_sentinel_mask(full_static: np.ndarray) -> np.ndarray:
    static = np.asarray(full_static, dtype=np.float64)
    if static.ndim != 2 or static.shape[1] != 196:
        raise ValueError("historical CAISR typed input requires N x 196 x_static values")
    return np.isfinite(static[:, 10:]).all(axis=1) & (static[:, 10] == 0.0) & np.all(static[:, 10:] == 0.0, axis=1)


def fit_historical_typed_caisr20(full_static: np.ndarray,
                                 rules: Sequence[Mapping[str, object]]) -> HistoricalTypedState:
    """Fit typed_v1 parameters using only the supplied selected training rows."""
    static = np.asarray(full_static, dtype=np.float64)
    if static.ndim != 2 or static.shape[1] != 196 or len(rules) != len(CAISR20_FEATURE_IDS):
        raise ValueError("historical typed fit requires N x 196 static values and 20 rules")
    sentinel = _caisr_sentinel_mask(static)
    center = np.zeros(20, dtype=np.float64)
    scale = np.ones(20, dtype=np.float64)
    counts = np.zeros(20, dtype=np.int64)
    all_missing = np.zeros(20, dtype=bool)
    methods = []
    for j, (index, rule) in enumerate(zip(CAISR20_X_STATIC_INDICES, rules)):
        prepared, valid, _ = _feature_scaling._prepare_values(static[:, index], rule)
        if index >= 10:
            valid &= ~sentinel
        values = np.asarray(prepared[valid], dtype=np.float64)
        counts[j] = int(values.size)
        if rule.get("scaling") == "identity":
            methods.append("identity")
            continue
        if values.size == 0:
            center[j] = 0.0
            scale[j] = 1.0
            methods.append("all_missing_fallback")
            all_missing[j] = True
            continue
        weights = np.ones(values.size, dtype=np.float64)
        q05, q25, q50, q75, q95 = (
            _feature_scaling._weighted_quantile(values, weights, q)
            for q in (0.05, 0.25, 0.50, 0.75, 0.95)
        )
        center[j] = float(q50)
        tolerance = 1e-6 * max(1.0, abs(center[j]))
        mean = float(np.average(values, weights=weights))
        candidates = (
            ("iqr", q75 - q25),
            ("p95_p05", q95 - q05),
            ("std", float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))),
        )
        method, chosen = "unit_fallback", 1.0
        for candidate_method, candidate_scale in candidates:
            if np.isfinite(candidate_scale) and candidate_scale > tolerance:
                method, chosen = candidate_method, float(candidate_scale)
                break
        methods.append(method)
        scale[j] = chosen
    return HistoricalTypedState(center, scale, tuple(methods), counts, all_missing)


def apply_historical_typed_caisr20(full_static: np.ndarray,
                                   rules: Sequence[Mapping[str, object]],
                                   state: HistoricalTypedState) -> np.ndarray:
    """Apply typed_v1 rules, sentinel masking, and fitted selected-row parameters."""
    static = np.asarray(full_static, dtype=np.float64)
    if static.ndim != 2 or static.shape[1] != 196 or len(rules) != 20:
        raise ValueError("historical typed transform requires N x 196 static values and 20 rules")
    sentinel = _caisr_sentinel_mask(static)
    out = np.empty((static.shape[0], 20), dtype=np.float64)
    for j, (index, rule) in enumerate(zip(CAISR20_X_STATIC_INDICES, rules)):
        prepared, valid, _ = _feature_scaling._prepare_values(static[:, index], rule)
        if index >= 10:
            valid &= ~sentinel
        if rule.get("scaling") == "robust":
            out[:, j] = (np.where(valid, prepared, state.center[j]) - state.center[j]) / state.scale[j]
        else:
            out[:, j] = np.where(valid, prepared, float(rule.get("identity_fill", 0.0)))
    if not np.isfinite(out).all():
        raise ValueError("historical typed transform produced nonfinite output")
    return out


def typed_caisr20(values: np.ndarray, rules: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Apply the frozen typed_v1 transform to raw x_static CAISR columns.

    This is the selected-column equivalent of ``feature_scaling._prepare_values``;
    it does not fit any statistics and therefore is safe for holdout rows.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 20 or len(rules) != 20:
        raise ValueError("raw CAISR20 values and rules must both have width 20")
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for j, rule in enumerate(rules):
        values_j = x[:, j].copy()
        valid = np.isfinite(values_j)
        if rule.get("missing_rule") == "nonfinite_or_nonpositive":
            valid &= values_j > 0
        expected = rule.get("expected_range") or {}
        if expected.get("hard", False):
            tolerance = float(rule.get("range_tolerance", 1e-6))
            lower, upper = expected.get("min"), expected.get("max")
            if lower is not None:
                near = valid & (values_j < float(lower)) & (values_j >= float(lower) - tolerance)
                values_j[near] = float(lower)
                valid &= (values_j >= float(lower) - tolerance)
            if upper is not None:
                near = valid & (values_j > float(upper)) & (values_j <= float(upper) + tolerance)
                values_j[near] = float(upper)
                valid &= (values_j <= float(upper) + tolerance)
        transformed = np.full(values_j.shape, np.nan, dtype=np.float64)
        transformed[valid] = values_j[valid] / float(rule["unit_scale"])
        if rule.get("nonlinear") == "log1p":
            nonnegative = valid & (transformed >= 0)
            transformed[nonnegative] = np.log1p(transformed[nonnegative])
            transformed[valid & ~nonnegative] = np.nan
            valid &= nonnegative
        valid &= np.isfinite(transformed)
        out[valid, j] = transformed[valid]
    return out


def mean_rank_fusion(scores: np.ndarray) -> np.ndarray:
    """Frozen average-tie percentile rank for one 15-seed holdout cohort."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 15 or not np.isfinite(values).all():
        raise ValueError("CAISR final fusion requires finite N x 15 scores")
    ranks = np.empty(values.shape, dtype=np.float64)
    n = values.shape[0]
    if n == 0:
        raise ValueError("cannot fuse an empty score matrix")
    for column in range(values.shape[1]):
        _, inverse, counts = np.unique(values[:, column], return_inverse=True, return_counts=True)
        less = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts[:-1])))
        ranks[:, column] = (less[inverse] + 0.5 * counts[inverse]) / n
    return ranks.mean(axis=1, dtype=np.float64)


def route_delta(core: float, caisr: float) -> float:
    return float(core) - float(caisr)


def interaction_delta(core_both: float, core_without_resp: float,
                      core_without_caisr: float, core_without_both: float) -> float:
    return float(core_both) - float(core_without_resp) - float(core_without_caisr) + float(core_without_both)


def manifest_payload() -> dict[str, object]:
    return {
        "schema": "featurelab_caisr20_core_protocol_manifest_v1",
        "route_id": ROUTE_ID,
        "feature_count": 20,
        "feature_ids": list(CAISR20_FEATURE_IDS),
        "x_static_indices": list(CAISR20_X_STATIC_INDICES),
        "demographic_feature_count": 0,
        "seeds": list(SEEDS),
        "holdout_to_train": dict(HOLDOUT_TO_TRAIN),
        "model_config_except_seed": {key: value for key, value in model_config(42).items() if key != "random_state"},
        "preprocessing_fit_scope": "selected training rows only",
        "primary_comparator_protocol": "inductive",
    }


__all__ = [
    "CAISR20_FEATURE_IDS", "CAISR20_X_STATIC_INDICES", "EXPECTED_SITE_COUNTS",
    "EXPECTED_SITE_POSITIVES", "HOLDOUT_TO_TRAIN", "ROUTE_ID", "SEEDS", "SITES",
    "HistoricalTypedState", "apply_historical_typed_caisr20", "fit_historical_typed_caisr20",
    "SelectedPreprocessor", "apply_selected_preprocessor", "fit_selected_preprocessor",
    "interaction_delta", "manifest_payload", "mean_rank_fusion", "model_config",
    "route_delta", "typed_caisr20",
]
