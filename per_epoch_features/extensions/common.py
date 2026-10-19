"""Pure numerical helpers shared by FeatureLab feature families."""
from __future__ import annotations

import numpy as np

STAGE_CODES: dict[str, tuple[int, ...]] = {
    "N3": (1,),
    "N2": (2,),
    "N1": (3,),
    "REM": (4,),
    "Wake": (5,),
    "NREM": (1, 2, 3),
    "Sleep": (1, 2, 3, 4),
    "All-stage": (1, 2, 3, 4, 5),
}


def stage_mask(
    stage_codes: np.ndarray,
    stage_valid: np.ndarray,
    stage_scope: str,
) -> np.ndarray:
    codes = np.asarray(stage_codes)
    valid = np.asarray(stage_valid)
    if codes.ndim != 1 or valid.ndim != 1 or codes.shape != valid.shape:
        raise ValueError("stage_codes and stage_valid must be equal-length vectors")
    if valid.dtype != np.dtype(bool):
        raise ValueError("stage_valid must be boolean")
    try:
        accepted = STAGE_CODES[stage_scope]
    except KeyError as exc:
        raise ValueError(f"Unknown stage scope: {stage_scope}") from exc
    return valid & np.isfinite(codes) & np.isin(codes, accepted)


def adjacent_pairs(epoch_start_sec: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    starts = np.asarray(epoch_start_sec)
    valid = np.asarray(valid_mask)
    if starts.ndim != 1 or valid.ndim != 1 or starts.shape != valid.shape:
        raise ValueError("epoch_start_sec and valid_mask must be equal-length vectors")
    if valid.dtype != np.dtype(bool):
        raise ValueError("valid_mask must be boolean")
    if len(starts) < 2:
        return np.zeros(0, dtype=bool)
    return (
        valid[:-1]
        & valid[1:]
        & np.isfinite(starts[:-1])
        & np.isfinite(starts[1:])
        & (np.diff(starts) == 30.0)
    )


def relative_eeg_power(
    log_p_band_over_beta: np.ndarray,
    log_p_beta: np.ndarray,
    log_p_zero: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(log_p_band_over_beta)
        + np.asarray(log_p_beta)
        - np.asarray(log_p_zero)
    )


def linear_quantile(values: np.ndarray, quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("linear_quantile requires a nonempty finite vector")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    return float(np.quantile(array, quantile, method="linear"))


def pool_mean_iqr(
    values: np.ndarray,
    valid_mask: np.ndarray,
    minimum_support: int,
) -> tuple[float, float, int]:
    array = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_mask)
    if array.ndim != 1 or valid.ndim != 1 or array.shape != valid.shape:
        raise ValueError("values and valid_mask must be equal-length vectors")
    if valid.dtype != np.dtype(bool):
        raise ValueError("valid_mask must be boolean")
    if minimum_support < 1:
        raise ValueError("minimum_support must be positive")
    selected = array[valid & np.isfinite(array)]
    support = int(len(selected))
    if support < minimum_support:
        return float("nan"), float("nan"), support
    lower, upper = np.quantile(selected, (0.25, 0.75), method="linear")
    return float(np.mean(selected)), float(upper - lower), support
