"""Independent cache validation and resume contract for the G2-B1 full run."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from g2a_io import read_cache

CACHE_SCHEMA_VERSION = "g2b1_full_v1"
REQUIRED_PROVENANCE = (
    "source_npz_sha256", "manifest_sha256", "spec_sha256",
    "g2a_features_sha256", "g2a_io_sha256",
    "g2b0_batch_safety_sha256", "g2b1_cache_contract_sha256",
    "g2b1_full_run_sha256", "python_version", "numpy_version",
    "cache_schema_version",
)
POOLED_FAMILIES = {
    "EEG54_stage_spectral", "spatial_asymmetry_candidate",
    "spatial_ap_gradient_candidate", "HRV11", "EMG24", "Resp14",
}
STATIC_FAMILIES = {"CAISR20", "Demo5"}


class Action(str, Enum):
    COMPUTE = "COMPUTE"
    SKIP_VERIFIED = "SKIP_VERIFIED"
    STOP_INVALID = "STOP_INVALID"


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]


def _scalar(array, key, errors):
    value = np.asarray(array)
    if value.shape != ():
        errors.append(f"{key}: not scalar")
        return None
    return value.item()


def _check_triplet(values, mask, support, name, shape, errors):
    shape_ok = values.shape == mask.shape == support.shape == shape
    dtype_ok = (values.dtype == np.dtype(np.float64)
                and mask.dtype == np.dtype(bool)
                and support.dtype == np.dtype(np.int64))
    if values.shape != shape: errors.append(f"{name}: values shape")
    if mask.shape != shape: errors.append(f"{name}: mask shape")
    if support.shape != shape: errors.append(f"{name}: support shape")
    if values.dtype != np.dtype(np.float64): errors.append(f"{name}: values dtype")
    if mask.dtype != np.dtype(bool): errors.append(f"{name}: mask dtype")
    if support.dtype != np.dtype(np.int64): errors.append(f"{name}: support dtype")
    if not (shape_ok and dtype_ok): return False
    if np.isinf(values).any(): errors.append(f"{name}: Inf")
    if not np.isfinite(values[mask]).all(): errors.append(f"{name}: nonfinite valid")
    if not np.isnan(values[~mask]).all(): errors.append(f"{name}: invalid not NaN")
    if np.any(support < 0): errors.append(f"{name}: negative support")
    return True


def _family_by_unique_id(manifest):
    first = {}
    for row in manifest.rows:
        first.setdefault(row["unique_feature_id"], row["family"])
    if set(first) != set(manifest.unique_ids):
        raise ValueError("Manifest unique family mapping mismatch")
    return first


def _check_support(mask, support, manifest, errors):
    families = _family_by_unique_id(manifest)
    for position, uid in enumerate(manifest.unique_ids):
        family = families[uid]
        if family in POOLED_FAMILIES:
            if mask[position] and support[position] < 5:
                errors.append(f"unique: {uid} valid pooled support < 5")
            if not mask[position] and support[position] >= 5:
                errors.append(f"unique: {uid} invalid pooled support >= 5")
        elif family in STATIC_FAMILIES:
            expected = 1 if mask[position] else 0
            if support[position] != expected:
                errors.append(f"unique: {uid} static support != {expected}")
        else:
            errors.append(f"unique: unsupported family {family}")


def validate_cache_arrays(arrays, expected_provenance, manifest):
    errors = []
    required = {
        "unique_feature_ids", "unique_values", "unique_valid_mask",
        "unique_support_count", *REQUIRED_PROVENANCE,
    }
    for view in manifest.view_sizes:
        required.update({
            f"{view}_values", f"{view}_valid_mask", f"{view}_support_count",
        })
    unexpected = sorted(set(arrays).difference(required))
    missing = sorted(required.difference(arrays))
    if unexpected: errors.append("unexpected keys: " + ",".join(unexpected))
    if missing:
        errors.append("missing keys: " + ",".join(missing))
        return ValidationResult(False, tuple(errors))
    if any(np.asarray(value).dtype == object for value in arrays.values()):
        errors.append("object dtype")
    ids = np.asarray(arrays["unique_feature_ids"])
    if ids.shape != (807,) or ids.tolist() != manifest.unique_ids:
        errors.append("unique_feature_ids mismatch")
    unique_values = np.asarray(arrays["unique_values"])
    unique_mask = np.asarray(arrays["unique_valid_mask"])
    unique_support = np.asarray(arrays["unique_support_count"])
    unique_safe = _check_triplet(
        unique_values, unique_mask, unique_support, "unique", (807,), errors)
    if unique_safe: _check_support(unique_mask, unique_support, manifest, errors)
    positions_by_id = {uid: index for index, uid in enumerate(manifest.unique_ids)}
    for view, size in manifest.view_sizes.items():
        values = np.asarray(arrays[f"{view}_values"])
        mask = np.asarray(arrays[f"{view}_valid_mask"])
        support = np.asarray(arrays[f"{view}_support_count"])
        view_safe = _check_triplet(values, mask, support, view, (size,), errors)
        if unique_safe and view_safe:
            rows = sorted(
                (row for row in manifest.rows if row["view_id"] == view),
                key=lambda row: int(row["view_position"]),
            )
            positions = np.asarray([
                positions_by_id[row["unique_feature_id"]] for row in rows])
            if not np.array_equal(values, unique_values[positions], equal_nan=True):
                errors.append(f"{view}: values mapping")
            if not np.array_equal(mask, unique_mask[positions]):
                errors.append(f"{view}: mask mapping")
            if not np.array_equal(support, unique_support[positions]):
                errors.append(f"{view}: support mapping")
    for key in REQUIRED_PROVENANCE:
        actual = _scalar(arrays[key], key, errors)
        if key not in expected_provenance:
            errors.append(f"{key}: expectation missing")
        elif actual != expected_provenance[key]:
            errors.append(f"{key}: mismatch")
    return ValidationResult(not errors, tuple(errors))


def assert_valid_cache_arrays(arrays, expected_provenance, manifest):
    result = validate_cache_arrays(arrays, expected_provenance, manifest)
    if not result.valid: raise ValueError("; ".join(result.errors))


def validate_existing_cache(path, expected_provenance, manifest):
    path = Path(path)
    if not path.is_file(): return ValidationResult(False, ("cache missing",))
    try:
        arrays = read_cache(path)
        return validate_cache_arrays(arrays, expected_provenance, manifest)
    except Exception as exc:
        return ValidationResult(False, (f"cache invalid: {type(exc).__name__}",))


def decide_resume_action(path, expected_provenance, manifest):
    path = Path(path)
    if not path.exists(): return Action.COMPUTE
    result = validate_existing_cache(path, expected_provenance, manifest)
    return Action.SKIP_VERIFIED if result.valid else Action.STOP_INVALID
