"""Immutable contracts shared by FeatureLab v2 source, family, and cache code."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible contract material deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class FeatureLabError(Exception):
    """Base class for FeatureLab failures."""


class RecordFatalError(FeatureLabError):
    """A source-level failure that prevents all families for one record."""


class SourceContractError(RecordFatalError):
    """The source file, schema, or canonical identity is invalid."""


class RecordKeyCollisionError(RecordFatalError):
    """One deterministic record key maps to different canonical identities."""


class FamilyInputReviewError(FeatureLabError):
    """A family-specific input cannot be validated without review."""


class FamilyComputeError(FeatureLabError):
    """A family failed after its input contract validated."""


class FamilyRunStatus(str, Enum):
    BUILT = "BUILT"
    SKIP_VERIFIED = "SKIP_VERIFIED"
    REQUIRES_INPUT_REVIEW = "REQUIRES_INPUT_REVIEW"
    FAILED = "FAILED"
    STOP_INVALID = "STOP_INVALID"


class RecordRunStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    STOP_RECORD = "STOP_RECORD"


@dataclass(frozen=True)
class SourceIdentity:
    namespace: str
    site_id: str
    record_id: str
    record_key: str
    source_size: int
    source_sha256: str
    extraction_schema: str
    validity_schema: str
    digest: str
    source_path: str
    source_mtime_ns: int
    source_mode: int

    def identity_material(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "site_id": self.site_id,
            "record_id": self.record_id,
            "record_key": self.record_key,
            "source_size": self.source_size,
            "source_sha256": self.source_sha256,
            "extraction_schema": self.extraction_schema,
            "validity_schema": self.validity_schema,
        }

    def canonical_json(self) -> str:
        return canonical_json({**self.identity_material(), "digest": self.digest})


@dataclass(frozen=True)
class SourceRecord:
    identity: SourceIdentity
    arrays: Mapping[str, np.ndarray]
    field_errors: Mapping[str, str]
    loaded_fields: tuple[str, ...]

    @classmethod
    def create(
        cls,
        identity: SourceIdentity,
        arrays: Mapping[str, np.ndarray],
        field_errors: Mapping[str, str] | None = None,
    ) -> "SourceRecord":
        copied: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            array = np.asarray(value).copy()
            array.setflags(write=False)
            copied[name] = array
        ordered = dict(sorted(copied.items()))
        errors = dict(sorted((field_errors or {}).items()))
        if set(ordered).intersection(errors):
            raise ValueError("Loaded fields and field errors must be disjoint")
        return cls(
            identity,
            MappingProxyType(ordered),
            MappingProxyType(errors),
            tuple(ordered),
        )


ShapeDimension = int | str


@dataclass(frozen=True)
class FieldRequirement:
    name: str
    shape: tuple[ShapeDimension, ...]
    accepted_dtypes: tuple[str, ...]
    columns: tuple[int, ...] = ()
    column_axis: int = -1

    def canonical_material(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "accepted_dtypes": list(self.accepted_dtypes),
            "columns": list(self.columns),
            "column_axis": self.column_axis,
        }


@dataclass(frozen=True)
class FamilyInputContract:
    family_id: str
    contract_version: str
    requirements: tuple[FieldRequirement, ...]
    stage_time_rules: tuple[str, ...] = ()
    validity_rules: tuple[str, ...] = ()

    def material(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "contract_version": self.contract_version,
            "requirements": [item.canonical_material() for item in self.requirements],
            "stage_time_rules": list(self.stage_time_rules),
            "validity_rules": list(self.validity_rules),
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.material())


@dataclass(frozen=True)
class ObservedField:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def material(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class FamilyInputIdentity:
    family_id: str
    source_identity_digest: str
    contract_digest: str
    observed_fields: tuple[ObservedField, ...]
    digest: str

    def material(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "source_identity_digest": self.source_identity_digest,
            "contract_digest": self.contract_digest,
            "observed_fields": [field.material() for field in self.observed_fields],
        }

    def canonical_json(self) -> str:
        return canonical_json({**self.material(), "digest": self.digest})


@dataclass(frozen=True)
class FeatureResult:
    family_id: str
    family_version: str
    feature_ids: tuple[str, ...]
    values: np.ndarray
    valid_mask: np.ndarray
    support_count: np.ndarray
    missing_reason: tuple[str, ...]
    feature_units: tuple[str, ...]
    source_identity: SourceIdentity
    processing_parameters: Mapping[str, Any]

    def replaced(self, **changes: Any) -> "FeatureResult":
        return replace(self, **changes)


@dataclass(frozen=True)
class FamilyRunResult:
    family_id: str
    status: FamilyRunStatus
    reason_code: str = ""
    cache_path: Path | None = None
    result: FeatureResult | None = None


@dataclass(frozen=True)
class RecordRunSummary:
    status: RecordRunStatus
    source_identity: SourceIdentity | None
    family_results: tuple[FamilyRunResult, ...]
    reason_code: str = ""


def validate_feature_result(result: FeatureResult) -> FeatureResult:
    length = len(result.feature_ids)
    parallel = (
        result.values,
        result.valid_mask,
        result.support_count,
        result.missing_reason,
        result.feature_units,
    )
    if any(len(item) != length for item in parallel):
        raise ValueError("FeatureResult fields must have equal lengths")
    if not result.family_id or not result.family_version:
        raise ValueError("FeatureResult family identity is required")
    if any(not feature_id for feature_id in result.feature_ids):
        raise ValueError("Feature IDs must be nonempty")
    if len(set(result.feature_ids)) != length:
        raise ValueError("Feature IDs must be unique")

    values = np.asarray(result.values)
    valid = np.asarray(result.valid_mask)
    support = np.asarray(result.support_count)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Feature values must be a one-dimensional floating array")
    if valid.ndim != 1 or valid.dtype != np.dtype(bool):
        raise ValueError("valid_mask must be a one-dimensional bool array")
    if support.ndim != 1 or not np.issubdtype(support.dtype, np.integer):
        raise ValueError("support_count must be a one-dimensional integer array")
    if np.any(support < 0):
        raise ValueError("support_count cannot be negative")
    if not np.all(np.isfinite(values[valid])):
        raise ValueError("Valid feature values must be finite")
    if not np.all(np.isnan(values[~valid])):
        raise ValueError("Invalid feature values must be NaN")
    if any(not reason for reason, is_valid in zip(result.missing_reason, valid) if not is_valid):
        raise ValueError("Every invalid feature needs a missing reason")
    return result
