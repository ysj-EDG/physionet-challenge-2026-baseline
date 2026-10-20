"""Strict, no-clobber family cache for FeatureLab v2."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import (
    FamilyInputIdentity,
    FeatureResult,
    canonical_json,
    canonical_sha256,
    validate_feature_result,
)
from .recipe import RecipeIdentity

CACHE_SCHEMA_VERSION = "featurelab_family_cache_v1"
COMPLETE_STATUS = "COMPLETE"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_FAMILY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_RECORD = re.compile(r"^rec_[0-9a-f]{64}$")
_SCALAR_TEXT_KEYS = frozenset(
    {
        "cache_schema",
        "completion_status",
        "family_id",
        "family_version",
        "recipe_id",
        "recipe_material_json",
        "recipe_provenance_json",
        "source_identity_json",
        "source_provenance_json",
        "source_identity_digest",
        "family_input_identity_json",
        "family_input_identity_digest",
        "processing_parameters_json",
        "payload_sha256",
    }
)
_VECTOR_TEXT_KEYS = frozenset({"feature_ids", "missing_reason", "feature_units"})
_REQUIRED_KEYS = _SCALAR_TEXT_KEYS | _VECTOR_TEXT_KEYS | frozenset(
    {"values", "valid_mask", "support_count"}
)


class CacheValidationError(ValueError):
    """An existing final cache does not satisfy the frozen schema."""


class ResumeAction(str, Enum):
    BUILD_NEW = "BUILD_NEW"
    SKIP_VERIFIED = "SKIP_VERIFIED"
    STOP_INVALID = "STOP_INVALID"


@dataclass(frozen=True)
class CacheExpectation:
    family_id: str
    family_version: str
    recipe_id: str
    feature_ids: tuple[str, ...]
    source_identity_digest: str
    family_input_identity_digest: str


def cache_path(
    root: Path,
    family_id: str,
    recipe_id: str,
    record_key: str,
) -> Path:
    if not _SAFE_FAMILY.fullmatch(family_id):
        raise ValueError("Unsafe family_id")
    if not _SHA256.fullmatch(recipe_id):
        raise ValueError("Invalid recipe_id")
    if not _SAFE_RECORD.fullmatch(record_key):
        raise ValueError("Invalid record_key")
    return Path(root) / family_id / recipe_id / f"{record_key}.npz"


def logical_payload_sha256(payload: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        array = np.asarray(payload[name])
        if array.dtype == object:
            raise CacheValidationError(f"Object array is forbidden: {name}")
        contiguous = np.ascontiguousarray(array)
        components = (
            name.encode("utf-8"),
            contiguous.dtype.str.encode("ascii"),
            canonical_json(list(contiguous.shape)).encode("ascii"),
            contiguous.tobytes(order="C"),
        )
        for component in components:
            digest.update(len(component).to_bytes(8, "big"))
            digest.update(component)
    return digest.hexdigest()


def _text_scalar(payload: Mapping[str, np.ndarray], key: str) -> str:
    array = np.asarray(payload[key])
    if array.shape != () or array.dtype.kind not in "US":
        raise CacheValidationError(f"{key} must be a scalar string")
    return str(array.item())


def _canonical_json_value(text: str, key: str):
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CacheValidationError(f"Invalid JSON in {key}") from exc
    if canonical_json(value) != text:
        raise CacheValidationError(f"Noncanonical JSON in {key}")
    return value


def _validate_payload(payload: Mapping[str, np.ndarray]) -> None:
    if set(payload) != _REQUIRED_KEYS:
        raise CacheValidationError("Cache keys do not match schema")
    if any(np.asarray(value).dtype == object for value in payload.values()):
        raise CacheValidationError("Object arrays are forbidden")
    for key in _SCALAR_TEXT_KEYS:
        _text_scalar(payload, key)
    for key in _VECTOR_TEXT_KEYS:
        array = np.asarray(payload[key])
        if array.ndim != 1 or array.dtype.kind not in "US":
            raise CacheValidationError(f"{key} must be a string vector")
    values = np.asarray(payload["values"])
    valid = np.asarray(payload["valid_mask"])
    support = np.asarray(payload["support_count"])
    if values.ndim != 1 or values.dtype != np.dtype(np.float64):
        raise CacheValidationError("values must be float64 vector")
    if valid.ndim != 1 or valid.dtype != np.dtype(bool):
        raise CacheValidationError("valid_mask must be bool vector")
    if support.ndim != 1 or support.dtype != np.dtype(np.int64):
        raise CacheValidationError("support_count must be int64 vector")
    length = len(np.asarray(payload["feature_ids"]))
    if any(
        len(np.asarray(payload[key])) != length
        for key in (
            "values",
            "valid_mask",
            "support_count",
            "missing_reason",
            "feature_units",
        )
    ):
        raise CacheValidationError("Parallel cache vectors differ in length")
    feature_ids = tuple(str(item) for item in np.asarray(payload["feature_ids"]).tolist())
    if any(not item for item in feature_ids) or len(set(feature_ids)) != len(feature_ids):
        raise CacheValidationError("Feature IDs must be nonempty and unique")
    if np.any(support < 0):
        raise CacheValidationError("Support cannot be negative")
    if not np.all(np.isfinite(values[valid])) or not np.all(np.isnan(values[~valid])):
        raise CacheValidationError("Cache values and validity disagree")
    reasons = tuple(str(item) for item in np.asarray(payload["missing_reason"]).tolist())
    if any(not reason for reason, is_valid in zip(reasons, valid) if not is_valid):
        raise CacheValidationError("Invalid values require missing reasons")

    if _text_scalar(payload, "cache_schema") != CACHE_SCHEMA_VERSION:
        raise CacheValidationError("Unknown cache schema")
    if _text_scalar(payload, "completion_status") != COMPLETE_STATUS:
        raise CacheValidationError("Cache is not complete")
    recipe_id = _text_scalar(payload, "recipe_id")
    if not _SHA256.fullmatch(recipe_id):
        raise CacheValidationError("Invalid recipe identity")
    source_digest = _text_scalar(payload, "source_identity_digest")
    family_input_digest = _text_scalar(payload, "family_input_identity_digest")
    payload_digest = _text_scalar(payload, "payload_sha256")
    if not all(_SHA256.fullmatch(item) for item in (source_digest, family_input_digest, payload_digest)):
        raise CacheValidationError("Invalid SHA256 field")

    source_identity = _canonical_json_value(
        _text_scalar(payload, "source_identity_json"), "source_identity_json"
    )
    source_provenance = _canonical_json_value(
        _text_scalar(payload, "source_provenance_json"), "source_provenance_json"
    )
    family_input = _canonical_json_value(
        _text_scalar(payload, "family_input_identity_json"),
        "family_input_identity_json",
    )
    recipe_material = _canonical_json_value(
        _text_scalar(payload, "recipe_material_json"), "recipe_material_json"
    )
    _canonical_json_value(
        _text_scalar(payload, "recipe_provenance_json"), "recipe_provenance_json"
    )
    processing_parameters = _canonical_json_value(
        _text_scalar(payload, "processing_parameters_json"),
        "processing_parameters_json",
    )
    if source_identity.get("digest") != source_digest:
        raise CacheValidationError("Source identity digest disagrees")
    source_material = {key: value for key, value in source_identity.items() if key != "digest"}
    if canonical_sha256(source_material) != source_digest:
        raise CacheValidationError("Source identity is internally inconsistent")
    if not isinstance(source_provenance, dict) or set(source_provenance) != {
        "source_path",
        "source_mtime_ns",
        "source_mode",
    }:
        raise CacheValidationError("Source provenance fields do not match schema")
    if (
        not isinstance(source_provenance["source_path"], str)
        or not source_provenance["source_path"]
        or not isinstance(source_provenance["source_mtime_ns"], int)
        or not isinstance(source_provenance["source_mode"], int)
    ):
        raise CacheValidationError("Source provenance values are invalid")
    if family_input.get("digest") != family_input_digest:
        raise CacheValidationError("Family input identity digest disagrees")
    family_material = {key: value for key, value in family_input.items() if key != "digest"}
    if canonical_sha256(family_material) != family_input_digest:
        raise CacheValidationError("Family input identity is internally inconsistent")
    if hashlib.sha256(
        _text_scalar(payload, "recipe_material_json").encode("utf-8")
    ).hexdigest() != recipe_id:
        raise CacheValidationError("Recipe material digest disagrees")
    if recipe_material.get("family_id") != _text_scalar(payload, "family_id"):
        raise CacheValidationError("Recipe family disagrees")
    if recipe_material.get("family_version") != _text_scalar(payload, "family_version"):
        raise CacheValidationError("Recipe version disagrees")
    if recipe_material.get("processing_parameters") != processing_parameters:
        raise CacheValidationError("Processing parameters disagree with recipe")

    without_digest = {
        key: np.asarray(value)
        for key, value in payload.items()
        if key != "payload_sha256"
    }
    if logical_payload_sha256(without_digest) != payload_digest:
        raise CacheValidationError("Logical payload digest mismatch")


def read_cache(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as stored:
            payload = {name: np.asarray(stored[name]).copy() for name in stored.files}
    except Exception as exc:
        raise CacheValidationError(f"Unreadable cache: {Path(path)}") from exc
    _validate_payload(payload)
    return payload


def decide_resume_action(path: Path, expected: CacheExpectation) -> ResumeAction:
    target = Path(path)
    if not target.exists():
        return ResumeAction.BUILD_NEW
    try:
        payload = read_cache(target)
        matches = (
            _text_scalar(payload, "family_id") == expected.family_id
            and _text_scalar(payload, "family_version") == expected.family_version
            and _text_scalar(payload, "recipe_id") == expected.recipe_id
            and tuple(str(item) for item in payload["feature_ids"].tolist())
            == expected.feature_ids
            and _text_scalar(payload, "source_identity_digest")
            == expected.source_identity_digest
            and _text_scalar(payload, "family_input_identity_digest")
            == expected.family_input_identity_digest
        )
    except (OSError, CacheValidationError, ValueError, TypeError):
        return ResumeAction.STOP_INVALID
    return ResumeAction.SKIP_VERIFIED if matches else ResumeAction.STOP_INVALID


def _payload_for(
    result: FeatureResult,
    recipe: RecipeIdentity,
    family_input: FamilyInputIdentity,
) -> dict[str, np.ndarray]:
    validate_feature_result(result)
    if result.family_id != recipe.family_id or result.family_version != recipe.family_version:
        raise ValueError("FeatureResult and recipe family identity disagree")
    if result.source_identity.digest != family_input.source_identity_digest:
        raise ValueError("FeatureResult source and family input identity disagree")
    payload: dict[str, np.ndarray] = {
        "cache_schema": np.asarray(CACHE_SCHEMA_VERSION),
        "completion_status": np.asarray(COMPLETE_STATUS),
        "family_id": np.asarray(result.family_id),
        "family_version": np.asarray(result.family_version),
        "recipe_id": np.asarray(recipe.recipe_id),
        "recipe_material_json": np.asarray(recipe.material_json),
        "recipe_provenance_json": np.asarray(canonical_json(dict(recipe.provenance))),
        "source_identity_json": np.asarray(result.source_identity.canonical_json()),
        "source_provenance_json": np.asarray(
            canonical_json(
                {
                    "source_path": result.source_identity.source_path,
                    "source_mtime_ns": result.source_identity.source_mtime_ns,
                    "source_mode": result.source_identity.source_mode,
                }
            )
        ),
        "source_identity_digest": np.asarray(result.source_identity.digest),
        "family_input_identity_json": np.asarray(family_input.canonical_json()),
        "family_input_identity_digest": np.asarray(family_input.digest),
        "processing_parameters_json": np.asarray(
            canonical_json(dict(result.processing_parameters))
        ),
        "feature_ids": np.asarray(result.feature_ids, dtype=np.str_),
        "values": np.asarray(result.values, dtype=np.float64),
        "valid_mask": np.asarray(result.valid_mask, dtype=bool),
        "support_count": np.asarray(result.support_count, dtype=np.int64),
        "missing_reason": np.asarray(result.missing_reason, dtype=np.str_),
        "feature_units": np.asarray(result.feature_units, dtype=np.str_),
    }
    payload["payload_sha256"] = np.asarray(logical_payload_sha256(payload))
    _validate_payload(payload)
    return payload


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_cache_atomic(
    path: Path,
    result: FeatureResult,
    recipe: RecipeIdentity,
    family_input: FamilyInputIdentity,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(os.fspath(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload_for(result, recipe, family_input)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp.npz",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        read_cache(temporary)
        os.link(temporary, target)
        read_cache(target)
        _fsync_directory(target.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
