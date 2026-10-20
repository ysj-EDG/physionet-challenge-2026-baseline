"""Read-only adapters for frozen FeatureLab source NPZ files."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

from .contracts import (
    FamilyInputContract,
    FamilyInputIdentity,
    FamilyInputReviewError,
    ObservedField,
    RecordKeyCollisionError,
    SourceContractError,
    SourceIdentity,
    SourceRecord,
    canonical_json,
    canonical_sha256,
)

SOURCE_NAMESPACE = "timegrid_v2.1.0"
EXPECTED_EXTRACTION_SCHEMA = "timegrid_v2.1.0"
EXPECTED_VALIDITY_SCHEMA = "validity_v0.2"
IDENTITY_FIELDS = (
    "site_id", "record_id", "extraction_version", "validity_schema_version"
)
FORBIDDEN_FIELDS = (
    "y", "label", "labels", "tte", "diagnosis", "prediction", "score"
)
SOURCE_FIELD_ALLOWLIST = (
        "X_seq",
        "X_ecg",
        "x_static",
        "epoch_start_sec",
        "epoch_duration_sec",
        "stage_code_raw",
        "stage_code_aligned",
        "stage_valid",
        "stage_probability",
        "stage_probability_valid",
        "stage_probability_channels_present",
        "eeg_channel_available",
        "eeg_clean_subsegment_count",
        "eeg_total_subsegment_count",
        "eeg_common_clean_subsegment_count",
        "emg_channel_available",
        "emg_preprocessing_success",
        "emg_epoch_success",
        "resp_channel_available",
        "resp_preprocessing_success",
        "resp_feature_valid",
        "arousal_valid",
        "resp_event_valid",
        "limb_event_valid",
        "ecg_channel_available",
        "ecg_signal_duration_sec",
        "hrv_success",
        "hrv_feature_valid",
        "circadian_time_valid",
        "ecg_alignment_valid",
        "annotation_status",
        "annotation_signal_lengths_json",
)


def _record_identity_material(site_id: str, record_id: str) -> dict[str, str]:
    return {"namespace": SOURCE_NAMESPACE, "site_id": site_id, "record_id": record_id}


def _record_key_digest(encoded_identity: bytes) -> str:
    return hashlib.sha256(encoded_identity).hexdigest()


def record_key_for(site_id: str, record_id: str) -> str:
    encoded = canonical_json(_record_identity_material(str(site_id), str(record_id))).encode(
        "utf-8"
    )
    return f"rec_{_record_key_digest(encoded)}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(container: np.lib.npyio.NpzFile, name: str) -> str:
    if name not in container.files:
        raise SourceContractError(f"Missing source identity field: {name}")
    try:
        value = np.asarray(container[name])
    except Exception as exc:
        raise SourceContractError(f"Cannot read source identity field: {name}") from exc
    if value.shape != ():
        raise SourceContractError(f"Source identity field must be scalar: {name}")
    return str(value.item())


def _relative_identity(path: Path, source_root: Path) -> tuple[str, str]:
    try:
        relative = path.relative_to(source_root)
    except ValueError as exc:
        raise SourceContractError("Source path is outside source root") from exc
    if len(relative.parts) != 2 or path.suffix != ".npz":
        raise SourceContractError("Source path must be <site_id>/<record_id>.npz")
    return relative.parent.name, relative.stem


def load_source_npz(path: Path, source_root: Path) -> SourceRecord:
    try:
        resolved_path = Path(path).resolve(strict=True)
        resolved_root = Path(source_root).resolve(strict=True)
        path_site, path_record = _relative_identity(resolved_path, resolved_root)
        before = resolved_path.stat()
        before_sha = _sha256_file(resolved_path)
    except SourceContractError:
        raise
    except OSError as exc:
        raise SourceContractError("Source path, stat, or bytes are unavailable") from exc

    arrays: dict[str, np.ndarray] = {}
    field_errors: dict[str, str] = {}
    try:
        with np.load(resolved_path, allow_pickle=False) as source:
            site_id = _scalar_text(source, "site_id")
            record_id = _scalar_text(source, "record_id")
            extraction_schema = _scalar_text(source, "extraction_version")
            validity_schema = _scalar_text(source, "validity_schema_version")
            if site_id != path_site or record_id != path_record:
                raise SourceContractError("Path and NPZ record identity disagree")
            if extraction_schema != EXPECTED_EXTRACTION_SCHEMA:
                raise SourceContractError("Unexpected extraction schema")
            if validity_schema != EXPECTED_VALIDITY_SCHEMA:
                raise SourceContractError("Unexpected validity schema")
            for name in sorted(set(SOURCE_FIELD_ALLOWLIST).intersection(source.files)):
                if name in FORBIDDEN_FIELDS:
                    continue
                try:
                    arrays[name] = np.asarray(source[name]).copy()
                except Exception as exc:
                    field_errors[name] = type(exc).__name__
    except SourceContractError:
        raise
    except Exception as exc:
        raise SourceContractError(f"Unreadable source NPZ: {resolved_path}") from exc

    try:
        after = resolved_path.stat()
        after_sha = _sha256_file(resolved_path)
    except OSError as exc:
        raise SourceContractError("Source post-read validation failed") from exc
    stable_stat = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if not stable_stat or before_sha != after_sha:
        raise SourceContractError("Source changed while it was being read")

    record_key = record_key_for(site_id, record_id)
    identity_material = {
        "namespace": SOURCE_NAMESPACE,
        "site_id": site_id,
        "record_id": record_id,
        "record_key": record_key,
        "source_size": before.st_size,
        "source_sha256": before_sha,
        "extraction_schema": extraction_schema,
        "validity_schema": validity_schema,
    }
    identity = SourceIdentity(
        **identity_material,
        digest=canonical_sha256(identity_material),
        source_path=str(resolved_path),
        source_mtime_ns=before.st_mtime_ns,
        source_mode=before.st_mode,
    )
    return SourceRecord.create(identity, arrays, field_errors)


def _validate_requirement_shape(
    family_id: str,
    name: str,
    actual_shape: tuple[int, ...],
    expected_shape: tuple[int | str, ...],
    dimensions: dict[str, int],
) -> None:
    if len(actual_shape) != len(expected_shape):
        raise FamilyInputReviewError(
            f"{family_id}: invalid rank for {name}: {actual_shape}"
        )
    for actual, expected in zip(actual_shape, expected_shape):
        if isinstance(expected, int):
            if actual != expected:
                raise FamilyInputReviewError(
                    f"{family_id}: invalid shape for {name}: {actual_shape}"
                )
        elif expected in dimensions and dimensions[expected] != actual:
            raise FamilyInputReviewError(
                f"{family_id}: inconsistent dimension {expected} for {name}"
            )
        else:
            dimensions[expected] = actual


def validate_family_input(
    source: SourceRecord,
    contract: FamilyInputContract,
) -> FamilyInputIdentity:
    dimensions: dict[str, int] = {}
    observed: list[ObservedField] = []
    for requirement in contract.requirements:
        if requirement.name in source.field_errors:
            raise FamilyInputReviewError(
                f"{contract.family_id}: cannot decode required field "
                f"{requirement.name} ({source.field_errors[requirement.name]})"
            )
        if requirement.name not in source.arrays:
            raise FamilyInputReviewError(
                f"{contract.family_id}: missing required field {requirement.name}"
            )
        array = np.asarray(source.arrays[requirement.name])
        _validate_requirement_shape(
            contract.family_id,
            requirement.name,
            tuple(array.shape),
            requirement.shape,
            dimensions,
        )
        accepted = {np.dtype(item).name for item in requirement.accepted_dtypes}
        if array.dtype.name not in accepted:
            raise FamilyInputReviewError(
                f"{contract.family_id}: invalid dtype for {requirement.name}: {array.dtype.name}"
            )
        if requirement.columns:
            try:
                axis = requirement.column_axis % array.ndim
            except ZeroDivisionError as exc:
                raise FamilyInputReviewError(
                    f"{contract.family_id}: columns declared for scalar {requirement.name}"
                ) from exc
            axis_size = array.shape[axis]
            if min(requirement.columns) < 0 or max(requirement.columns) >= axis_size:
                raise FamilyInputReviewError(
                    f"{contract.family_id}: source columns out of range for {requirement.name}"
                )
        observed.append(
            ObservedField(requirement.name, tuple(array.shape), array.dtype.str)
        )

    material = {
        "family_id": contract.family_id,
        "source_identity_digest": source.identity.digest,
        "contract_digest": contract.digest,
        "observed_fields": [field.material() for field in observed],
    }
    return FamilyInputIdentity(
        family_id=contract.family_id,
        source_identity_digest=source.identity.digest,
        contract_digest=contract.digest,
        observed_fields=tuple(observed),
        digest=canonical_sha256(material),
    )


def build_source_inventory(
    paths: Iterable[Path],
    source_root: Path,
) -> tuple[SourceIdentity, ...]:
    identities: list[SourceIdentity] = []
    by_key: dict[str, SourceIdentity] = {}
    seen_paths: set[str] = set()
    for path in paths:
        identity = load_source_npz(path, source_root).identity
        previous = by_key.get(identity.record_key)
        if previous is not None:
            previous_tuple = (previous.namespace, previous.site_id, previous.record_id)
            current_tuple = (identity.namespace, identity.site_id, identity.record_id)
            if previous_tuple != current_tuple:
                raise RecordKeyCollisionError(
                    f"Record key collision for {identity.record_key}"
                )
            if previous.source_sha256 != identity.source_sha256:
                raise SourceContractError("Canonical record identity maps to different files")
        resolved = identity.source_path
        if resolved in seen_paths:
            raise SourceContractError("Duplicate source path in inventory")
        seen_paths.add(resolved)
        by_key[identity.record_key] = identity
        identities.append(identity)
    return tuple(identities)
