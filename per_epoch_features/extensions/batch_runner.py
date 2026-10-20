"""Bounded, resumable batch orchestration for FeatureLab family caches.

The batch layer deliberately delegates all numerical work and cache publication to
``runner.run_record``.  Its responsibilities are path-only inventory construction,
bounded scheduling, private checkpointing, source immutability checks, and strict
readback verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import tempfile
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import registry
from .cache import read_cache
from .contracts import (
    FamilyInputReviewError,
    FamilyRunStatus,
    RecordFatalError,
    RecordRunStatus,
    RecordRunSummary,
    SourceContractError,
    canonical_json,
    canonical_sha256,
)
from .common import STAGE_CODES
from .registry import FamilyExtractor
from .recipe import build_recipe_identity
from .runner import run_record
from .sources import load_source_npz, record_key_for, validate_family_input


BATCH_CHECKPOINT_SCHEMA = "featurelab_batch_checkpoint_v1"
PILOT_RECORD_LIMIT = 12
FULL_AUTHORIZATION_TEXT = "FEATURELAB_FULL_EXTRACTION_APPROVED"
FULL_RECORD_COUNT = 6600
FULL_SITE_COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}
FULL_FAMILY_IDS = (
    "eeg_tail",
    "eeg_temporal",
    "eeg_coherence_stage",
    "caisr_stage",
    "emg_stage",
    "resp_stage",
)
FULL_FAMILY_DIMENSIONS = {
    "eeg_tail": 288,
    "eeg_temporal": 96,
    "eeg_coherence_stage": 90,
    "caisr_stage": 35,
    "emg_stage": 72,
    "resp_stage": 56,
}
FULL_RECIPE_IDS = {
    "eeg_tail": "17f79dd79b79c4cc26ab0477b6c936e28cfb31161e9c0bb86ff44a21fa496054",
    "eeg_temporal": "5dc03f81224b71d166ec736f0517575e982debecab46f06c85ae2c0a1446f8bd",
    "eeg_coherence_stage": "c2c6949634746176d99c204792e5f860384530fee1a35ce54d5f4e41eb676a39",
    "caisr_stage": "6529a32282e29ec44ef5af0290379c2d11f3d70527504a513b5121e6069ed7f3",
    "emg_stage": "29cd008bf82a47a6fa823c400d8bca4b5e71f6a7c68a8b7a7eb7682da74ec91b",
    "resp_stage": "3c1c051e7091cfeac73160a112f048208156cbdac1b5f4e33e24e4ce1825dd35",
}
FULL_FEATURE_ID_COUNT = 637
FULL_FEATURE_ID_DIGEST = "92f071616f961256ac993a68624ef8f44f262a0a6738011e4bb0c4ec405d8039"
FULL_AUTHORIZATION_PATH = Path(
    "/database/home/gaohaojie/workspace/feature-cache-extension/"
    "full_batch/FULL_EXTRACTION_APPROVED.txt"
)
FULL_START_FREE_BYTES = 2 * 1024**3
FULL_RUNTIME_FREE_BYTES = 1024**3
FULL_RSS_STOP_KB = 1024 * 1024


class BatchVerificationError(ValueError):
    """A published cache disagrees with the in-memory result that produced it."""


@dataclass(frozen=True, order=True)
class SourceInventoryEntry:
    """Path-derived identity only; constructing this object never opens the NPZ."""

    relative_path: str
    site_id: str
    record_id: str
    record_key: str

    def source_path(self, source_root: Path) -> Path:
        return Path(source_root) / self.relative_path

    def material(self) -> dict[str, str]:
        return {
            "relative_path": self.relative_path,
            "site_id": self.site_id,
            "record_id": self.record_id,
            "record_key": self.record_key,
        }


@dataclass(frozen=True)
class BatchRecordResult:
    entry: SourceInventoryEntry
    run: RecordRunSummary
    elapsed_sec: float
    cache_bytes: int
    published_bytes: int
    source_immutable: bool | None
    readback_verified: bool
    verification_error: str = ""
    incompatibility_signatures: tuple[tuple[str, str], ...] = ()

@dataclass(frozen=True)
class BatchSafetyStop:
    reason_code: str
    completed_count: int
    abnormal_status: str
    safe_to_resume: bool

    def material(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "completed_count": self.completed_count,
            "abnormal_status": self.abnormal_status,
            "safe_to_resume": self.safe_to_resume,
        }

    @classmethod
    def from_material(cls, material: Mapping[str, Any]) -> "BatchSafetyStop":
        return cls(
            reason_code=str(material["reason_code"]),
            completed_count=int(material["completed_count"]),
            abnormal_status=str(material["abnormal_status"]),
            safe_to_resume=bool(material["safe_to_resume"]),
        )



@dataclass(frozen=True)
class BatchRunSummary:
    records: tuple[BatchRecordResult, ...]
    selected_count: int
    executed_count: int
    requested_family_ids: tuple[str, ...]
    workers: int
    dry_run: bool
    interrupted: bool
    elapsed_sec: float
    peak_rss_kb: int
    status_counts: Mapping[str, int]
    inventory_digest: str
    request_digest: str
    execution_mode: str
    safety_stop: BatchSafetyStop | None
    final_acceptance: bool

@dataclass(frozen=True)
class FullPreflightSummary:
    contract_valid: bool
    inventory_size: int
    site_counts: Mapping[str, int]
    requested_family_ids: tuple[str, ...]
    feature_count: int
    feature_id_digest: str
    recipe_ids: Mapping[str, str]
    available_cache_bytes: int
    peak_rss_kb: int
    authorization_present: bool
    authorization_valid: bool
    execution_permitted: bool
    checkpoint_compatible: bool



@dataclass(frozen=True)
class PilotSelection:
    inventory: tuple[SourceInventoryEntry, ...]
    checked_count: int
    eligible_count: int
    rejection_counts: Mapping[str, int]
    site_counts: Mapping[str, int]


def build_path_inventory(
    source_root: Path,
    paths: Iterable[Path] | None = None,
) -> tuple[SourceInventoryEntry, ...]:
    """Build a deterministic inventory from paths without decoding any NPZ field."""

    root = Path(source_root).resolve(strict=True)
    candidates = (
        tuple(paths)
        if paths is not None
        else tuple(path for path in root.glob("*/*.npz") if path.is_file())
    )
    entries: list[SourceInventoryEntry] = []
    seen_paths: set[str] = set()
    by_key: dict[str, tuple[str, str]] = {}
    for candidate in candidates:
        resolved = Path(candidate).resolve(strict=True)
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise SourceContractError("Inventory source is outside source root") from exc
        if len(relative.parts) != 2 or resolved.suffix != ".npz":
            raise SourceContractError(
                "Inventory source must be <site_id>/<record_id>.npz"
            )
        relative_text = relative.as_posix()
        if relative_text in seen_paths:
            raise SourceContractError("Duplicate source path in batch inventory")
        seen_paths.add(relative_text)
        site_id = relative.parent.name
        record_id = relative.stem
        record_key = record_key_for(site_id, record_id)
        canonical = (site_id, record_id)
        previous = by_key.get(record_key)
        if previous is not None and previous != canonical:
            raise ValueError(f"Record key collision for {record_key}")
        by_key[record_key] = canonical
        entries.append(
            SourceInventoryEntry(relative_text, site_id, record_id, record_key)
        )
    return tuple(sorted(entries))


def round_robin_by_site(
    inventory: Sequence[SourceInventoryEntry],
) -> tuple[SourceInventoryEntry, ...]:
    """Return a stable site-interleaved ordering without reading source contents."""

    grouped: dict[str, deque[SourceInventoryEntry]] = defaultdict(deque)
    for entry in sorted(inventory):
        grouped[entry.site_id].append(entry)
    ordered: list[SourceInventoryEntry] = []
    sites = sorted(grouped)
    while any(grouped.values()):
        for site in sites:
            if grouped[site]:
                ordered.append(grouped[site].popleft())
    return tuple(ordered)


def _pilot_candidate_quality(source, preferred: bool) -> tuple[int, ...] | None:
    stages = np.asarray(source.arrays["stage_code_aligned"])
    valid = np.asarray(source.arrays["stage_valid"])
    lawful = valid & np.isfinite(stages)
    counts = {
        name: int(np.sum(lawful & np.isin(stages, codes)))
        for name, codes in STAGE_CODES.items()
    }
    if (
        counts["Wake"] < 20
        or counts["Sleep"] < 20
        or counts["REM"] < 5
        or counts["N2"] < 5
        or counts["N3"] < 5
    ):
        return None
    eeg = int(np.sum(np.asarray(source.arrays["eeg_channel_available"], dtype=bool)))
    emg_available = np.asarray(
        source.arrays["emg_channel_available"], dtype=bool
    )
    emg_preprocessed = np.asarray(
        source.arrays["emg_preprocessing_success"], dtype=bool
    )
    emg = int(np.sum(emg_available & emg_preprocessed))
    resp = int(
        np.sum(
            np.any(
                np.asarray(source.arrays["resp_feature_valid"], dtype=bool),
                axis=0,
            )
        )
    )
    if eeg == 0 or emg == 0 or resp == 0:
        return None
    stage_floor = min(counts[name] for name in ("Wake", "REM", "N2", "N3"))
    return (
        int(preferred),
        eeg,
        emg,
        resp,
        stage_floor,
        counts["All-stage"],
    )


def select_pilot_inventory(
    inventory: Sequence[SourceInventoryEntry],
    source_root: Path,
    *,
    max_candidates: int = 80,
    max_records: int = PILOT_RECORD_LIMIT,
    per_site: int = 2,
    preferred_paths: Sequence[Path] = (),
) -> PilotSelection:
    """Select a deterministic label-free cross-site pilot within hard bounds."""

    if not 1 <= max_candidates <= 80:
        raise ValueError("max_candidates must be in [1, 80]")
    if not 1 <= max_records <= PILOT_RECORD_LIMIT:
        raise ValueError(f"max_records must be in [1, {PILOT_RECORD_LIMIT}]")
    if per_site <= 0:
        raise ValueError("per_site must be positive")
    root = Path(source_root).resolve(strict=True)
    ordered = round_robin_by_site(inventory)
    by_relative = {entry.relative_path: entry for entry in ordered}
    preferred_relatives: list[str] = []
    for path in preferred_paths:
        resolved = Path(path).resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise SourceContractError(
                "Preferred pilot source is outside source root"
            ) from exc
        if relative not in by_relative:
            raise SourceContractError("Preferred pilot source is absent from inventory")
        if relative not in preferred_relatives:
            preferred_relatives.append(relative)
    preferred_set = set(preferred_relatives)
    candidates = tuple(
        [by_relative[relative] for relative in preferred_relatives]
        + [entry for entry in ordered if entry.relative_path not in preferred_set]
    )
    specs = registry.list_families()
    eligible: list[tuple[tuple[int, ...], SourceInventoryEntry]] = []
    rejections: Counter[str] = Counter()
    checked = 0
    for entry in candidates:
        if checked >= max_candidates:
            break
        checked += 1
        try:
            source = load_source_npz(entry.source_path(root), root)
        except RecordFatalError:
            rejections["SOURCE_CONTRACT"] += 1
            continue
        failed_family = None
        for spec in specs:
            try:
                validate_family_input(source, spec.input_contract)
            except FamilyInputReviewError:
                failed_family = spec.family_id
                break
        if failed_family is not None:
            rejections[f"FAMILY_INPUT:{failed_family}"] += 1
            continue
        quality = _pilot_candidate_quality(
            source, entry.relative_path in preferred_set
        )
        if quality is None:
            rejections["SUPPORT_OR_MODALITY"] += 1
            continue
        eligible.append((quality, entry))

    grouped: dict[str, list[tuple[tuple[int, ...], SourceInventoryEntry]]] = (
        defaultdict(list)
    )
    for quality, entry in eligible:
        grouped[entry.site_id].append((quality, entry))
    shortlists: list[SourceInventoryEntry] = []
    for site in sorted(grouped):
        ranked = sorted(
            grouped[site],
            key=lambda item: (
                tuple(-component for component in item[0]),
                item[1].relative_path,
            ),
        )
        shortlists.extend(entry for _quality, entry in ranked[:per_site])
    selected = round_robin_by_site(shortlists)[:max_records]
    site_counts = Counter(entry.site_id for entry in selected)
    return PilotSelection(
        inventory=selected,
        checked_count=checked,
        eligible_count=len(eligible),
        rejection_counts=dict(sorted(rejections.items())),
        site_counts=dict(sorted(site_counts.items())),
    )


def assert_result_payload_equal(
    result,
    payload: Mapping[str, Any],
    expected_recipe_id: str,
) -> None:
    """Compare every persisted FeatureResult field plus family/source/recipe IDs."""

    if result is None:
        raise BatchVerificationError("Missing in-memory FeatureResult")
    scalar_checks = (
        (str(np.asarray(payload["family_id"]).item()), result.family_id, "family ID"),
        (
            str(np.asarray(payload["family_version"]).item()),
            result.family_version,
            "family version",
        ),
        (
            str(np.asarray(payload["recipe_id"]).item()),
            expected_recipe_id,
            "recipe identity",
        ),
        (
            str(np.asarray(payload["source_identity_json"]).item()),
            result.source_identity.canonical_json(),
            "source identity",
        ),
        (
            str(np.asarray(payload["processing_parameters_json"]).item()),
            canonical_json(dict(result.processing_parameters)),
            "processing parameters",
        ),
    )
    for observed, expected, field in scalar_checks:
        if observed != expected:
            raise BatchVerificationError(f"Cache readback disagrees on {field}")
    sequence_checks = (
        (
            tuple(str(item) for item in payload["feature_ids"].tolist()),
            result.feature_ids,
            "feature IDs",
        ),
        (
            tuple(str(item) for item in payload["missing_reason"].tolist()),
            result.missing_reason,
            "missing reasons",
        ),
        (
            tuple(str(item) for item in payload["feature_units"].tolist()),
            result.feature_units,
            "feature units",
        ),
    )
    for observed, expected, field in sequence_checks:
        if observed != expected:
            raise BatchVerificationError(f"Cache readback disagrees on {field}")
    array_checks = (
        ("values", result.values, True),
        ("valid_mask", result.valid_mask, False),
        ("support_count", result.support_count, False),
    )
    for field, expected, equal_nan in array_checks:
        if not np.array_equal(
            np.asarray(payload[field]), np.asarray(expected), equal_nan=equal_nan
        ):
            raise BatchVerificationError(f"Cache readback disagrees on {field}")


def _source_snapshot(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = Path(path).stat()
    return {
        "sha256": digest.hexdigest(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mode": stat.st_mode,
    }


def _stopped_record(reason: str) -> RecordRunSummary:
    return RecordRunSummary(
        status=RecordRunStatus.STOP_RECORD,
        source_identity=None,
        family_results=(),
        reason_code=reason,
    )


def _error_fingerprint(
    scope: str,
    error: BaseException,
    source_path: Path,
) -> str:
    detail = str(error)
    for value in {str(source_path), str(source_path.resolve())}:
        detail = detail.replace(value, "<SOURCE>")
    return canonical_sha256(
        {
            "scope": scope,
            "error_type": type(error).__name__,
            "detail": " ".join(detail.split()),
        }
    )


def _diagnose_incompatibilities(
    run: RecordRunSummary,
    source_path: Path,
    source_root: Path,
) -> tuple[tuple[str, str], ...]:
    affected = tuple(
        item.family_id
        for item in run.family_results
        if item.status is FamilyRunStatus.REQUIRES_INPUT_REVIEW
    )
    if run.status is not RecordRunStatus.STOP_RECORD and not affected:
        return ()
    try:
        source = load_source_npz(source_path, source_root)
    except RecordFatalError as exc:
        if run.status is RecordRunStatus.STOP_RECORD:
            return (("source", _error_fingerprint("source", exc, source_path)),)
        return ()
    if run.status is RecordRunStatus.STOP_RECORD:
        return ()

    signatures: list[tuple[str, str]] = []
    for family_id in affected:
        try:
            spec = registry.get_family(family_id)
            validate_family_input(source, spec.input_contract)
        except FamilyInputReviewError as exc:
            signatures.append(
                (
                    family_id,
                    _error_fingerprint(
                        f"family:{family_id}", exc, source_path
                    ),
                )
            )
    return tuple(sorted(signatures))


def _execute_entry(
    entry: SourceInventoryEntry,
    source_root: Path,
    cache_root: Path,
    family_ids: tuple[str, ...],
    config_overrides: Mapping[str, Mapping[str, object]],
    extractors: Mapping[str, FamilyExtractor] | None,
) -> BatchRecordResult:
    source_path = entry.source_path(source_root)
    started = time.monotonic()
    try:
        before = _source_snapshot(source_path)
    except Exception as exc:
        return BatchRecordResult(
            entry,
            _stopped_record(f"BATCH_SOURCE_BEFORE:{type(exc).__name__}"),
            round(time.monotonic() - started, 6),
            0,
            0,
            None,
            False,
            f"SOURCE_BEFORE:{type(exc).__name__}",
        )
    run = run_record(
        source_path,
        source_root,
        cache_root,
        family_ids,
        config_overrides,
        extractors=extractors,
    )
    incompatibility_signatures = _diagnose_incompatibilities(
        run, source_path, source_root
    )
    try:
        after = _source_snapshot(source_path)
        immutable = before == after
    except Exception as exc:
        after = None
        immutable = False
        source_error = f"SOURCE_AFTER:{type(exc).__name__}"
    else:
        source_error = "" if immutable else "SOURCE_CHANGED_DURING_BATCH"

    cache_bytes = 0
    published_bytes = 0
    verification_errors: list[str] = []
    for family in run.family_results:
        if (
            family.cache_path is None
            or family.status
            not in {
                FamilyRunStatus.BUILT,
                FamilyRunStatus.SKIP_VERIFIED,
                FamilyRunStatus.STOP_INVALID,
            }
        ):
            continue
        try:
            payload = read_cache(family.cache_path)
            size = family.cache_path.stat().st_size
            cache_bytes += size
            if family.status is FamilyRunStatus.BUILT:
                assert_result_payload_equal(
                    family.result, payload, family.cache_path.parent.name
                )
                published_bytes += size
        except Exception as exc:
            verification_errors.append(
                f"{family.family_id}:{type(exc).__name__}"
            )
    if source_error:
        verification_errors.append(source_error)
    return BatchRecordResult(
        entry=entry,
        run=run,
        elapsed_sec=round(time.monotonic() - started, 6),
        cache_bytes=cache_bytes,
        published_bytes=published_bytes,
        source_immutable=immutable,
        readback_verified=not verification_errors,
        verification_error=";".join(verification_errors),
        incompatibility_signatures=incompatibility_signatures,
    )


def _request_material(
    family_ids: tuple[str, ...],
    config_overrides: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    return {
        "family_ids": list(family_ids),
        "config_overrides": {
            family: dict(config_overrides[family])
            for family in sorted(config_overrides)
        },
    }


def _checkpoint_base(
    inventory: Sequence[SourceInventoryEntry],
    inventory_digest: str,
) -> dict[str, Any]:
    return {
        "schema": BATCH_CHECKPOINT_SCHEMA,
        "inventory_digest": inventory_digest,
        "inventory": [entry.material() for entry in inventory],
        "requests": {},
    }


def _load_checkpoint(
    path: Path,
    inventory: Sequence[SourceInventoryEntry],
    inventory_digest: str,
) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return _checkpoint_base(inventory, inventory_digest)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SourceContractError("Unreadable batch checkpoint") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != BATCH_CHECKPOINT_SCHEMA
        or payload.get("inventory_digest") != inventory_digest
        or payload.get("inventory") != [entry.material() for entry in inventory]
        or not isinstance(payload.get("requests"), dict)
    ):
        raise SourceContractError("Batch checkpoint does not match inventory")
    return payload


def _atomic_write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(
            target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _checkpoint_record(result: BatchRecordResult) -> dict[str, Any]:
    return {
        "record_status": result.run.status.value,
        "reason_code": result.run.reason_code,
        "family_statuses": {
            item.family_id: {
                "status": item.status.value,
                "reason_code": item.reason_code,
            }
            for item in result.run.family_results
        },
        "elapsed_sec": result.elapsed_sec,
        "cache_bytes": result.cache_bytes,
        "published_bytes": result.published_bytes,
        "source_immutable": result.source_immutable,
        "readback_verified": result.readback_verified,
        "verification_error": result.verification_error,
        "incompatibility_signatures": dict(result.incompatibility_signatures),
    }


def _peak_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _available_cache_bytes(cache_root: Path) -> int:
    probe = Path(cache_root).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def _authorization_state(
    full_authorization_file: Path | None,
    *,
    require_authorization: bool,
) -> tuple[bool, bool]:
    if full_authorization_file is None:
        if require_authorization:
            raise PermissionError("Full extraction requires an authorization file")
        raise PermissionError("Full preflight requires the frozen authorization path")
    requested = Path(full_authorization_file).resolve()
    expected = Path(FULL_AUTHORIZATION_PATH).resolve()
    if requested != expected:
        raise PermissionError("Full extraction authorization path is not frozen")
    if not requested.exists():
        if require_authorization:
            raise PermissionError("Full extraction authorization is unavailable")
        return False, False
    try:
        authorization = requested.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PermissionError("Full extraction authorization is unavailable") from exc
    if authorization != FULL_AUTHORIZATION_TEXT:
        raise PermissionError("Full extraction authorization is invalid")
    return True, True


def _live_full_contract() -> dict[str, Any]:
    specs = registry.list_families()
    family_ids = tuple(spec.family_id for spec in specs)
    dimensions = {
        spec.family_id: len(spec.feature_definitions) for spec in specs
    }
    feature_ids = tuple(
        definition.feature_id
        for spec in specs
        for definition in spec.feature_definitions
    )
    code_root = Path(__file__).resolve().parents[1]
    recipe_ids = {
        spec.family_id: build_recipe_identity(
            spec, dict(spec.default_config), code_root
        ).recipe_id
        for spec in specs
    }
    return {
        "family_ids": family_ids,
        "dimensions": dimensions,
        "feature_ids": feature_ids,
        "feature_id_digest": canonical_sha256(list(feature_ids)),
        "recipe_ids": recipe_ids,
    }


def _validate_full_request(
    inventory: Sequence[SourceInventoryEntry],
    family_ids: tuple[str, ...],
    config_overrides: Mapping[str, Mapping[str, object]],
    *,
    max_records: int,
    workers: int,
    full_authorization_file: Path | None,
    expected_inventory_size: int | None,
    extractors: Mapping[str, FamilyExtractor] | None,
    require_authorization: bool,
) -> dict[str, Any]:
    authorization_present, authorization_valid = _authorization_state(
        full_authorization_file,
        require_authorization=require_authorization,
    )
    if max_records != FULL_RECORD_COUNT:
        raise ValueError("Full mode max_records must equal 6600")
    if expected_inventory_size != FULL_RECORD_COUNT:
        raise ValueError("Full mode expected_inventory_size must equal 6600")
    if len(inventory) != FULL_RECORD_COUNT:
        raise ValueError("Full inventory must contain exactly 6600 records")
    observed_sites = dict(sorted(Counter(item.site_id for item in inventory).items()))
    if observed_sites != FULL_SITE_COUNTS:
        raise ValueError("Full inventory site distribution differs from frozen contract")
    if family_ids != FULL_FAMILY_IDS:
        raise ValueError("Full family set and order differ from frozen contract")
    if config_overrides:
        raise ValueError("Full mode requires FamilySpec default parameters")
    if workers != 1:
        raise ValueError("First Full extraction requires workers=1")
    if extractors is not None:
        raise ValueError("Full extraction forbids extractor overrides")

    live = _live_full_contract()
    if live["family_ids"] != FULL_FAMILY_IDS:
        raise ValueError("Live family registry differs from frozen contract")
    if live["dimensions"] != FULL_FAMILY_DIMENSIONS:
        raise ValueError("Live family dimensions differ from frozen contract")
    feature_ids = live["feature_ids"]
    if (
        len(feature_ids) != FULL_FEATURE_ID_COUNT
        or len(set(feature_ids)) != FULL_FEATURE_ID_COUNT
        or live["feature_id_digest"] != FULL_FEATURE_ID_DIGEST
    ):
        raise ValueError("Live ordered Feature IDs differ from frozen contract")
    if live["recipe_ids"] != FULL_RECIPE_IDS:
        raise ValueError("Live Recipe IDs differ from frozen contract")
    return {
        **live,
        "site_counts": observed_sites,
        "authorization_present": authorization_present,
        "authorization_valid": authorization_valid,
    }


def _validate_execution_mode(
    execution_mode: str,
    max_records: int,
) -> None:
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    if execution_mode == "pilot":
        if max_records > PILOT_RECORD_LIMIT:
            raise ValueError(
                f"Pilot mode is limited to {PILOT_RECORD_LIMIT} records"
            )
        return
    if execution_mode != "full":
        raise ValueError("execution_mode must be pilot or full")


def run_full_preflight(
    inventory: Sequence[SourceInventoryEntry],
    source_root: Path,
    cache_root: Path,
    checkpoint_path: Path,
    family_ids: Sequence[str],
    config_overrides: Mapping[str, Mapping[str, object]],
    *,
    max_records: int,
    workers: int,
    full_authorization_file: Path | None,
    expected_inventory_size: int | None,
) -> FullPreflightSummary:
    """Validate the frozen Full contract without decoding sources or writing state."""

    del source_root
    ordered = tuple(sorted(inventory))
    if len({entry.relative_path for entry in ordered}) != len(ordered):
        raise ValueError("Batch inventory contains duplicate paths")
    if len({entry.record_key for entry in ordered}) != len(ordered):
        raise ValueError("Batch inventory contains duplicate record keys")
    requested = tuple(family_ids)
    contract = _validate_full_request(
        ordered,
        requested,
        config_overrides,
        max_records=max_records,
        workers=workers,
        full_authorization_file=full_authorization_file,
        expected_inventory_size=expected_inventory_size,
        extractors=None,
        require_authorization=False,
    )
    available = _available_cache_bytes(cache_root)
    peak = _peak_rss_kb()
    if available < FULL_START_FREE_BYTES:
        raise RuntimeError("Full preflight has less than 2 GiB free")
    if peak > FULL_RSS_STOP_KB:
        raise RuntimeError("Full preflight RSS exceeds the frozen safety threshold")

    inventory_digest = canonical_sha256([item.material() for item in ordered])
    request_material = _request_material(requested, config_overrides)
    request_digest = canonical_sha256(request_material)
    checkpoint_compatible = True
    if Path(checkpoint_path).exists():
        checkpoint = _load_checkpoint(checkpoint_path, ordered, inventory_digest)
        state = checkpoint["requests"].get(request_digest)
        if state is not None and state.get("request") != request_material:
            raise SourceContractError("Checkpoint request digest collision")
    return FullPreflightSummary(
        contract_valid=True,
        inventory_size=len(ordered),
        site_counts=contract["site_counts"],
        requested_family_ids=requested,
        feature_count=len(contract["feature_ids"]),
        feature_id_digest=contract["feature_id_digest"],
        recipe_ids=dict(contract["recipe_ids"]),
        available_cache_bytes=available,
        peak_rss_kb=peak,
        authorization_present=contract["authorization_present"],
        authorization_valid=contract["authorization_valid"],
        execution_permitted=(
            contract["authorization_present"] and contract["authorization_valid"]
        ),
        checkpoint_compatible=checkpoint_compatible,
    )


def _safety_stop(
    reason_code: str,
    completed_count: int,
    abnormal_status: str,
    *,
    safe_to_resume: bool,
) -> BatchSafetyStop:
    return BatchSafetyStop(
        reason_code=reason_code,
        completed_count=completed_count,
        abnormal_status=abnormal_status,
        safe_to_resume=safe_to_resume,
    )


def _immediate_full_stop(
    result: BatchRecordResult,
    completed_count: int,
) -> BatchSafetyStop | None:
    stop_invalid = next(
        (
            item
            for item in result.run.family_results
            if item.status is FamilyRunStatus.STOP_INVALID
        ),
        None,
    )
    if stop_invalid is not None:
        return _safety_stop(
            "CACHE_STOP_INVALID",
            completed_count,
            f"{stop_invalid.family_id}:{stop_invalid.reason_code}",
            safe_to_resume=False,
        )
    if result.source_immutable is not True:
        unavailable = result.source_immutable is None or any(
            token in result.verification_error
            for token in ("SOURCE_BEFORE:", "SOURCE_AFTER:")
        )
        return _safety_stop(
            "SOURCE_SNAPSHOT_UNAVAILABLE" if unavailable else "SOURCE_IMMUTABILITY",
            completed_count,
            result.verification_error or result.run.reason_code,
            safe_to_resume=False,
        )
    if not result.readback_verified:
        return _safety_stop(
            "CACHE_READBACK_FAILED",
            completed_count,
            result.verification_error or "BATCH_VERIFY_FAILED",
            safe_to_resume=False,
        )
    for item in result.run.family_results:
        if item.status is not FamilyRunStatus.FAILED:
            continue
        if item.reason_code.startswith(("REGISTRY:", "RECIPE:")):
            return _safety_stop(
                "FROZEN_RUNTIME_CONTRACT",
                completed_count,
                f"{item.family_id}:{item.reason_code}",
                safe_to_resume=False,
            )
    return None


def _record_systemic_errors(
    result: BatchRecordResult,
    request_state: dict[str, Any],
    completed_count: int,
) -> BatchSafetyStop | None:
    signatures: list[tuple[str, str]] = []
    diagnosed = dict(result.incompatibility_signatures)
    if (
        result.run.status is RecordRunStatus.STOP_RECORD
        and result.run.reason_code.startswith("SOURCE_CONTRACT:")
        and "source" in diagnosed
    ):
        signatures.append(
            (
                "REPEATED_SOURCE_CONTRACT",
                "|".join(
                    (
                        "source",
                        result.entry.site_id,
                        diagnosed["source"],
                    )
                ),
            )
        )
    for item in result.run.family_results:
        if (
            item.status is FamilyRunStatus.REQUIRES_INPUT_REVIEW
            and item.family_id in diagnosed
        ):
            signatures.append(
                (
                    "REPEATED_FAMILY_INPUT",
                    "|".join(
                        (
                            "family",
                            result.entry.site_id,
                            item.family_id,
                            diagnosed[item.family_id],
                        )
                    ),
                )
            )

    counters = request_state.setdefault("systemic_errors", {})
    stop: BatchSafetyStop | None = None
    for reason_code, signature in signatures:
        record_keys = list(counters.get(signature, []))
        if result.entry.record_key not in record_keys:
            record_keys.append(result.entry.record_key)
        counters[signature] = sorted(record_keys)
        if len(record_keys) >= 2 and stop is None:
            stop = _safety_stop(
                reason_code,
                completed_count,
                signature,
                safe_to_resume=False,
            )
    request_state["systemic_errors"] = dict(sorted(counters.items()))
    return stop


def _resource_stop(
    cache_root: Path,
    completed_count: int,
    *,
    startup: bool,
) -> BatchSafetyStop | None:
    available = _available_cache_bytes(cache_root)
    threshold = FULL_START_FREE_BYTES if startup else FULL_RUNTIME_FREE_BYTES
    if available < threshold:
        return _safety_stop(
            "LOW_DISK_STARTUP" if startup else "LOW_DISK_RUNTIME",
            completed_count,
            f"available_bytes={available};required_bytes={threshold}",
            safe_to_resume=True,
        )
    peak = _peak_rss_kb()
    if peak > FULL_RSS_STOP_KB:
        return _safety_stop(
            "HIGH_RSS_STARTUP" if startup else "HIGH_RSS_RUNTIME",
            completed_count,
            f"peak_rss_kb={peak};limit_kb={FULL_RSS_STOP_KB}",
            safe_to_resume=True,
        )
    return None


def _status_counts(results: Sequence[BatchRecordResult]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        if result.run.status is RecordRunStatus.STOP_RECORD:
            counts[RecordRunStatus.STOP_RECORD.value] += 1
        for family in result.run.family_results:
            counts[family.status.value] += 1
        if not result.readback_verified:
            counts["BATCH_VERIFY_FAILED"] += 1
    return dict(sorted(counts.items()))


def _batch_summary(
    records: Sequence[BatchRecordResult],
    *,
    selected_count: int,
    requested: tuple[str, ...],
    workers: int,
    dry_run: bool,
    interrupted: bool,
    started: float | None,
    inventory_digest: str,
    request_digest: str,
    execution_mode: str,
    safety_stop: BatchSafetyStop | None,
) -> BatchRunSummary:
    ordered_records = tuple(sorted(records, key=lambda item: item.entry))
    accepted_records = (
        len(ordered_records) == selected_count
        and all(
            item.run.status is RecordRunStatus.COMPLETE
            and item.source_immutable is True
            and item.readback_verified
            for item in ordered_records
        )
    )
    final_acceptance = (
        safety_stop is None
        and (
            dry_run
            or (
                not interrupted
                and accepted_records
            )
        )
    )
    return BatchRunSummary(
        records=ordered_records,
        selected_count=selected_count,
        executed_count=len(ordered_records),
        requested_family_ids=requested,
        workers=workers,
        dry_run=dry_run,
        interrupted=interrupted or safety_stop is not None,
        elapsed_sec=(
            0.0 if started is None else round(time.monotonic() - started, 6)
        ),
        peak_rss_kb=_peak_rss_kb(),
        status_counts=_status_counts(ordered_records),
        inventory_digest=inventory_digest,
        request_digest=request_digest,
        execution_mode=execution_mode,
        safety_stop=safety_stop,
        final_acceptance=final_acceptance,
    )


def run_batch(
    inventory: Sequence[SourceInventoryEntry],
    source_root: Path,
    cache_root: Path,
    checkpoint_path: Path,
    family_ids: Sequence[str],
    config_overrides: Mapping[str, Mapping[str, object]],
    *,
    max_records: int,
    workers: int = 1,
    dry_run: bool = False,
    stop_after: int | None = None,
    extractors: Mapping[str, FamilyExtractor] | None = None,
    execution_mode: str = "pilot",
    full_authorization_file: Path | None = None,
    expected_inventory_size: int | None = None,
) -> BatchRunSummary:
    """Execute one selected-family request per record under an explicit bound."""

    ordered = tuple(sorted(inventory))
    if len({entry.relative_path for entry in ordered}) != len(ordered):
        raise ValueError("Batch inventory contains duplicate paths")
    if len({entry.record_key for entry in ordered}) != len(ordered):
        raise ValueError("Batch inventory contains duplicate record keys")
    requested = tuple(family_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("family_ids must be nonempty and unique")
    if workers <= 0:
        raise ValueError("workers must be positive")
    _validate_execution_mode(execution_mode, max_records)
    if execution_mode == "full":
        _validate_full_request(
            ordered,
            requested,
            config_overrides,
            max_records=max_records,
            workers=workers,
            full_authorization_file=full_authorization_file,
            expected_inventory_size=expected_inventory_size,
            extractors=extractors,
            require_authorization=True,
        )
    startup_stop = (
        _resource_stop(Path(cache_root), 0, startup=True)
        if execution_mode == "full"
        else None
    )
    selected = ordered[:max_records]
    if stop_after is not None and stop_after < 0:
        raise ValueError("stop_after cannot be negative")
    executable = selected[:stop_after] if stop_after is not None else selected
    interrupted = len(executable) < len(selected)
    inventory_material = [entry.material() for entry in selected]
    inventory_digest = canonical_sha256(inventory_material)
    request_material = _request_material(requested, config_overrides)
    request_digest = canonical_sha256(request_material)
    if dry_run:
        return _batch_summary(
            (),
            selected_count=len(selected),
            requested=requested,
            workers=workers,
            dry_run=True,
            interrupted=False,
            started=None,
            inventory_digest=inventory_digest,
            request_digest=request_digest,
            execution_mode=execution_mode,
            safety_stop=startup_stop,
        )

    checkpoint = _load_checkpoint(checkpoint_path, selected, inventory_digest)
    request_state = checkpoint["requests"].setdefault(
        request_digest,
        {"request": request_material, "records": {}},
    )
    if request_state.get("request") != request_material:
        raise SourceContractError("Checkpoint request digest collision")
    started = time.monotonic()
    completed: list[BatchRecordResult] = []
    safety_stop: BatchSafetyStop | None = None
    if execution_mode == "full":
        request_state.setdefault("systemic_errors", {})
        request_state.setdefault("safety_stop_history", [])
        active_material = request_state.get("active_safety_stop")
        if active_material is not None:
            active = BatchSafetyStop.from_material(active_material)
            if not active.safe_to_resume:
                return _batch_summary(
                    (),
                    selected_count=len(selected),
                    requested=requested,
                    workers=workers,
                    dry_run=False,
                    interrupted=True,
                    started=started,
                    inventory_digest=inventory_digest,
                    request_digest=request_digest,
                    execution_mode=execution_mode,
                    safety_stop=active,
                )
            if startup_stop is None:
                request_state["safety_stop_history"].append(active.material())
                request_state["active_safety_stop"] = None
                _atomic_write_checkpoint(checkpoint_path, checkpoint)
            else:
                request_state["active_safety_stop"] = startup_stop.material()
                _atomic_write_checkpoint(checkpoint_path, checkpoint)
                return _batch_summary(
                    (),
                    selected_count=len(selected),
                    requested=requested,
                    workers=workers,
                    dry_run=False,
                    interrupted=True,
                    started=started,
                    inventory_digest=inventory_digest,
                    request_digest=request_digest,
                    execution_mode=execution_mode,
                    safety_stop=startup_stop,
                )
        elif startup_stop is not None:
            request_state["active_safety_stop"] = startup_stop.material()
            _atomic_write_checkpoint(checkpoint_path, checkpoint)
            return _batch_summary(
                (),
                selected_count=len(selected),
                requested=requested,
                workers=workers,
                dry_run=False,
                interrupted=True,
                started=started,
                inventory_digest=inventory_digest,
                request_digest=request_digest,
                execution_mode=execution_mode,
                safety_stop=startup_stop,
            )


    def persist(result: BatchRecordResult) -> None:
        previous = request_state["records"].get(result.entry.record_key, {})
        material = _checkpoint_record(result)
        material["attempts"] = int(previous.get("attempts", 0)) + 1
        request_state["records"][result.entry.record_key] = material
        request_state["records"] = dict(sorted(request_state["records"].items()))
        _atomic_write_checkpoint(checkpoint_path, checkpoint)

    if workers == 1:
        for entry in executable:
            result = _execute_entry(
                entry,
                Path(source_root),
                Path(cache_root),
                requested,
                config_overrides,
                extractors,
            )
            completed.append(result)
            systemic_stop = (
                _record_systemic_errors(result, request_state, len(completed))
                if execution_mode == "full"
                else None
            )
            persist(result)
            if execution_mode == "full":
                safety_stop = (
                    _immediate_full_stop(result, len(completed))
                    or systemic_stop
                    or _resource_stop(
                        Path(cache_root), len(completed), startup=False
                    )
                )
                if safety_stop is not None:
                    request_state["active_safety_stop"] = safety_stop.material()
                    _atomic_write_checkpoint(checkpoint_path, checkpoint)
                    break
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _execute_entry,
                    entry,
                    Path(source_root),
                    Path(cache_root),
                    requested,
                    config_overrides,
                    extractors,
                ): entry
                for entry in executable
            }
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                persist(result)

    return _batch_summary(
        completed,
        selected_count=len(selected),
        requested=requested,
        workers=workers,
        dry_run=False,
        interrupted=interrupted,
        started=started,
        inventory_digest=inventory_digest,
        request_digest=request_digest,
        execution_mode=execution_mode,
        safety_stop=safety_stop,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-records", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--family", action="append", dest="families")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--execution-mode", choices=("pilot", "full"), default="pilot"
    )
    parser.add_argument("--full-authorization-file", type=Path)
    parser.add_argument("--expected-inventory-size", type=int)
    return parser


def summary_exit_code(summary: BatchRunSummary) -> int:
    if summary.safety_stop is not None:
        return 2
    if (
        summary.execution_mode == "full"
        and not summary.dry_run
        and not summary.final_acceptance
    ):
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inventory = build_path_inventory(args.source_root)
        family_ids = tuple(args.families or (
            spec.family_id for spec in registry.list_families()
        ))
        if args.preflight_only:
            if args.execution_mode != "full" or args.dry_run:
                raise ValueError(
                    "--preflight-only requires full mode and is distinct from --dry-run"
                )
            report = run_full_preflight(
                inventory,
                args.source_root,
                args.cache_root,
                args.checkpoint,
                family_ids,
                {},
                max_records=args.max_records,
                workers=args.workers,
                full_authorization_file=args.full_authorization_file,
                expected_inventory_size=args.expected_inventory_size,
            )
            print(
                canonical_json(
                    {
                        "preflight_only": True,
                        "contract_valid": report.contract_valid,
                        "inventory_size": report.inventory_size,
                        "site_counts": dict(report.site_counts),
                        "family_ids": list(report.requested_family_ids),
                        "feature_count": report.feature_count,
                        "feature_id_digest": report.feature_id_digest,
                        "recipe_ids": dict(report.recipe_ids),
                        "available_cache_bytes": report.available_cache_bytes,
                        "peak_rss_kb": report.peak_rss_kb,
                        "authorization_present": report.authorization_present,
                        "authorization_valid": report.authorization_valid,
                        "execution_permitted": report.execution_permitted,
                        "checkpoint_compatible": report.checkpoint_compatible,
                    }
                )
            )
            return 0

        summary = run_batch(
            inventory,
            args.source_root,
            args.cache_root,
            args.checkpoint,
            family_ids,
            {},
            max_records=args.max_records,
            workers=args.workers,
            dry_run=args.dry_run,
            execution_mode=args.execution_mode,
            full_authorization_file=args.full_authorization_file,
            expected_inventory_size=args.expected_inventory_size,
        )
    except (OSError, PermissionError, RuntimeError, ValueError) as exc:
        print(
            canonical_json(
                {
                    "status": "REJECTED",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
        )
        return 2

    print(
        canonical_json(
            {
                "selected_count": summary.selected_count,
                "executed_count": summary.executed_count,
                "workers": summary.workers,
                "dry_run": summary.dry_run,
                "interrupted": summary.interrupted,
                "elapsed_sec": summary.elapsed_sec,
                "peak_rss_kb": summary.peak_rss_kb,
                "status_counts": dict(summary.status_counts),
                "inventory_digest": summary.inventory_digest,
                "request_digest": summary.request_digest,
                "final_acceptance": summary.final_acceptance,
                "safety_stop": (
                    None
                    if summary.safety_stop is None
                    else summary.safety_stop.material()
                ),
            }
        )
    )
    return summary_exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
