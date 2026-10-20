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
    FamilyRunStatus,
    RecordRunStatus,
    RecordRunSummary,
    SourceContractError,
    canonical_json,
    canonical_sha256,
)
from .registry import FamilyExtractor
from .runner import run_record
from .sources import record_key_for


BATCH_CHECKPOINT_SCHEMA = "featurelab_batch_checkpoint_v1"
PILOT_RECORD_LIMIT = 12
FULL_AUTHORIZATION_TEXT = "FEATURELAB_FULL_EXTRACTION_APPROVED"


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
        if family.cache_path is None:
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
    }


def _validate_execution_mode(
    execution_mode: str,
    max_records: int,
    full_authorization_file: Path | None,
    expected_inventory_size: int | None,
    inventory_size: int,
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
    if full_authorization_file is None:
        raise PermissionError("Full extraction requires an authorization file")
    try:
        authorization = Path(full_authorization_file).read_text(
            encoding="utf-8"
        ).strip()
    except OSError as exc:
        raise PermissionError("Full extraction authorization is unavailable") from exc
    if authorization != FULL_AUTHORIZATION_TEXT:
        raise PermissionError("Full extraction authorization is invalid")
    if expected_inventory_size is None:
        raise ValueError("Full mode requires expected_inventory_size")
    if inventory_size != expected_inventory_size:
        raise ValueError("Inventory size differs from explicit full-run expectation")
    if max_records != expected_inventory_size:
        raise ValueError("Full mode max_records must equal expected inventory size")


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
    _validate_execution_mode(
        execution_mode,
        max_records,
        full_authorization_file,
        expected_inventory_size,
        len(ordered),
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
        return BatchRunSummary(
            records=(),
            selected_count=len(selected),
            executed_count=0,
            requested_family_ids=requested,
            workers=workers,
            dry_run=True,
            interrupted=False,
            elapsed_sec=0.0,
            peak_rss_kb=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            status_counts={},
            inventory_digest=inventory_digest,
            request_digest=request_digest,
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
            persist(result)
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

    completed.sort(key=lambda item: item.entry)
    counts: Counter[str] = Counter()
    for result in completed:
        if result.run.status is RecordRunStatus.STOP_RECORD:
            counts[RecordRunStatus.STOP_RECORD.value] += 1
        for family in result.run.family_results:
            counts[family.status.value] += 1
        if not result.readback_verified:
            counts["BATCH_VERIFY_FAILED"] += 1
    return BatchRunSummary(
        records=tuple(completed),
        selected_count=len(selected),
        executed_count=len(completed),
        requested_family_ids=requested,
        workers=workers,
        dry_run=False,
        interrupted=interrupted,
        elapsed_sec=round(time.monotonic() - started, 6),
        peak_rss_kb=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        status_counts=dict(sorted(counts.items())),
        inventory_digest=inventory_digest,
        request_digest=request_digest,
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
    parser.add_argument(
        "--execution-mode", choices=("pilot", "full"), default="pilot"
    )
    parser.add_argument("--full-authorization-file", type=Path)
    parser.add_argument("--expected-inventory-size", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inventory = build_path_inventory(args.source_root)
    family_ids = tuple(args.families or (
        spec.family_id for spec in registry.list_families()
    ))
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
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
