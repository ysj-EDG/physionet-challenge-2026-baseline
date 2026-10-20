"""Bounded smoke orchestration for FeatureLab Batch 1.

Synthetic cache checks, bounded metadata-only source selection, one real
record build/readback/resume, and the public/private report boundary.
Real patient values stay in the private report only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import registry
from .cache import read_cache
from .contracts import (
    FamilyRunStatus,
    RecordRunStatus,
    SourceContractError,
    SourceIdentity,
    SourceRecord,
    canonical_json,
)
from .registry import FamilyExtractor
from .runner import run_record
from .sources import load_source_npz, validate_family_input

MIN_VALID_EPOCHS = 20
SLEEP_STAGE_CODES = (1.0, 2.0, 3.0, 4.0)
WAKE_STAGE_CODE = 5.0
SYNTHETIC_SITE_ID = "SITE_SMOKE"
SYNTHETIC_RECORD_ID = "smoke-001"
PUBLIC_SOURCE_TYPE = "timegrid_v2.1.0 NPZ"

_PUBLIC_SUMMARY_ALLOWED_KEYS = frozenset(
    {
        "source_type",
        "edf_accessed",
        "record_status",
        "status_counts",
        "families",
        "elapsed_sec",
        "resume_skips",
        "resume_extractor_calls",
        "source_immutable",
    }
)
_PUBLIC_FAMILY_ALLOWED_KEYS = frozenset(
    {
        "family_id",
        "status",
        "dimension",
        "valid_count",
        "missing_count",
        "support_min",
        "support_median",
        "support_max",
        "cache_bytes",
    }
)
_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "record_key",
        "site_id",
        "record_id",
        "source_path",
        "source_sha256",
        "values",
        "thresholds",
        "support_count",
        "feature_ids",
        "identity_mapping",
        "cache_path",
        "recipe_id",
        "sha256",
        "reason",
        "reason_code",
        "exception",
        "error",
        "message",
    }
)


@dataclass(frozen=True)
class SyntheticCacheCheckSummary:
    check_a_build_then_skip: bool
    check_a_zero_extractor_calls_on_resume: bool
    check_a_single_then_six_reuse: bool
    check_a_six_then_single_reuse: bool
    check_b_single_parameter_invalidation: bool
    check_c_missing_family_recovery: bool
    check_d_family_input_review_isolation: bool
    check_d_record_stop_no_execution: bool
    built_count: int
    skipped_count: int
    synthetic_examples: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class PublicFamilySummary:
    family_id: str
    status: str
    dimension: int
    valid_count: int | None
    missing_count: int | None
    support_min: int | None
    support_median: int | None
    support_max: int | None
    cache_bytes: int | None


@dataclass(frozen=True)
class PublicSmokeSummary:
    source_type: str
    edf_accessed: bool
    record_status: str
    status_counts: Mapping[str, int]
    families: tuple[PublicFamilySummary, ...]
    elapsed_sec: float
    resume_skips: int
    resume_extractor_calls: int
    source_immutable: bool

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "edf_accessed": self.edf_accessed,
            "record_status": self.record_status,
            "status_counts": dict(self.status_counts),
            "families": [dataclasses.asdict(item) for item in self.families],
            "elapsed_sec": self.elapsed_sec,
            "resume_skips": self.resume_skips,
            "resume_extractor_calls": self.resume_extractor_calls,
            "source_immutable": self.source_immutable,
        }


def assert_public_summary_safe(mapping: Mapping[str, Any]) -> None:
    if not isinstance(mapping, Mapping):
        raise ValueError("Public summary must be a mapping")
    unknown = set(mapping) - _PUBLIC_SUMMARY_ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Forbidden public summary keys: {sorted(unknown)}")
    _reject_forbidden_nested_keys(mapping)
    status_counts = mapping.get("status_counts", {})
    if not isinstance(status_counts, Mapping):
        raise ValueError("status_counts must be a mapping")
    allowed_statuses = {status.value for status in FamilyRunStatus}
    unknown_statuses = set(status_counts) - allowed_statuses
    if unknown_statuses:
        raise ValueError(f"Unknown public status keys: {sorted(unknown_statuses)}")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in status_counts.values()
    ):
        raise ValueError("Public status counts must be nonnegative integers")
    for family in mapping.get("families", ()):
        if not isinstance(family, Mapping):
            raise ValueError("Family summaries must be mappings")
        unknown_family = set(family) - _PUBLIC_FAMILY_ALLOWED_KEYS
        if unknown_family:
            raise ValueError(f"Forbidden family summary keys: {sorted(unknown_family)}")
        for key in family:
            if key in _PUBLIC_FORBIDDEN_KEYS:
                raise ValueError(f"Forbidden family summary key: {key}")
        for key in (
            "valid_count",
            "missing_count",
            "support_min",
            "support_median",
            "support_max",
            "cache_bytes",
        ):
            value = family.get(key)
            if value is not None and not isinstance(value, int):
                raise ValueError(f"Family summary {key} must be an integer or null")


def _reject_forbidden_nested_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _PUBLIC_FORBIDDEN_KEYS:
                raise ValueError(f"Forbidden nested public summary key: {key}")
            _reject_forbidden_nested_keys(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_forbidden_nested_keys(item)


def _synthetic_payload(
    *,
    site_id: str = SYNTHETIC_SITE_ID,
    record_id: str = SYNTHETIC_RECORD_ID,
    epochs: int = 120,
) -> dict[str, Any]:
    stages = np.resize(
        np.asarray([2.0, 2.0, 4.0, 5.0, 1.0, 3.0], dtype=np.float32), epochs
    )
    return {
        "site_id": np.asarray(site_id),
        "record_id": np.asarray(record_id),
        "extraction_version": np.asarray("timegrid_v2.1.0"),
        "validity_schema_version": np.asarray("validity_v0.2"),
        "X_seq": np.arange(epochs * 483, dtype=np.float32).reshape(epochs, 483),
        "stage_valid": np.ones(epochs, dtype=bool),
        "stage_code_aligned": stages,
        "resp_feature_valid": np.ones((epochs, 14), dtype=bool),
        "epoch_start_sec": np.arange(epochs, dtype=np.float32) * 30.0,
        "epoch_duration_sec": np.full(epochs, 30.0, dtype=np.float32),
        "eeg_channel_available": np.ones(6, dtype=bool),
        "eeg_clean_subsegment_count": np.ones((epochs, 6), dtype=np.int32),
        "eeg_common_clean_subsegment_count": np.ones(epochs, dtype=np.int32),
        "emg_channel_available": np.ones(3, dtype=bool),
        "emg_preprocessing_success": np.ones(3, dtype=bool),
        "emg_epoch_success": np.ones((epochs, 3), dtype=bool),
        "x_static": np.zeros(196, dtype=np.float32),
        "annotation_status": np.asarray("available"),
        "y": np.asarray({"forbidden": "label"}, dtype=object),
    }


def _write_synthetic_source(
    root: Path,
    payload: Mapping[str, Any] | None = None,
    *,
    path_site: str = SYNTHETIC_SITE_ID,
    path_record: str = SYNTHETIC_RECORD_ID,
) -> Path:
    material = dict(payload) if payload is not None else _synthetic_payload()
    path = Path(root) / path_site / f"{path_record}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **material)
    return path


def _counting_extractors(
    specs, calls: list[str]
) -> dict[str, FamilyExtractor]:
    wrappers: dict[str, FamilyExtractor] = {}
    for spec in specs:
        original = spec.extractor
        if original is None:
            continue

        def counted(
            source, config, original=original, family_id=spec.family_id
        ):
            calls.append(family_id)
            return original(source, config)

        wrappers[spec.family_id] = counted
    return wrappers


def _support_ok(source: SourceRecord) -> bool:
    try:
        stages = np.asarray(source.arrays["stage_code_aligned"])
        valid = np.asarray(source.arrays["stage_valid"])
    except KeyError:
        return False
    mask = valid & np.isfinite(stages)
    sleep = int(np.sum(mask & np.isin(stages, SLEEP_STAGE_CODES)))
    wake = int(np.sum(mask & (stages == WAKE_STAGE_CODE)))
    return sleep >= MIN_VALID_EPOCHS and wake >= MIN_VALID_EPOCHS


def select_smoke_source(
    source_root: Path, max_candidates: int = 80
) -> SourceIdentity:
    root = Path(source_root)
    candidates = sorted(
        path for path in root.rglob("*.npz") if path.is_file()
    )
    specs = registry.list_families()
    checked = 0
    for path in candidates:
        if checked >= max_candidates:
            break
        checked += 1
        try:
            source = load_source_npz(path, root)
        except SourceContractError:
            continue
        if not _support_ok(source):
            continue
        if all(
            _family_input_ok(source, spec) for spec in specs
        ):
            return source.identity
    raise SourceContractError(
        f"No smoke candidate passed within {checked} metadata checks"
    )


def _family_input_ok(source: SourceRecord, spec) -> bool:
    try:
        validate_family_input(source, spec.input_contract)
    except Exception:
        return False
    return True


def _statuses(result) -> list[str]:
    return [item.status.value for item in result.family_results]


def _synthetic_examples(result) -> tuple[Mapping[str, Any], ...]:
    examples = []
    for item in result.family_results:
        if item.result is None:
            continue
        for feature_id, value, support in zip(
            item.result.feature_ids,
            item.result.values,
            item.result.support_count,
        ):
            if np.isfinite(value):
                examples.append(
                    {
                        "source": "synthetic",
                        "family_id": item.family_id,
                        "feature_id": feature_id,
                        "value": float(value),
                        "support": int(support),
                        "label": "labeled synthetic example (no patient data)",
                    }
                )
                if len(examples) >= 4:
                    return tuple(examples)
    return tuple(examples)


def run_synthetic_cache_checks(temp_root: Path) -> SyntheticCacheCheckSummary:
    root = Path(temp_root)
    specs = registry.list_families()
    family_ids = tuple(spec.family_id for spec in specs)
    built_total = 0
    skipped_total = 0

    def run(source_path, source_root, cache_root, ids, overrides=None, calls=None):
        wrappers = _counting_extractors(specs, calls) if calls is not None else None
        return run_record(
            source_path, source_root, cache_root, ids, overrides or {},
            extractors=wrappers,
        )

    # Check A: first build, identical resume, request-order reuse.
    calls_a: list[str] = []
    a_source = _write_synthetic_source(root / "A" / "npz")
    a_cache = root / "A" / "cache"
    first = run(a_source, a_source.parent.parent, a_cache, family_ids, calls=calls_a)
    built_total += sum(
        item.status is FamilyRunStatus.BUILT for item in first.family_results
    )
    check_a_build = _statuses(first) == ["BUILT"] * 6
    calls_after_build = len(calls_a)
    again = run(a_source, a_source.parent.parent, a_cache, family_ids, calls=calls_a)
    skipped_total += sum(
        item.status is FamilyRunStatus.SKIP_VERIFIED
        for item in again.family_results
    )
    check_a_skip = _statuses(again) == ["SKIP_VERIFIED"] * 6
    check_a_zero = len(calls_a) == calls_after_build

    a2_source = _write_synthetic_source(root / "A2" / "npz")
    a2_cache = root / "A2" / "cache"
    single = run(a2_source, a2_source.parent.parent, a2_cache, (family_ids[0],))
    combined = run(a2_source, a2_source.parent.parent, a2_cache, family_ids)
    check_a_single_then_six = (
        single.family_results[0].status is FamilyRunStatus.BUILT
        and combined.family_results[0].status is FamilyRunStatus.SKIP_VERIFIED
    )

    a3_source = _write_synthetic_source(root / "A3" / "npz")
    a3_cache = root / "A3" / "cache"
    combined3 = run(a3_source, a3_source.parent.parent, a3_cache, family_ids)
    single3 = run(a3_source, a3_source.parent.parent, a3_cache, (family_ids[0],))
    check_a_six_then_single = (
        all(item.status is FamilyRunStatus.BUILT for item in combined3.family_results)
        and single3.family_results[0].status is FamilyRunStatus.SKIP_VERIFIED
    )

    # Check B: one EEG-tail parameter change invalidates only that recipe.
    b_source = _write_synthetic_source(root / "B" / "npz")
    b_cache = root / "B" / "cache"
    baseline = run(b_source, b_source.parent.parent, b_cache, family_ids)
    baseline_paths = {
        item.family_id: item.cache_path for item in baseline.family_results
    }
    changed = run(
        b_source, b_source.parent.parent, b_cache, family_ids,
        {"eeg_tail": {"minimum_support": 21}},
    )
    check_b = (
        changed.family_results[0].status is FamilyRunStatus.BUILT
        and changed.family_results[0].cache_path != baseline_paths["eeg_tail"]
        and all(
            item.status is FamilyRunStatus.SKIP_VERIFIED
            and item.cache_path == baseline_paths[item.family_id]
            for item in changed.family_results[1:]
        )
    )

    # Check C: removing one family result from a disposable copy rebuilds only it.
    c_source = _write_synthetic_source(root / "C" / "npz")
    c_cache = root / "C" / "cache"
    run(c_source, c_source.parent.parent, c_cache, family_ids)
    c_copy = root / "C" / "cache_copy"
    shutil.copytree(c_cache, c_copy)
    shutil.rmtree(c_copy / "resp_stage")
    recovered = run(c_source, c_source.parent.parent, c_copy, family_ids)
    by_family = {item.family_id: item for item in recovered.family_results}
    check_c = (
        by_family["resp_stage"].status is FamilyRunStatus.BUILT
        and all(
            item.status is FamilyRunStatus.SKIP_VERIFIED
            for family_id, item in by_family.items()
            if family_id != "resp_stage"
        )
    )

    # Check D: family-input review is isolated; record-identity failure stops all.
    d_payload = _synthetic_payload()
    del d_payload["emg_epoch_success"]
    d_source = _write_synthetic_source(root / "D" / "npz", d_payload)
    d_cache = root / "D" / "cache"
    isolated = run(d_source, d_source.parent.parent, d_cache, family_ids)
    by_family = {item.family_id: item for item in isolated.family_results}
    check_d_review = (
        by_family["emg_stage"].status is FamilyRunStatus.REQUIRES_INPUT_REVIEW
        and by_family["emg_stage"].cache_path is None
        and all(
            item.status is FamilyRunStatus.BUILT
            for family_id, item in by_family.items()
            if family_id != "emg_stage"
        )
    )

    calls_d: list[str] = []
    d2_payload = _synthetic_payload(site_id="OTHER")
    d2_source = _write_synthetic_source(root / "D2" / "npz", d2_payload)
    stopped = run(
        d2_source, d2_source.parent.parent, root / "D2" / "cache", family_ids,
        calls=calls_d,
    )
    check_d_stop = (
        stopped.status is RecordRunStatus.STOP_RECORD
        and stopped.family_results == ()
        and not calls_d
    )

    return SyntheticCacheCheckSummary(
        check_a_build_then_skip=check_a_build and check_a_skip,
        check_a_zero_extractor_calls_on_resume=check_a_zero,
        check_a_single_then_six_reuse=check_a_single_then_six,
        check_a_six_then_single_reuse=check_a_six_then_single,
        check_b_single_parameter_invalidation=check_b,
        check_c_missing_family_recovery=check_c,
        check_d_family_input_review_isolation=check_d_review,
        check_d_record_stop_no_execution=check_d_stop,
        built_count=built_total,
        skipped_count=skipped_total,
        synthetic_examples=_synthetic_examples(first),
    )


def _source_snapshot(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = Path(path).stat()
    return {
        "sha256": digest.hexdigest(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mode": stat.st_mode,
    }


def _metrics_from_result(result) -> dict[str, Any]:
    support = np.asarray(result.support_count)
    return {
        "dimension": len(result.feature_ids),
        "valid_count": int(np.sum(result.valid_mask)),
        "missing_count": int(np.sum(~result.valid_mask)),
        "support_min": int(np.min(support)),
        "support_median": int(np.median(support)),
        "support_max": int(np.max(support)),
    }


def _metrics_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    valid = np.asarray(payload["valid_mask"])
    support = np.asarray(payload["support_count"])
    return {
        "dimension": int(len(payload["feature_ids"])),
        "valid_count": int(np.sum(valid)),
        "missing_count": int(np.sum(~valid)),
        "support_min": int(np.min(support)),
        "support_median": int(np.median(support)),
        "support_max": int(np.max(support)),
    }


def _assert_result_matches_payload(
    result, payload: Mapping[str, Any], expected_recipe_id: str
) -> None:
    feature_ids = tuple(str(item) for item in payload["feature_ids"].tolist())
    missing_reason = tuple(
        str(item) for item in payload["missing_reason"].tolist()
    )
    feature_units = tuple(str(item) for item in payload["feature_units"].tolist())
    checks = (
        (feature_ids == result.feature_ids, "feature IDs"),
        (
            np.array_equal(
                np.asarray(payload["values"]),
                np.asarray(result.values),
                equal_nan=True,
            ),
            "values",
        ),
        (
            np.array_equal(payload["valid_mask"], result.valid_mask),
            "valid mask",
        ),
        (
            np.array_equal(payload["support_count"], result.support_count),
            "support count",
        ),
        (missing_reason == result.missing_reason, "missing reasons"),
        (feature_units == result.feature_units, "feature units"),
        (
            str(np.asarray(payload["processing_parameters_json"]).item())
            == canonical_json(dict(result.processing_parameters)),
            "processing parameters",
        ),
        (
            str(np.asarray(payload["recipe_id"]).item()) == expected_recipe_id,
            "recipe identity",
        ),
    )
    for matches, field in checks:
        if not matches:
            raise SourceContractError(f"Cache readback disagrees on {field}")


def run_real_smoke(
    source_path: Path,
    source_root: Path,
    cache_root: Path,
    private_report_path: Path,
) -> PublicSmokeSummary:
    source_path = Path(source_path)
    source_root = Path(source_root)
    cache_root = Path(cache_root)
    private_path = Path(private_report_path)
    repo_root = Path(__file__).resolve().parents[2]
    if private_path.resolve().is_relative_to(repo_root.resolve()):
        raise ValueError("Private report must live outside the repository")

    before = _source_snapshot(source_path)
    started = time.monotonic()
    build = run_record(
        source_path, source_root, cache_root,
        tuple(spec.family_id for spec in registry.list_families()),
        {},
    )
    elapsed = round(time.monotonic() - started, 3)

    if build.source_identity is None or not build.family_results:
        raise SourceContractError(
            f"Real smoke record stopped before computation: {build.reason_code}"
        )

    planned = {
        spec.family_id: len(spec.feature_definitions)
        for spec in registry.list_families()
    }
    readback: dict[str, Mapping[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for item in build.family_results:
        if item.cache_path is not None and item.cache_path.exists():
            payload = read_cache(item.cache_path)
            readback[item.family_id] = payload
        if item.result is not None:
            metrics[item.family_id] = _metrics_from_result(item.result)
    for family_id, payload in readback.items():
        if family_id in metrics:
            item = next(
                candidate
                for candidate in build.family_results
                if candidate.family_id == family_id
            )
            _assert_result_matches_payload(
                item.result, payload, item.cache_path.parent.name
            )
            stored = _metrics_from_payload(payload)
            if stored != metrics[family_id]:
                raise SourceContractError(
                    f"Cache readback metrics disagree for {family_id}"
                )
        else:
            metrics[family_id] = _metrics_from_payload(payload)

    calls: list[str] = []
    wrappers = _counting_extractors(registry.list_families(), calls)
    resume = run_record(
        source_path, source_root, cache_root,
        tuple(spec.family_id for spec in registry.list_families()),
        {},
        extractors=wrappers,
    )

    after = _source_snapshot(source_path)
    immutable = before == after
    if not immutable:
        raise SourceContractError("Source NPZ changed during the smoke")

    diagnostic = load_source_npz(source_path, source_root)
    stages = np.asarray(diagnostic.arrays["stage_code_aligned"])
    stage_valid = np.asarray(diagnostic.arrays["stage_valid"])
    stage_mask = stage_valid & np.isfinite(stages)
    stage_counts = {
        str(int(code)): int(count)
        for code, count in zip(
            *np.unique(stages[stage_mask], return_counts=True)
        )
    }
    starts = np.asarray(diagnostic.arrays["epoch_start_sec"])
    time_range = [float(np.min(starts)), float(np.max(starts))]

    _write_private_report(
        private_path,
        build,
        resume,
        calls,
        before,
        after,
        immutable,
        elapsed,
        metrics,
        stage_counts,
        time_range,
    )
    return _public_summary(build, resume, calls, elapsed, immutable, planned)


def _write_private_report(
    path: Path,
    build,
    resume,
    calls: list[str],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    immutable: bool,
    elapsed: float,
    metrics: Mapping[str, Mapping[str, Any]],
    stage_counts: Mapping[str, int],
    time_range: list[float],
) -> None:
    identity = build.source_identity
    family_reports = []
    for item in build.family_results:
        cache_path = item.cache_path
        recipe_id = cache_path.parent.name if cache_path is not None else None
        family_metrics = metrics.get(item.family_id)
        examples = []
        if item.result is not None:
            for feature_id, value, support in zip(
                item.result.feature_ids,
                item.result.values,
                item.result.support_count,
            ):
                if np.isfinite(value):
                    examples.append(
                        {
                            "feature_id": feature_id,
                            "value": float(value),
                            "support": int(support),
                        }
                    )
                    if len(examples) >= 3:
                        break
        family_reports.append(
            {
                "family_id": item.family_id,
                "status": item.status.value,
                "reason_code": item.reason_code,
                "recipe_id": recipe_id,
                "cache_path": str(cache_path) if cache_path is not None else None,
                "dimension": (
                    family_metrics["dimension"] if family_metrics else None
                ),
                "valid_count": (
                    family_metrics["valid_count"] if family_metrics else None
                ),
                "missing_count": (
                    family_metrics["missing_count"] if family_metrics else None
                ),
                "support_min": (
                    family_metrics["support_min"] if family_metrics else None
                ),
                "support_median": (
                    family_metrics["support_median"] if family_metrics else None
                ),
                "support_max": (
                    family_metrics["support_max"] if family_metrics else None
                ),
                "cache_bytes": (
                    cache_path.stat().st_size
                    if cache_path is not None and cache_path.exists()
                    else None
                ),
                "example_values": examples,
            }
        )
    report = {
        "report_kind": "featurelab_batch1_real_smoke_private",
        "source_path": identity.source_path,
        "record_key": identity.record_key,
        "site_id": identity.site_id,
        "record_id": identity.record_id,
        "source_sha256_before": before["sha256"],
        "source_sha256_after": after["sha256"],
        "source_stat_before": {
            "size": before["size"],
            "mtime_ns": before["mtime_ns"],
            "mode": before["mode"],
        },
        "source_stat_after": {
            "size": after["size"],
            "mtime_ns": after["mtime_ns"],
            "mode": after["mode"],
        },
        "source_immutable": immutable,
        "elapsed_sec": elapsed,
        "family_reports": family_reports,
        "resume": {
            "statuses": [item.status.value for item in resume.family_results],
            "extractor_calls": len(calls),
        },
        "stage_epoch_counts": stage_counts,
        "time_range_sec": time_range,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(report) + "\n", encoding="utf-8")


def _public_summary(
    build,
    resume,
    calls: list[str],
    elapsed: float,
    immutable: bool,
    planned: Mapping[str, int],
) -> PublicSmokeSummary:
    families = []
    for item in build.family_results:
        family_metrics = None
        if item.result is not None:
            family_metrics = _metrics_from_result(item.result)
        cache_path = item.cache_path
        families.append(
            PublicFamilySummary(
                family_id=item.family_id,
                status=item.status.value,
                dimension=planned.get(item.family_id, 0),
                valid_count=family_metrics["valid_count"] if family_metrics else None,
                missing_count=(
                    family_metrics["missing_count"] if family_metrics else None
                ),
                support_min=family_metrics["support_min"] if family_metrics else None,
                support_median=(
                    family_metrics["support_median"] if family_metrics else None
                ),
                support_max=family_metrics["support_max"] if family_metrics else None,
                cache_bytes=(
                    cache_path.stat().st_size
                    if cache_path is not None and cache_path.exists()
                    else None
                ),
            )
        )
    status_counts = {
        status: sum(item.status == status for item in build.family_results)
        for status in FamilyRunStatus
    }
    return PublicSmokeSummary(
        source_type=PUBLIC_SOURCE_TYPE,
        edf_accessed=False,
        record_status=build.status.value,
        status_counts=status_counts,
        families=tuple(families),
        elapsed_sec=elapsed,
        resume_skips=sum(
            item.status is FamilyRunStatus.SKIP_VERIFIED
            for item in resume.family_results
        ),
        resume_extractor_calls=len(calls),
        source_immutable=immutable,
    )
