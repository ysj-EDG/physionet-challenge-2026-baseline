"""Selected-family FeatureLab execution over one read-only source record."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from . import registry
from .cache import (
    CacheExpectation,
    ResumeAction,
    cache_path,
    decide_resume_action,
    write_cache_atomic,
)
from .contracts import (
    FamilyInputReviewError,
    FamilyRunResult,
    FamilyRunStatus,
    RecordFatalError,
    RecordRunStatus,
    RecordRunSummary,
    validate_feature_result,
)
from .recipe import build_recipe_identity
from .registry import FamilyExtractor
from .sources import load_source_npz, validate_family_input


def _reason(category: str, error: BaseException) -> str:
    return f"{category}:{type(error).__name__}"


def run_record(
    source_path: Path,
    source_root: Path,
    cache_root: Path,
    family_ids: Sequence[str],
    config_overrides: Mapping[str, Mapping[str, object]],
    extractors: Mapping[str, FamilyExtractor] | None = None,
) -> RecordRunSummary:
    try:
        source = load_source_npz(source_path, source_root)
    except RecordFatalError as exc:
        return RecordRunSummary(
            status=RecordRunStatus.STOP_RECORD,
            source_identity=None,
            family_results=(),
            reason_code=_reason("SOURCE_CONTRACT", exc),
        )

    results: list[FamilyRunResult] = []
    code_root = Path(__file__).resolve().parents[1]
    for family_id in family_ids:
        try:
            spec = registry.get_family(family_id)
        except Exception as exc:
            results.append(
                FamilyRunResult(
                    family_id=family_id,
                    status=FamilyRunStatus.FAILED,
                    reason_code=_reason("REGISTRY", exc),
                )
            )
            continue

        try:
            family_input = validate_family_input(source, spec.input_contract)
        except FamilyInputReviewError as exc:
            results.append(
                FamilyRunResult(
                    family_id=family_id,
                    status=FamilyRunStatus.REQUIRES_INPUT_REVIEW,
                    reason_code=_reason("FAMILY_INPUT", exc),
                )
            )
            continue

        config = dict(spec.default_config)
        config.update(config_overrides.get(family_id, {}))
        try:
            recipe = build_recipe_identity(spec, config, code_root)
        except Exception as exc:
            results.append(
                FamilyRunResult(
                    family_id=family_id,
                    status=FamilyRunStatus.FAILED,
                    reason_code=_reason("RECIPE", exc),
                )
            )
            continue

        ordered_features = tuple(item.feature_id for item in spec.feature_definitions)
        target = cache_path(
            cache_root,
            spec.family_id,
            recipe.recipe_id,
            source.identity.record_key,
        )
        expectation = CacheExpectation(
            family_id=spec.family_id,
            family_version=spec.family_version,
            recipe_id=recipe.recipe_id,
            feature_ids=ordered_features,
            source_identity_digest=source.identity.digest,
            family_input_identity_digest=family_input.digest,
        )
        action = decide_resume_action(target, expectation)
        if action is ResumeAction.SKIP_VERIFIED:
            results.append(
                FamilyRunResult(
                    family_id=family_id,
                    status=FamilyRunStatus.SKIP_VERIFIED,
                    reason_code="CACHE_VERIFIED",
                    cache_path=target,
                )
            )
            continue
        if action is ResumeAction.STOP_INVALID:
            results.append(
                FamilyRunResult(
                    family_id=family_id,
                    status=FamilyRunStatus.STOP_INVALID,
                    reason_code="EXISTING_CACHE_INVALID",
                    cache_path=target,
                )
            )
            continue

        extractor = spec.extractor
        if extractors is not None and family_id in extractors:
            extractor = extractors[family_id]
        if extractor is None:
            results.append(
                FamilyRunResult(
                    family_id=family_id,
                    status=FamilyRunStatus.FAILED,
                    reason_code="COMPUTE:MissingExtractor",
                    cache_path=target,
                )
            )
            continue
        try:
            result = extractor(source, config)
            validate_feature_result(result)
            if result.family_id != spec.family_id or result.family_version != spec.family_version:
                raise ValueError("Extractor family identity mismatch")
            if result.feature_ids != ordered_features:
                raise ValueError("Extractor feature order mismatch")
            if result.source_identity.digest != source.identity.digest:
                raise ValueError("Extractor source identity mismatch")
        except Exception as exc:
            results.append(
                FamilyRunResult(
                    family_id=family_id,
                    status=FamilyRunStatus.FAILED,
                    reason_code=_reason("COMPUTE", exc),
                    cache_path=target,
                )
            )
            continue

        try:
            write_cache_atomic(target, result, recipe, family_input)
        except FileExistsError:
            race_action = decide_resume_action(target, expectation)
            if race_action is ResumeAction.SKIP_VERIFIED:
                results.append(
                    FamilyRunResult(
                        family_id=family_id,
                        status=FamilyRunStatus.SKIP_VERIFIED,
                        reason_code="CONCURRENT_WINNER",
                        cache_path=target,
                    )
                )
            else:
                results.append(
                    FamilyRunResult(
                        family_id=family_id,
                        status=FamilyRunStatus.STOP_INVALID,
                        reason_code="CONCURRENT_TARGET_INVALID",
                        cache_path=target,
                    )
                )
            continue
        except Exception as exc:
            results.append(
                FamilyRunResult(
                    family_id=family_id,
                    status=FamilyRunStatus.FAILED,
                    reason_code=_reason("CACHE_WRITE", exc),
                    cache_path=target,
                )
            )
            continue
        results.append(
            FamilyRunResult(
                family_id=family_id,
                status=FamilyRunStatus.BUILT,
                reason_code="CACHE_PUBLISHED",
                cache_path=target,
                result=result,
            )
        )

    successful = {FamilyRunStatus.BUILT, FamilyRunStatus.SKIP_VERIFIED}
    record_status = (
        RecordRunStatus.COMPLETE
        if all(item.status in successful for item in results)
        else RecordRunStatus.PARTIAL
    )
    return RecordRunSummary(
        status=record_status,
        source_identity=source.identity,
        family_results=tuple(results),
    )
