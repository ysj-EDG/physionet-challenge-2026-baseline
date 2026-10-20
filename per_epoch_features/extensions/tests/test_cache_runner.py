from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from extensions.cache import (
    CACHE_SCHEMA_VERSION,
    CacheExpectation,
    CacheValidationError,
    ResumeAction,
    cache_path,
    decide_resume_action,
    logical_payload_sha256,
    read_cache,
    write_cache_atomic,
)
from extensions.contracts import (
    FamilyInputContract,
    FamilyRunStatus,
    FeatureResult,
    FieldRequirement,
    RecordRunStatus,
)
from extensions.recipe import build_recipe_identity
from extensions.registry import FamilySpec, FeatureDefinition
from extensions.runner import run_record
from extensions.sources import load_source_npz, validate_family_input
from extensions.tests.fixtures import source_payload, write_source_npz

CALL_COUNTS = {"a": 0, "b": 0}


def dummy_extract_a(source, config):
    CALL_COUNTS["a"] += 1
    if config.get("raise"):
        raise RuntimeError("dummy a failure")
    scale = float(config.get("scale", 1.0))
    value = float(np.mean(source.arrays["X_seq"][:, 0])) * scale
    return FeatureResult(
        family_id="dummy_a",
        family_version="1.0.0",
        feature_ids=("dummy.a.mean",),
        values=np.asarray([value], dtype=np.float64),
        valid_mask=np.asarray([True], dtype=bool),
        support_count=np.asarray([source.arrays["X_seq"].shape[0]], dtype=np.int64),
        missing_reason=("",),
        feature_units=("ratio",),
        source_identity=source.identity,
        processing_parameters={"scale": scale},
    )


def dummy_extract_b(source, config):
    CALL_COUNTS["b"] += 1
    if config.get("raise"):
        raise RuntimeError("dummy b failure")
    value = float(np.mean(source.arrays["stage_code_aligned"]))
    return FeatureResult(
        family_id="dummy_b",
        family_version="1.0.0",
        feature_ids=("dummy.b.mean",),
        values=np.asarray([value], dtype=np.float64),
        valid_mask=np.asarray([True], dtype=bool),
        support_count=np.asarray([source.arrays["stage_code_aligned"].shape[0]], dtype=np.int64),
        missing_reason=("",),
        feature_units=("stage_code",),
        source_identity=source.identity,
        processing_parameters={},
    )


def _feature(family: str) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=f"dummy.{family}.mean",
        source_field="X_seq" if family == "a" else "stage_code_aligned",
        source_columns=(0,),
        formula_id="dummy_mean",
        parameters={},
        validity_expression="family validity",
        minimum_support=1,
        unit="ratio" if family == "a" else "stage_code",
    )


def _spec_a() -> FamilySpec:
    return FamilySpec(
        family_id="dummy_a",
        family_version="1.0.0",
        extractor_symbol="extensions.tests.test_cache_runner.dummy_extract_a",
        default_config={"scale": 1.0},
        input_contract=FamilyInputContract(
            family_id="dummy_a",
            contract_version="1",
            requirements=(
                FieldRequirement("X_seq", ("T", 483), ("float32",), columns=(0,)),
                FieldRequirement("stage_valid", ("T",), ("bool",)),
            ),
            validity_rules=("stage_valid distinguishes invalid upstream zero",),
        ),
        feature_definitions=(_feature("a"),),
        numerical_symbols=(
            "extensions.tests.test_cache_runner.dummy_extract_a",
            "extensions.contracts.FeatureResult",
        ),
        numerical_constants=("extensions.tests.test_cache_runner.CALL_COUNTS",),
        numerical_dependencies={"numpy": np.__version__},
        extractor=dummy_extract_a,
    )


def _spec_b() -> FamilySpec:
    return FamilySpec(
        family_id="dummy_b",
        family_version="1.0.0",
        extractor_symbol="extensions.tests.test_cache_runner.dummy_extract_b",
        default_config={},
        input_contract=FamilyInputContract(
            family_id="dummy_b",
            contract_version="1",
            requirements=(
                FieldRequirement("stage_code_aligned", ("T",), ("float32",)),
                FieldRequirement("resp_feature_valid", ("T", 14), ("bool",), columns=(0,)),
            ),
            validity_rules=("resp_feature_valid is authoritative",),
        ),
        feature_definitions=(_feature("b"),),
        numerical_symbols=(
            "extensions.tests.test_cache_runner.dummy_extract_b",
            "extensions.contracts.FeatureResult",
        ),
        numerical_constants=("extensions.tests.test_cache_runner.CALL_COUNTS",),
        numerical_dependencies={"numpy": np.__version__},
        extractor=dummy_extract_b,
    )


def _expectation(spec, recipe, source, family_input) -> CacheExpectation:
    return CacheExpectation(
        family_id=spec.family_id,
        family_version=spec.family_version,
        recipe_id=recipe.recipe_id,
        feature_ids=tuple(item.feature_id for item in spec.feature_definitions),
        source_identity_digest=source.identity.digest,
        family_input_identity_digest=family_input.digest,
    )


def _raw_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]).copy() for name in stored.files}


def _rewrite(path: Path, changes: dict[str, object], *, refresh_digest: bool) -> None:
    payload = _raw_payload(path)
    payload.update({name: np.asarray(value) for name, value in changes.items()})
    if refresh_digest:
        without_digest = {name: value for name, value in payload.items() if name != "payload_sha256"}
        payload["payload_sha256"] = np.asarray(logical_payload_sha256(without_digest))
    np.savez_compressed(path, **payload)


class CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        CALL_COUNTS.update(a=0, b=0)

    def _context(self, root: Path):
        source_root = root / "source"
        source = load_source_npz(write_source_npz(source_root), source_root)
        spec = _spec_a()
        family_input = validate_family_input(source, spec.input_contract)
        recipe = build_recipe_identity(spec, {}, Path("per_epoch_features"))
        result = spec.extractor(source, spec.default_config)
        expected = _expectation(spec, recipe, source, family_input)
        target = cache_path(root / "cache", spec.family_id, recipe.recipe_id, source.identity.record_key)
        return source, spec, family_input, recipe, result, expected, target

    def test_roundtrip_build_skip_and_exact_feature_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = self._context(Path(directory))
            _, _, family_input, recipe, result, expected, target = context
            self.assertEqual(decide_resume_action(target, expected), ResumeAction.BUILD_NEW)
            write_cache_atomic(target, result, recipe, family_input)
            payload = read_cache(target)
            self.assertEqual(str(payload["cache_schema"].item()), CACHE_SCHEMA_VERSION)
            source_provenance = json.loads(str(payload["source_provenance_json"].item()))
            self.assertEqual(source_provenance["source_path"], result.source_identity.source_path)
            self.assertEqual(
                source_provenance["source_mtime_ns"],
                result.source_identity.source_mtime_ns,
            )
            self.assertEqual(source_provenance["source_mode"], result.source_identity.source_mode)
            self.assertEqual(tuple(payload["feature_ids"].tolist()), result.feature_ids)
            np.testing.assert_array_equal(payload["values"], result.values)
            self.assertEqual(decide_resume_action(target, expected), ResumeAction.SKIP_VERIFIED)
            wrong_order = replace(expected, feature_ids=tuple(reversed(expected.feature_ids)) + ("extra",))
            self.assertEqual(decide_resume_action(target, wrong_order), ResumeAction.STOP_INVALID)

    def test_schema_completion_family_recipe_source_and_payload_validation(self) -> None:
        mutations = (
            ("schema", {"cache_schema": np.asarray("wrong")}, True, None),
            ("incomplete", {"completion_status": np.asarray("INCOMPLETE")}, True, None),
            ("payload", {"values": np.asarray([999.0])}, False, None),
            ("family", {}, True, {"family_id": "other"}),
            ("recipe", {}, True, {"recipe_id": "f" * 64}),
            ("source", {}, True, {"source_identity_digest": "e" * 64}),
        )
        for label, changes, refresh, expected_changes in mutations:
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                context = self._context(Path(directory))
                _, _, family_input, recipe, result, expected, target = context
                write_cache_atomic(target, result, recipe, family_input)
                if changes:
                    _rewrite(target, changes, refresh_digest=refresh)
                if expected_changes:
                    expected = replace(expected, **expected_changes)
                self.assertEqual(decide_resume_action(target, expected), ResumeAction.STOP_INVALID)

    def test_corrupt_or_truncated_cache_stops_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            *_, expected, target = self._context(Path(directory))
            target.parent.mkdir(parents=True)
            target.write_bytes(b"not-an-npz")
            self.assertEqual(decide_resume_action(target, expected), ResumeAction.STOP_INVALID)

    def test_malformed_source_provenance_raises_cache_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = self._context(Path(directory))
            _, _, family_input, recipe, result, _, target = context
            write_cache_atomic(target, result, recipe, family_input)
            _rewrite(
                target,
                {"source_provenance_json": np.asarray("1")},
                refresh_digest=True,
            )
            with self.assertRaises(CacheValidationError):
                read_cache(target)

    def test_logical_digest_includes_dtype_shape_and_bytes(self) -> None:
        int32 = {"x": np.asarray([1, 2], dtype=np.int32)}
        int64 = {"x": np.asarray([1, 2], dtype=np.int64)}
        reshaped = {"x": np.asarray([[1, 2]], dtype=np.int32)}
        changed = {"x": np.asarray([1, 3], dtype=np.int32)}
        digests = {
            logical_payload_sha256(payload)
            for payload in (int32, int64, reshaped, changed)
        }
        self.assertEqual(len(digests), 4)

    def test_no_clobber_and_owned_temporary_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = self._context(Path(directory))
            _, _, family_input, recipe, result, _, target = context
            write_cache_atomic(target, result, recipe, family_input)
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                write_cache_atomic(target, result, recipe, family_input)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp.npz")), [])

    def test_orphan_temporary_is_ignored_and_write_failure_cleans_only_owned_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = self._context(Path(directory))
            _, _, family_input, recipe, result, expected, target = context
            target.parent.mkdir(parents=True)
            orphan = target.parent / f".{target.name}.orphan.tmp.npz"
            orphan.write_bytes(b"orphan")
            self.assertEqual(decide_resume_action(target, expected), ResumeAction.BUILD_NEW)
            with mock.patch("extensions.cache.os.link", side_effect=OSError("publish failed")):
                with self.assertRaises(OSError):
                    write_cache_atomic(target, result, recipe, family_input)
            self.assertTrue(orphan.exists())
            self.assertEqual(
                [item for item in target.parent.glob(f".{target.name}.*.tmp.npz") if item != orphan],
                [],
            )

    def test_simulated_concurrent_winner_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = self._context(Path(directory))
            _, _, family_input, recipe, result, expected, target = context

            def concurrent_winner(source, destination):
                shutil.copyfile(source, destination)
                raise FileExistsError(os.fspath(destination))

            with mock.patch("extensions.cache.os.link", side_effect=concurrent_winner):
                with self.assertRaises(FileExistsError):
                    write_cache_atomic(target, result, recipe, family_input)
            self.assertEqual(decide_resume_action(target, expected), ResumeAction.SKIP_VERIFIED)
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp.npz")), [])


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        CALL_COUNTS.update(a=0, b=0)
        self.registry_patch = mock.patch(
            "extensions.registry._FAMILIES", (_spec_a(), _spec_b())
        )
        self.registry_patch.start()

    def tearDown(self) -> None:
        self.registry_patch.stop()

    def _paths(self, root: Path, payload=None):
        source_root = root / "source"
        source_path = write_source_npz(source_root, payload=payload)
        return source_path, source_root, root / "cache"

    def _run(self, paths, family_ids, config_overrides=None):
        return run_record(
            paths[0],
            paths[1],
            paths[2],
            family_ids,
            config_overrides or {},
        )

    def test_unrequested_family_is_never_invoked_and_identical_run_skips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))
            first = self._run(paths, ("dummy_a",))
            second = self._run(paths, ("dummy_a",))
            self.assertEqual(first.status, RecordRunStatus.COMPLETE)
            self.assertEqual(first.family_results[0].status, FamilyRunStatus.BUILT)
            self.assertEqual(second.family_results[0].status, FamilyRunStatus.SKIP_VERIFIED)
            self.assertEqual(CALL_COUNTS, {"a": 1, "b": 0})

    def test_single_then_multi_and_multi_then_single_reuse_same_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))
            a_first = self._run(paths, ("dummy_a",))
            both_after = self._run(paths, ("dummy_a", "dummy_b"))
            self.assertEqual(
                [item.status for item in both_after.family_results],
                [FamilyRunStatus.SKIP_VERIFIED, FamilyRunStatus.BUILT],
            )
            self.assertEqual(
                a_first.source_identity.canonical_json(),
                both_after.source_identity.canonical_json(),
            )
            self.assertEqual(CALL_COUNTS, {"a": 1, "b": 1})

        CALL_COUNTS.update(a=0, b=0)
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))
            both_first = self._run(paths, ("dummy_a", "dummy_b"))
            a_after = self._run(paths, ("dummy_a",))
            self.assertEqual(a_after.family_results[0].status, FamilyRunStatus.SKIP_VERIFIED)
            self.assertEqual(
                both_first.source_identity.canonical_json(),
                a_after.source_identity.canonical_json(),
            )
            self.assertEqual(CALL_COUNTS, {"a": 1, "b": 1})

    def test_family_input_review_does_not_block_valid_sibling_or_publish_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload()
            del payload["resp_feature_valid"]
            paths = self._paths(Path(directory), payload=payload)
            summary = self._run(paths, ("dummy_a", "dummy_b"))
            self.assertEqual(summary.status, RecordRunStatus.PARTIAL)
            self.assertEqual(
                [item.status for item in summary.family_results],
                [FamilyRunStatus.BUILT, FamilyRunStatus.REQUIRES_INPUT_REVIEW],
            )
            self.assertEqual(CALL_COUNTS, {"a": 1, "b": 0})
            self.assertEqual(list((paths[2] / "dummy_b").glob("**/*.npz")), [])

    def test_record_identity_error_stops_before_any_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(site_id="OTHER")
            paths = self._paths(Path(directory), payload=payload)
            summary = self._run(paths, ("dummy_a", "dummy_b"))
            self.assertEqual(summary.status, RecordRunStatus.STOP_RECORD)
            self.assertEqual(summary.family_results, ())
            self.assertEqual(CALL_COUNTS, {"a": 0, "b": 0})

    def test_compute_failure_is_family_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))
            summary = self._run(
                paths,
                ("dummy_a", "dummy_b"),
                {"dummy_b": {"raise": True}},
            )
            self.assertEqual(summary.status, RecordRunStatus.PARTIAL)
            self.assertEqual(
                [item.status for item in summary.family_results],
                [FamilyRunStatus.BUILT, FamilyRunStatus.FAILED],
            )
            self.assertEqual(CALL_COUNTS, {"a": 1, "b": 1})

    def test_invalid_existing_family_cache_is_local_and_never_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))
            first = self._run(paths, ("dummy_a", "dummy_b"))
            b_path = first.family_results[1].cache_path
            _rewrite(b_path, {"values": np.asarray([999.0])}, refresh_digest=False)
            invalid_bytes = b_path.read_bytes()
            second = self._run(paths, ("dummy_a", "dummy_b"))
            self.assertEqual(
                [item.status for item in second.family_results],
                [FamilyRunStatus.SKIP_VERIFIED, FamilyRunStatus.STOP_INVALID],
            )
            self.assertEqual(CALL_COUNTS, {"a": 1, "b": 1})
            self.assertEqual(b_path.read_bytes(), invalid_bytes)

    def test_runner_accepts_verified_concurrent_winner_without_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))

            def concurrent_winner(source, destination):
                shutil.copyfile(source, destination)
                raise FileExistsError(os.fspath(destination))

            with mock.patch("extensions.cache.os.link", side_effect=concurrent_winner):
                summary = self._run(paths, ("dummy_a",))
            self.assertEqual(summary.family_results[0].status, FamilyRunStatus.SKIP_VERIFIED)
            self.assertEqual(summary.family_results[0].reason_code, "CONCURRENT_WINNER")
            self.assertEqual(CALL_COUNTS, {"a": 1, "b": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
