from __future__ import annotations

import hashlib
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import extensions
from extensions.contracts import (
    FamilyInputContract,
    FamilyInputReviewError,
    FeatureResult,
    FieldRequirement,
    RecordFatalError,
    RecordKeyCollisionError,
    validate_feature_result,
)
from extensions.sources import (
    build_source_inventory,
    load_source_npz,
    record_key_for,
    validate_family_input,
)
from extensions.tests.fixtures import source_payload, write_source_npz


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family_a_contract() -> FamilyInputContract:
    return FamilyInputContract(
        family_id="family_a",
        contract_version="1",
        requirements=(
            FieldRequirement("X_seq", ("T", 483), ("float32",)),
            FieldRequirement("stage_valid", ("T",), ("bool",)),
        ),
        stage_time_rules=("T is the uncompressed epoch axis",),
        validity_rules=("stage_valid must gate upstream finite zero fills",),
    )


def _family_b_contract() -> FamilyInputContract:
    return FamilyInputContract(
        family_id="family_b",
        contract_version="1",
        requirements=(
            FieldRequirement("stage_code_aligned", ("T",), ("float32",)),
            FieldRequirement("resp_feature_valid", ("T", 14), ("bool",), columns=tuple(range(14))),
        ),
        stage_time_rules=("T is the uncompressed epoch axis",),
        validity_rules=("resp_feature_valid is authoritative per feature",),
    )


class RecordKeyTests(unittest.TestCase):
    def test_package_exports_task1_public_contracts(self) -> None:
        expected = {
            "FamilyInputContract",
            "FamilyInputIdentity",
            "FeatureResult",
            "SourceIdentity",
            "SourceRecord",
            "load_source_npz",
            "record_key_for",
            "validate_family_input",
            "validate_feature_result",
        }
        self.assertTrue(expected.issubset(set(extensions.__all__)))

    def test_record_key_is_deterministic_cross_site_and_path_safe(self) -> None:
        first = record_key_for("SITE_A", "record-001")
        self.assertEqual(first, record_key_for("SITE_A", "record-001"))
        self.assertRegex(first, r"^rec_[0-9a-f]{64}$")
        self.assertNotEqual(first, record_key_for("SITE_B", "record-001"))
        hostile = record_key_for("../SITE A", "../../record/with spaces")
        self.assertRegex(hostile, r"^rec_[0-9a-f]{64}$")
        self.assertIsNone(re.search(r"[/.\\ ]", hostile))

    def test_forced_digest_collision_between_distinct_tuples_is_record_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                write_source_npz(root, site_id="SITE_A", record_id="r1"),
                write_source_npz(root, site_id="SITE_B", record_id="r2"),
            ]
            with mock.patch(
                "extensions.sources._record_key_digest",
                return_value="0" * 64,
            ):
                with self.assertRaises(RecordKeyCollisionError):
                    build_source_inventory(paths, root)


class SourceLoadingTests(unittest.TestCase):
    def test_source_and_family_identity_do_not_depend_on_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_source_npz(root)
            source = load_source_npz(path, root)
            a = _family_a_contract()
            b = _family_b_contract()

            a_alone = validate_family_input(source, a)
            a_then_b = (validate_family_input(source, a), validate_family_input(source, b))[0]
            b_then_a = (validate_family_input(source, b), validate_family_input(source, a))[1]
            together = {c.family_id: validate_family_input(source, c) for c in (a, b)}["family_a"]

            source_json = source.identity.canonical_json()
            self.assertEqual(source_json, load_source_npz(path, root).identity.canonical_json())
            self.assertEqual(
                {identity.canonical_json() for identity in (a_alone, a_then_b, b_then_a, together)},
                {a_alone.canonical_json()},
            )

    def test_loader_uses_allow_pickle_false_and_never_exposes_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_source_npz(root)
            with mock.patch(
                "extensions.sources.np.load",
                wraps=np.load,
            ) as load_spy:
                source = load_source_npz(path, root)
            self.assertEqual(load_spy.call_args.kwargs["allow_pickle"], False)
            self.assertNotIn("y", source.arrays)
            self.assertNotIn("y", source.loaded_fields)

    def test_core_source_errors_are_record_fatal(self) -> None:
        cases = {}
        wrong_version = source_payload()
        wrong_version["extraction_version"] = np.asarray("wrong")
        cases["wrong extraction schema"] = ("SITE_A", "record-001", wrong_version)
        path_mismatch = source_payload(site_id="SITE_OTHER")
        cases["path scalar mismatch"] = ("SITE_A", "record-001", path_mismatch)
        missing_identity = source_payload()
        del missing_identity["record_id"]
        cases["missing record identity"] = ("SITE_A", "record-001", missing_identity)

        for label, (path_site, path_record, payload) in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = write_source_npz(
                    root,
                    site_id=path_site,
                    record_id=path_record,
                    payload=payload,
                )
                with self.assertRaises(RecordFatalError):
                    load_source_npz(path, root)

    def test_missing_source_path_is_record_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RecordFatalError):
                load_source_npz(root / "SITE_A" / "missing.npz", root)

    def test_duplicate_identity_with_changed_file_content_is_record_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_source_npz(root)

            def paths_with_content_change():
                yield path
                changed = source_payload()
                changed["X_seq"] = changed["X_seq"] + np.float32(1.0)
                write_source_npz(root, payload=changed)
                yield path

            with self.assertRaises(RecordFatalError):
                build_source_inventory(paths_with_content_change(), root)

    def test_family_input_errors_are_isolated_and_do_not_mutate_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = source_payload()
            del payload["stage_valid"]
            path = write_source_npz(root, payload=payload)
            source = load_source_npz(path, root)
            before_b = source.arrays["resp_feature_valid"].copy()

            with self.assertRaises(FamilyInputReviewError):
                validate_family_input(source, _family_a_contract())
            identity_b = validate_family_input(source, _family_b_contract())

            self.assertEqual(identity_b.family_id, "family_b")
            np.testing.assert_array_equal(source.arrays["resp_feature_valid"], before_b)

    def test_malformed_family_shape_and_validity_dtype_are_family_review_errors(self) -> None:
        cases = {}
        bad_shape = source_payload()
        bad_shape["X_seq"] = bad_shape["X_seq"][:, :-1]
        cases["shape"] = bad_shape
        bad_validity = source_payload()
        bad_validity["stage_valid"] = bad_validity["stage_valid"].astype(np.int8)
        cases["validity dtype"] = bad_validity

        for label, payload in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = load_source_npz(write_source_npz(root, payload=payload), root)
                with self.assertRaises(FamilyInputReviewError):
                    validate_family_input(source, _family_a_contract())
                self.assertEqual(
                    validate_family_input(source, _family_b_contract()).family_id,
                    "family_b",
                )

    def test_loading_does_not_change_source_bytes_hash_or_stat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_source_npz(root)
            before_stat = path.stat()
            before_hash = _sha256(path)
            load_source_npz(path, root)
            after_stat = path.stat()
            self.assertEqual(_sha256(path), before_hash)
            self.assertEqual(after_stat.st_size, before_stat.st_size)
            self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)
            self.assertEqual(after_stat.st_mode, before_stat.st_mode)


class FeatureResultTests(unittest.TestCase):
    def _result(self) -> FeatureResult:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = load_source_npz(write_source_npz(root), root)
            return FeatureResult(
                family_id="family_a",
                family_version="1",
                feature_ids=("valid_zero", "valid_one", "missing"),
                values=np.asarray([0.0, 1.0, np.nan], dtype=np.float64),
                valid_mask=np.asarray([True, True, False], dtype=bool),
                support_count=np.asarray([4, 4, 0], dtype=np.int64),
                missing_reason=("", "", "LOW_SUPPORT"),
                feature_units=("ratio", "ratio", "ratio"),
                source_identity=source.identity,
                processing_parameters={"minimum_support": 4},
            )

    def test_feature_result_preserves_valid_zero_and_explicit_missingness(self) -> None:
        result = self._result()
        self.assertIs(validate_feature_result(result), result)
        self.assertEqual(result.values[0], 0.0)

    def test_invalid_finite_zero_cannot_masquerade_as_missing(self) -> None:
        result = self._result()
        result.values[2] = 0.0
        with self.assertRaises(ValueError):
            validate_feature_result(result)

    def test_result_invariants_reject_lengths_values_supports_and_reasons(self) -> None:
        result = self._result()
        mutations = (
            ("duplicate IDs", {"feature_ids": ("same", "same", "missing")}),
            ("unequal lengths", {"feature_units": ("ratio", "ratio")}),
            ("valid NaN", {"values": np.asarray([np.nan, 1.0, np.nan])}),
            ("negative support", {"support_count": np.asarray([4, -1, 0])}),
            ("empty missing reason", {"missing_reason": ("", "", "")}),
        )
        for label, changes in mutations:
            with self.subTest(label):
                candidate = result.replaced(**changes)
                with self.assertRaises(ValueError):
                    validate_feature_result(candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
