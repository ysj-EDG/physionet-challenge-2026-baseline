from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import extensions.smoke as smoke
from extensions.contracts import SourceContractError
from extensions.tests.fixtures import source_payload, write_source_npz


class SmokeSelectionTests(unittest.TestCase):
    def test_selection_is_bounded_deterministic_and_label_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "npz"
            write_source_npz(
                root,
                site_id="AAA",
                record_id="low",
                payload=source_payload(site_id="AAA", record_id="low", epochs=4),
            )
            write_source_npz(
                root,
                site_id="ZZZ",
                record_id="good",
                payload=source_payload(site_id="ZZZ", record_id="good", epochs=120),
            )
            with mock.patch(
                "extensions.smoke.load_source_npz", wraps=smoke.load_source_npz
            ) as loader:
                identity = smoke.select_smoke_source(root, max_candidates=80)
            self.assertEqual(identity.site_id, "ZZZ")
            self.assertEqual(identity.record_id, "good")
            self.assertLessEqual(loader.call_count, 80)
            repeat = smoke.select_smoke_source(root, max_candidates=80)
            self.assertEqual(repeat.digest, identity.digest)

    def test_selection_stops_within_candidate_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "npz"
            write_source_npz(
                root,
                site_id="AAA",
                record_id="low",
                payload=source_payload(site_id="AAA", record_id="low", epochs=4),
            )
            write_source_npz(
                root,
                site_id="ZZZ",
                record_id="good",
                payload=source_payload(site_id="ZZZ", record_id="good", epochs=120),
            )
            with self.assertRaises(SourceContractError):
                smoke.select_smoke_source(root, max_candidates=1)


class SmokeSyntheticCheckTests(unittest.TestCase):
    def test_synthetic_checks_all_pass_with_labeled_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = smoke.run_synthetic_cache_checks(Path(directory))
        for name in (
            "check_a_build_then_skip",
            "check_a_zero_extractor_calls_on_resume",
            "check_a_single_then_six_reuse",
            "check_a_six_then_single_reuse",
            "check_b_single_parameter_invalidation",
            "check_c_missing_family_recovery",
            "check_d_family_input_review_isolation",
            "check_d_record_stop_no_execution",
        ):
            self.assertTrue(getattr(summary, name), name)
        self.assertGreater(summary.built_count, 0)
        self.assertGreater(summary.skipped_count, 0)
        self.assertTrue(summary.synthetic_examples)
        for example in summary.synthetic_examples:
            self.assertEqual(example["source"], "synthetic")
            self.assertIn("feature_id", example)
            self.assertIn("value", example)


class SmokePrivacyTests(unittest.TestCase):
    def test_public_summary_rejects_forbidden_content(self) -> None:
        for payload in (
            {"record_key": "rec_abc"},
            {"site_id": "X", "record_id": "Y"},
            {"values": [0.1, 0.2]},
            {"thresholds": {"q88": 1.0}},
            {"support_count": [20, 20]},
            {"feature_ids": ["a.b.c"]},
            {"source_path": "/data/x.npz"},
            {"identity_mapping": {"rec_abc": "X/Y"}},
            {"families": [{"family_id": "eeg_tail", "values": [1.0]}]},
            {"families": [{"family_id": "eeg_tail", "cache_path": "/tmp/x"}]},
        ):
            with self.subTest(payload), self.assertRaises(ValueError):
                smoke.assert_public_summary_safe(payload)

    def test_real_smoke_refuses_private_report_inside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "npz"
            source_path = write_source_npz(
                source_root, payload=source_payload(epochs=120)
            )
            cache_root = root / "cache"
            inside = Path(__file__).resolve().parents[3] / "docs" / "private.json"
            with self.assertRaises(ValueError):
                smoke.run_real_smoke(source_path, source_root, cache_root, inside)

    def test_real_smoke_orchestration_and_private_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "npz"
            source_path = write_source_npz(
                source_root, payload=source_payload(epochs=120)
            )
            cache_root = root / "cache"
            private_path = root / "private" / "report.json"
            public = smoke.run_real_smoke(
                source_path, source_root, cache_root, private_path
            )
            self.assertEqual(public.record_status, "COMPLETE")
            self.assertEqual(public.resume_skips, 6)
            self.assertEqual(public.resume_extractor_calls, 0)
            self.assertTrue(public.source_immutable)
            self.assertEqual(len(public.families), 6)
            self.assertTrue(
                all(item.valid_count is not None for item in public.families)
            )
            smoke.assert_public_summary_safe(public.to_public_dict())
            private = json.loads(private_path.read_text(encoding="utf-8"))
            for key in (
                "source_sha256_before",
                "source_sha256_after",
                "record_key",
                "family_reports",
                "stage_epoch_counts",
                "time_range_sec",
            ):
                self.assertIn(key, private)
            self.assertEqual(len(private["family_reports"]), 6)
            first = private["family_reports"][0]
            self.assertIn("example_values", first)
            self.assertTrue(first["example_values"])
            second = smoke.run_real_smoke(
                source_path, source_root, cache_root, private_path
            )
            self.assertEqual(second.resume_extractor_calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
