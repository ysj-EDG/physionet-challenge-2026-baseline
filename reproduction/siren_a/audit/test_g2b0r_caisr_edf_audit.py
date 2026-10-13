import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from g2b0r_caisr_edf_audit import (
    FILE_MISSING, FILE_PRESENT_NO_STAGE, FILE_PRESENT_STAGE_READABLE,
    FILE_PRESENT_UNREADABLE, aggregate_rows, align_stage, audit_edf,
    compare_npz_stage, exact_paths, load_identities, publish_chunk,
    read_verified_chunk, stage_summary,
)


class FakeSignal:
    def __init__(self, label, data, frequency=1 / 30):
        self.label = label
        self.data = np.asarray(data, dtype=float)
        self.sampling_frequency = frequency


class FakeEdf:
    def __init__(self, signals): self.signals = signals


class CaisrAuditTests(unittest.TestCase):
    def test_missing_unreadable_no_stage_and_normalized_stage_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); missing = root / "missing.edf"
            self.assertEqual(audit_edf(missing, lambda *_a, **_k: None)["file_status"], FILE_MISSING)
            unreadable = root / "bad.edf"; unreadable.touch()
            def fail(*_a, **_k): raise OSError("bad edf")
            self.assertEqual(audit_edf(unreadable, fail)["file_status"], FILE_PRESENT_UNREADABLE)
            no_stage = root / "none.edf"; no_stage.touch()
            self.assertEqual(audit_edf(no_stage, lambda *_a, **_k: FakeEdf([FakeSignal("event", [1])]))["file_status"], FILE_PRESENT_NO_STAGE)
            good = root / "good.edf"; good.touch()
            row = audit_edf(good, lambda *_a, **_k: FakeEdf([FakeSignal(" Stage_CAISR ", [1, 2])]))
            self.assertEqual(row["file_status"], FILE_PRESENT_STAGE_READABLE)

    def test_validity_is_exact_no_rounding_and_threshold_bins(self):
        values = np.array([1, 2, 3, 4, 5, 0, 9, np.nan, np.inf, -np.inf, 1.1])
        summary = stage_summary(values)
        self.assertEqual((summary["valid_count"], summary["invalid_count"], summary["unknown_count"]), (5, 6, 3))
        for count, expected in ((19, "5-19"), (20, ">=20"), (21, ">=20")):
            with self.subTest(count=count):
                self.assertEqual(stage_summary(np.ones(count))["valid_bin"], expected)

    def test_alignment_shorter_longer_and_raw_ge20_aligned_lt20(self):
        short = align_stage(np.array([1, 2]), 4)
        np.testing.assert_array_equal(short[:2], [1, 2]); self.assertTrue(np.isnan(short[2:]).all())
        np.testing.assert_array_equal(align_stage(np.arange(8.0), 3), [0, 1, 2])
        raw = np.ones(21); aligned = align_stage(raw, 19)
        self.assertEqual(stage_summary(raw)["valid_count"], 21)
        self.assertEqual(stage_summary(aligned)["valid_count"], 19)

    def test_npz_stage_and_annotation_matches_and_mismatches(self):
        aligned = np.array([1.0, np.nan, 5.0]); valid = np.array([True, False, True])
        match = compare_npz_stage("available", aligned, valid, "available", aligned.copy(), valid.copy())
        self.assertTrue(all(match.values()))
        mismatch = compare_npz_stage("available", aligned, valid, "missing", np.array([1, 2, 5.0]), np.ones(3, bool))
        self.assertFalse(mismatch["annotation_status_match"])
        self.assertFalse(mismatch["stage_code_match"]); self.assertFalse(mismatch["stage_valid_match"])

    def test_sample_frequency_warning_and_stage_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.edf"; path.touch()
            row = audit_edf(path, lambda *_a, **_k: FakeEdf([FakeSignal("stage_caisr", [1, 1, 2, 9], 1.0)]))
            self.assertTrue(row["sampling_frequency_warning"])
            self.assertEqual(row["stage_counts"], {"N3": 2, "N2": 1, "N1": 0, "REM": 0, "Wake": 0})

    def test_identity_loader_uses_exact_three_columns_and_paths_are_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); demographics = root / "demographics.csv"
            with demographics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["SiteID", "BidsFolder", "SessionID", "extra"])
                writer.writeheader(); writer.writerow({"SiteID":"I0002", "BidsFolder":"sub-001", "SessionID":"01", "extra":"ignored"})
            rows = load_identities(demographics, expected_total=1, expected_counts={"I0002": 1})
            self.assertEqual(rows, [{"SiteID":"I0002", "BidsFolder":"sub-001", "SessionID":"01"}])
            edf, npz = exact_paths(rows[0], root, root / "npz")
            self.assertEqual(edf.name, "sub-001_ses-01_caisr_annotations.edf")
            self.assertEqual(npz.name, "sub-001_ses-01.npz")
            with self.assertRaises(ValueError): exact_paths({"SiteID":"I0002", "BidsFolder":"../escape", "SessionID":"01"}, root, root / "npz")

    def test_aggregate_mutually_exclusive_counts_close_by_site(self):
        rows = []
        for site, statuses in {"I0002":[FILE_MISSING, FILE_PRESENT_STAGE_READABLE], "I0006":[FILE_PRESENT_UNREADABLE], "S0001":[FILE_PRESENT_NO_STAGE]}.items():
            for status in statuses:
                rows.append({"site":site, "file_status":status, "raw_valid_count":21 if status == FILE_PRESENT_STAGE_READABLE else None,
                             "aligned_valid_count":20 if status == FILE_PRESENT_STAGE_READABLE else None})
        result = aggregate_rows(rows, {"I0002":2,"I0006":1,"S0001":1})
        self.assertEqual(result["Overall"]["total"], 4)
        for site in ("I0002", "I0006", "S0001", "Overall"):
            self.assertEqual(sum(result[site]["file_status"].values()), result[site]["total"])

    def test_checkpoint_publish_resume_and_identity_mismatch_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.json"
            payload = {"run_identity":"abc", "rows":[{"site":"I0002"}]}
            publish_chunk(path, payload)
            self.assertEqual(read_verified_chunk(path, "abc"), payload)
            with self.assertRaises(FileExistsError): publish_chunk(path, payload)
            with self.assertRaises(ValueError): read_verified_chunk(path, "different")
            path.write_text("broken", encoding="utf-8")
            with self.assertRaises(ValueError): read_verified_chunk(path, "abc")

    def test_source_contains_no_forbidden_processing_surface(self):
        source = (HERE / "g2b0r_caisr_edf_audit.py").read_text(encoding="utf-8")
        self.assertNotIn("extract_all", source)
        self.assertNotIn('["y"]', source)
        self.assertNotIn("glob(", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
