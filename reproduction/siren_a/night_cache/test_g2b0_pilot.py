import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from g2a_features import compute_night_features
from g2a_io import load_manifest
from g2b0_pilot import (
    MAX_DISTINCT_READS, MAX_NEW_PILOT, MAX_TOTAL_RECORDS, ReadBudget,
    assert_boundary_effects, boundary_flags, build_inventory, build_parser,
    build_pilot_payload, deterministic_candidates, execute_bounded_pilot, probability_qc,
    select_targeted_records, validate_public_report_text,
)
from test_g2a_features import record

MANIFEST = load_manifest(ROOT / "docs" / "G1B_FEATURE_MANIFEST_FROZEN.csv")


class G2B0PilotTests(unittest.TestCase):
    def test_deterministic_candidates_exclude_golden_and_obey_all_caps(self):
        paths = [Path(f"/input/{site}/r{i:04d}.npz") for site in ("I0002", "I0006", "S0001") for i in range(40)]
        golden = set(paths[::7][:18])
        first = deterministic_candidates(paths, golden, seed=20261013)
        second = deterministic_candidates(paths, golden, seed=20261013)
        self.assertEqual(first, second)
        self.assertTrue(set(first).isdisjoint(golden))
        self.assertLessEqual(len(first) + 18, MAX_DISTINCT_READS)
        self.assertEqual((MAX_NEW_PILOT, MAX_TOTAL_RECORDS, MAX_DISTINCT_READS), (12, 30, 80))

    def test_read_budget_counts_distinct_paths_and_stops_at_80(self):
        loaded = []
        budget = ReadBudget(lambda path: loaded.append(path) or {"path": path}, maximum=MAX_DISTINCT_READS)
        paths = [Path(f"/x/r{i}.npz") for i in range(80)]
        for path in paths: budget.load(path)
        budget.load(paths[0])
        self.assertEqual((budget.distinct_count, budget.read_calls), (80, 81))
        with self.assertRaisesRegex(RuntimeError, "80"):
            budget.load(Path("/x/overflow.npz"))

    def test_four_synthetic_boundaries_are_feature_local(self):
        scenarios = []
        no_hrv = record(); no_hrv["hrv_success"][:] = False; scenarios.append(("no_hrv", no_hrv))
        missing_annotation = record(); missing_annotation["annotation_status"] = "missing"
        missing_annotation["stage_code_aligned"][:] = np.nan
        missing_annotation["stage_valid"][:] = False
        scenarios.append(("annotation_missing", missing_annotation))
        race_unavailable = record(); race_unavailable["x_static"][4:9] = 0; race_unavailable["x_static"][7] = 1; scenarios.append(("race_unavailable", race_unavailable))
        incomplete = record(); incomplete["stage_probability_channels_present"][0] = False; scenarios.append(("probability_incomplete", incomplete))
        for expected, rec in scenarios:
            with self.subTest(expected=expected):
                flags = boundary_flags(rec); self.assertTrue(flags[expected])
                result = compute_night_features(rec, MANIFEST)
                self.assertTrue(assert_boundary_effects(expected, result))
                self.assertEqual(result.unique_feature_values.shape, (807,))
    def test_no_hrv_boundary_requires_an_independently_computable_eeg_output(self):
        rec = record()
        rec["hrv_success"][:] = False
        rec["eeg_clean_subsegment_count"][:] = 0
        self.assertFalse(boundary_flags(rec)["no_hrv"])
        rec["annotation_status"] = "missing"
        rec["stage_code_aligned"][:] = np.nan
        rec["stage_valid"][:] = False
        self.assertTrue(boundary_flags(rec)["annotation_missing"])
        rec["stage_probability_channels_present"][0] = False
        self.assertFalse(boundary_flags(rec)["probability_incomplete"])


    def test_probability_qc_classifies_non9_ranges_stage_and_channels(self):
        rec = record(t=3, w=5)
        rec["stage_valid"] = np.array([True, False, True])
        rec["stage_probability"] = np.array([
            [9.0, -0.1, 1.2, 10.0, 0.5],
            [0.2, 9.0, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ])
        qc = probability_qc(rec)
        self.assertEqual(qc["value_9"], 2)
        self.assertEqual(qc["non9_lt_0"], 1)
        self.assertEqual(qc["non9_gt_1"], 2)
        self.assertEqual(qc["non9_between_1_9"], 1)
        self.assertEqual(qc["non9_gt_9"], 1)
        self.assertEqual((qc["non9_oor_stage_valid"], qc["non9_oor_stage_invalid"]), (3, 0))
        self.assertEqual(qc["anomalous_rows"], 2)
        self.assertEqual(qc["value_9_by_channel"], [1, 1, 0, 0, 0])

    def test_targeted_selector_adds_only_new_boundary_coverage(self):
        fixtures = []
        no_hrv = record(); no_hrv["hrv_success"][:] = False; fixtures.append((Path("/x/I0002/a.npz"), no_hrv))
        annotation = record(); annotation["annotation_status"] = "missing"
        annotation["stage_code_aligned"][:] = np.nan; annotation["stage_valid"][:] = False
        fixtures.append((Path("/x/I0006/b.npz"), annotation))
        race = record(); race["x_static"][4:9] = 0; race["x_static"][7] = 1; fixtures.append((Path("/x/S0001/c.npz"), race))
        incomplete = record(); incomplete["stage_probability_channels_present"][0] = False; fixtures.append((Path("/x/S0001/d.npz"), incomplete))
        ordinary = record(); fixtures.append((Path("/x/S0001/e.npz"), ordinary))
        records = {str(path): rec for path, rec in fixtures}
        budget = ReadBudget(lambda path: records[str(path)], maximum=80)
        selected, coverage = select_targeted_records([path for path, _ in fixtures], budget, maximum=12)
        self.assertEqual([path.name for path, _, _ in selected], ["a.npz", "b.npz", "c.npz", "d.npz"])
        self.assertTrue(all(coverage[name] for name in ("no_hrv", "annotation_missing", "race_unavailable", "probability_incomplete")))
        self.assertEqual(len(selected), 4)
        self.assertEqual(budget.distinct_count, 5)

    def test_annotation_missing_keeps_record_and_only_stage_dependent_families_missing(self):
        rec = record(); rec["annotation_status"] = "missing"
        rec["stage_code_aligned"][:] = np.nan; rec["stage_valid"][:] = False
        result = compute_night_features(rec, MANIFEST)
        first = {}
        for row in MANIFEST.rows: first.setdefault(row["unique_feature_id"], row)
        families = {}
        for i, uid in enumerate(result.unique_ids):
            families.setdefault(first[uid]["family"], []).append(i)
        for family in ("EEG54_stage_spectral", "spatial_asymmetry_candidate",
                       "spatial_ap_gradient_candidate", "CAISR20"):
            self.assertFalse(result.unique_feature_valid_mask[families[family]].any(), family)
        for family in ("HRV11", "EMG24", "Resp14", "Demo5"):
            self.assertTrue(result.unique_feature_valid_mask[families[family]].any(), family)
        self.assertEqual(result.unique_feature_values.shape, (807,))
        self.assertFalse(boundary_flags(rec)["no_hrv"])
        self.assertTrue(boundary_flags(rec)["annotation_missing"])

    def test_inventory_is_name_only_and_rejects_suffix_symlink_or_count_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {"I0002": 2, "I0006": 2, "S0001": 2}
            for site in expected:
                (root / site).mkdir()
                for i in range(2): (root / site / f"r{i}.npz").touch()
            rows, summary = build_inventory(root, expected)
            self.assertEqual(len(rows), 6); self.assertEqual(summary["total"], 6)
            self.assertNotIn("content", rows[0]); self.assertIn("relative_path", rows[0])
            (root / "I0002" / "bad.txt").touch()
            with self.assertRaises(ValueError): build_inventory(root, expected)
            (root / "I0002" / "bad.txt").unlink()
            (root / "I0002" / "link.npz").symlink_to(root / "I0002" / "r0.npz")
            with self.assertRaises(ValueError): build_inventory(root, expected)

    def test_payload_and_cli_have_no_label_or_full_run_surface(self):
        result = compute_night_features(record(), MANIFEST)
        payload = build_pilot_payload(result, {"sample_seed": 20261013})
        self.assertNotIn("y", payload); self.assertNotIn("age", payload)
        parser = build_parser(); options = {option for action in parser._actions for option in action.option_strings}
        self.assertFalse(options & {"--full", "--all", "--batch-6600"})
        self.assertTrue({"--input-root", "--golden-root", "--output-root", "--manifest", "--spec"}.issubset(options))
        with self.assertRaises(ValueError): validate_public_report_text("patient cache /database/private/r001.npz")
        self.assertTrue(validate_public_report_text("aggregate sites: I0002=1; no patient rows"))


    @unittest.skipUnless(
        Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g2a-private/g2a_smoke_v1").is_dir(),
        "local G2-A Golden Cache unavailable",
    )
    def test_bounded_executor_replays_golden_without_candidates(self):
        input_root = Path("/database/home/gaohaojie/workspace/python-example-2026/validity_v0_2_full_6600/workspace/result_bundle/npz")
        golden_root = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g2a-private/g2a_smoke_v1")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot"
            metadata = execute_bounded_pilot(
                input_root=input_root, golden_root=golden_root, output_root=output,
                manifest_path=ROOT / "docs" / "G1B_FEATURE_MANIFEST_FROZEN.csv",
                spec_path=ROOT / "docs" / "G1B_FEATURE_SPEC_FROZEN.md",
                candidate_paths=[], expected_counts={"I0002":319, "I0006":1142, "S0001":5139},
            )
            self.assertEqual(metadata["golden_exact"], 18)
            self.assertEqual(metadata["pilot_records"], 0)
            self.assertEqual(metadata["inventory"]["total"], 6600)
            self.assertLessEqual(metadata["distinct_npz_reads"], 80)
            self.assertTrue(output.is_dir())


if __name__ == "__main__":
    unittest.main(verbosity=2)
