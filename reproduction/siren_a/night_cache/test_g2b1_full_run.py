import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from g2a_io import load_manifest, read_cache
from g2b1_cache_contract import Action
from g2b1_full_run import (
    EXPECTED_COUNTS, aggregate_coverage, build_inventory, build_or_resume,
    build_parser, cache_name, load_caisr_audit, preflight,
    validate_public_coverage_rows, verify_full_cache, verify_reference_caches,
)
from test_g2a_features import record

MANIFEST_PATH = ROOT / "docs" / "G1B_FEATURE_MANIFEST_FROZEN.csv"
SPEC_PATH = ROOT / "docs" / "G1B_FEATURE_SPEC_FROZEN.md"
MANIFEST = load_manifest(MANIFEST_PATH)


def write_record(path, rec=None):
    rec = record(t=5, w=5) if rec is None else rec
    payload = dict(rec)
    payload["annotation_signal_lengths_json"] = np.asarray(json.dumps(payload.pop("annotation_signal_lengths")))
    payload["extraction_version"] = np.asarray(payload["extraction_version"])
    payload["validity_schema_version"] = np.asarray(payload["validity_schema_version"])
    payload["annotation_status"] = np.asarray(payload["annotation_status"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def small_fixture(root):
    input_root = root / "input"
    expected = {"I0002": 1, "I0006": 1, "S0001": 1}
    records = []
    for number, site in enumerate(expected):
        rec = record(t=5, w=5)
        if site == "I0002":
            rec["annotation_status"] = "missing"
            rec["stage_code_aligned"][:] = np.nan; rec["stage_valid"][:] = False
        path = input_root / site / f"record_{number}.npz"; write_record(path, rec)
        records.append((site, path, rec))
    audit_root = root / "audit"; audit_root.mkdir()
    audit_rows = []
    for site, path, rec in records:
        missing = rec["annotation_status"] == "missing"
        count = int(np.asarray(rec["stage_valid"], bool).sum())
        audit_rows.append({
            "site": site, "BidsFolder": path.stem, "SessionID": "",
            "record_key": path.stem,
            "file_status": "FILE_MISSING" if missing else "FILE_PRESENT_STAGE_READABLE",
            "aligned_valid_count": 0 if missing else count,
            "npz_stage_valid_count": count,
        })
    (audit_root / "summary.json").write_text(json.dumps({"rows": audit_rows}))
    return input_root, audit_root, expected


class G2B1FullRunTests(unittest.TestCase):
    def test_inventory_is_sorted_anonymous_and_rejects_symlink_and_count_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); input_root, _audit, expected = small_fixture(root)
            rows, summary = build_inventory(input_root, expected)
            self.assertEqual([row["cache_name"] for row in rows], [cache_name(i) for i in range(3)])
            self.assertEqual(summary["site_counts"], expected); self.assertEqual(summary["total"], 3)
            self.assertTrue(all("record_" not in row["cache_name"] for row in rows))
            link = input_root / "I0002" / "link.npz"; link.symlink_to(input_root / "I0002" / "record_0.npz")
            with self.assertRaises(ValueError): build_inventory(input_root, expected)
            link.unlink()
            with self.assertRaises(ValueError): build_inventory(input_root, {**expected, "I0002": 2})

    def test_preflight_checks_hashes_audit_capacity_and_output_separation_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); input_root, audit_root, expected = small_fixture(root)
            output = root / "output"
            result = preflight(input_root, output, MANIFEST_PATH, SPEC_PATH, audit_root,
                               expected_counts=expected, expected_audit=(1, 2, 3),
                               estimated_cache_bytes=20000, disk_free_bytes=10**9)
            self.assertEqual(result["inventory"]["total"], 3)
            self.assertEqual(result["caisr"], {"annotation_missing":1, "readable_lt20":2, "stage_source_lt20":3})
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                preflight(input_root, input_root / "inside", MANIFEST_PATH, SPEC_PATH, audit_root,
                          expected_counts=expected, expected_audit=(1,2,3),
                          estimated_cache_bytes=20000, disk_free_bytes=10**9)
            with self.assertRaises(ValueError):
                preflight(input_root, output, MANIFEST_PATH, SPEC_PATH, audit_root,
                          expected_counts=expected, expected_audit=(1,2,3),
                          estimated_cache_bytes=10**9, disk_free_bytes=100)

    def test_caisr_mapping_closes_and_rejects_duplicate_or_unknown_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); input_root, audit_root, expected = small_fixture(root)
            inventory, _ = build_inventory(input_root, expected)
            result = load_caisr_audit(audit_root, inventory, expected_facts=(1,2,3))
            self.assertEqual(len(result["by_relative_path"]), 3)
            payload = json.loads((audit_root / "summary.json").read_text())
            payload["rows"].append(dict(payload["rows"][0]))
            (audit_root / "summary.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError): load_caisr_audit(audit_root, inventory, expected_facts=(1,2,3))

    def test_build_resume_verify_and_coverage_keep_all_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); input_root, audit_root, expected = small_fixture(root)
            output = root / "output"
            pf = preflight(input_root, output, MANIFEST_PATH, SPEC_PATH, audit_root,
                           expected_counts=expected, expected_audit=(1,2,3),
                           estimated_cache_bytes=20000, disk_free_bytes=10**9)
            first = build_or_resume(pf, output, mode="build", stop_after=1)
            self.assertEqual((first["computed"], first["complete"]), (1, False))
            second = build_or_resume(pf, output, mode="resume")
            self.assertEqual((second["computed"], second["skipped"], second["complete"]), (2, 1, True))
            verified = verify_full_cache(pf, output)
            self.assertEqual(verified["verified"], 3)
            self.assertEqual(verified["unique_positions"], 3 * 807)
            self.assertEqual(verified["view_positions"], 3 * 827)
            coverage = aggregate_coverage(pf, output)
            self.assertEqual(len(coverage["public_rows"]), 4 * 807)
            self.assertTrue(validate_public_coverage_rows(coverage["public_rows"])); self.assertEqual(coverage["records"], 3)
            missing_cache = output / "cache" / cache_name(2); missing_cache.unlink()
            self.assertEqual(build_or_resume(pf, output, mode="resume")["computed"], 1)

    def test_invalid_existing_cache_and_run_identity_drift_stop_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); input_root, audit_root, expected = small_fixture(root)
            output = root / "output"
            pf = preflight(input_root, output, MANIFEST_PATH, SPEC_PATH, audit_root,
                           expected_counts=expected, expected_audit=(1,2,3),
                           estimated_cache_bytes=20000, disk_free_bytes=10**9)
            build_or_resume(pf, output, mode="build")
            cache = output / "cache" / cache_name(0); cache.write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "STOP_INVALID"):
                build_or_resume(pf, output, mode="resume")
            altered = dict(pf); altered["run_identity"] = dict(pf["run_identity"])
            altered["run_identity"]["g2b1_full_run_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "identity"):
                build_or_resume(altered, output, mode="resume")

    def test_annotation_missing_retains_independent_modalities_and_short_stage_uses_five_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); input_root, audit_root, expected = small_fixture(root)
            output = root / "output"
            pf = preflight(input_root, output, MANIFEST_PATH, SPEC_PATH, audit_root,
                           expected_counts=expected, expected_audit=(1,2,3),
                           estimated_cache_bytes=20000, disk_free_bytes=10**9)
            build_or_resume(pf, output, mode="build")
            item = next(row for row in pf["inventory_rows"] if row["site"] == "I0002")
            arrays = read_cache(output / "cache" / item["cache_name"])
            first = {}
            for row in MANIFEST.rows: first.setdefault(row["unique_feature_id"], row)
            families = {}
            for i, uid in enumerate(MANIFEST.unique_ids): families.setdefault(first[uid]["family"], []).append(i)
            for family in ("EEG54_stage_spectral", "spatial_asymmetry_candidate", "spatial_ap_gradient_candidate", "CAISR20"):
                self.assertFalse(arrays["unique_valid_mask"][families[family]].any())
            for family in ("HRV11", "EMG24", "Resp14", "Demo5"):
                self.assertTrue(arrays["unique_valid_mask"][families[family]].any())
            self.assertEqual(arrays["unique_values"].shape, (807,))

    def test_cli_requires_explicit_mode_and_has_no_training_or_transform_options(self):
        parser = build_parser()
        with self.assertRaises(SystemExit): parser.parse_args([])
        options = {option for action in parser._actions for option in action.option_strings}
        self.assertTrue({"--preflight-only", "--build", "--verify-only", "--resume"}.issubset(options))
        self.assertFalse(options & {"--train", "--impute", "--scale", "--normalize", "--rank"})

    def test_reference_replay_supports_golden_and_pilot_without_modifying_them(self):
        golden = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g2a-private/g2a_smoke_v1")
        pilot = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g2b0-private/g2b0_pilot_v1/retry_004")
        if not golden.is_dir() or not pilot.is_dir(): self.skipTest("private reference caches unavailable")
        golden_result = verify_reference_caches(golden, "private_sample_map.json", MANIFEST)
        pilot_result = verify_reference_caches(pilot, "pilot_mapping.json", MANIFEST)
        self.assertEqual(golden_result["matched"], 18)
        self.assertEqual(pilot_result["matched"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
