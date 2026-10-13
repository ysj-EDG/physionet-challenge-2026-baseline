import concurrent.futures
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from g2a_io import load_manifest, write_cache_atomic
from g2b1_cache_contract import (
    Action, CACHE_SCHEMA_VERSION, REQUIRED_PROVENANCE,
    assert_valid_cache_arrays, decide_resume_action,
    validate_cache_arrays, validate_existing_cache,
)

MANIFEST = load_manifest(ROOT / "docs" / "G1B_FEATURE_MANIFEST_FROZEN.csv")


def provenance():
    return {
        "source_npz_sha256": "1" * 64,
        "manifest_sha256": MANIFEST.sha256,
        "spec_sha256": "2" * 64,
        "g2a_features_sha256": "3" * 64,
        "g2a_io_sha256": "4" * 64,
        "g2b0_batch_safety_sha256": "5" * 64,
        "g2b1_cache_contract_sha256": "6" * 64,
        "g2b1_full_run_sha256": "7" * 64,
        "python_version": "3.9.18",
        "numpy_version": "1.26.4",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
    }


def valid_payload():
    first = {}
    for row in MANIFEST.rows:
        first.setdefault(row["unique_feature_id"], row)
    pooled = {"EEG54_stage_spectral", "spatial_asymmetry_candidate",
              "spatial_ap_gradient_candidate", "HRV11", "EMG24", "Resp14"}
    values = np.arange(807, dtype=np.float64)
    mask = np.ones(807, dtype=bool)
    support = np.asarray([5 if first[uid]["family"] in pooled else 1
                          for uid in MANIFEST.unique_ids], dtype=np.int64)
    payload = {
        "unique_feature_ids": np.asarray(MANIFEST.unique_ids),
        "unique_values": values, "unique_valid_mask": mask,
        "unique_support_count": support,
        **{key: np.asarray(value) for key, value in provenance().items()},
    }
    index = {uid: position for position, uid in enumerate(MANIFEST.unique_ids)}
    for view in MANIFEST.view_sizes:
        rows = [row for row in MANIFEST.rows if row["view_id"] == view]
        positions = np.asarray([index[row["unique_feature_id"]] for row in rows])
        payload[f"{view}_values"] = values[positions]
        payload[f"{view}_valid_mask"] = mask[positions]
        payload[f"{view}_support_count"] = support[positions]
    return payload


def synchronize_unique_to_views(payload, position):
    uid = MANIFEST.unique_ids[position]
    for view in MANIFEST.view_sizes:
        rows = [row for row in MANIFEST.rows if row["view_id"] == view]
        for offset, row in enumerate(rows):
            if row["unique_feature_id"] == uid:
                payload[f"{view}_values"][offset] = payload["unique_values"][position]
                payload[f"{view}_valid_mask"][offset] = payload["unique_valid_mask"][position]
                payload[f"{view}_support_count"][offset] = payload["unique_support_count"][position]


class G2B1CacheContractTests(unittest.TestCase):
    def test_schema_and_provenance_are_independent_from_pilot(self):
        self.assertEqual(CACHE_SCHEMA_VERSION, "g2b1_full_v1")
        self.assertEqual(set(REQUIRED_PROVENANCE), set(provenance()))
        self.assertNotIn("g2b0_pilot_sha256", REQUIRED_PROVENANCE)
        self.assertNotIn("sample_seed", REQUIRED_PROVENANCE)
        self.assertIn("g2b1_cache_contract_sha256", REQUIRED_PROVENANCE)
        self.assertIn("g2b1_full_run_sha256", REQUIRED_PROVENANCE)
        payload = valid_payload(); payload["cache_schema_version"] = np.asarray("g2b0_pilot_v1")
        result = validate_cache_arrays(payload, provenance(), MANIFEST)
        self.assertFalse(result.valid); self.assertIn("cache_schema_version", " ".join(result.errors))

    def test_complete_cache_validates_and_resume_uses_three_states(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            self.assertEqual(decide_resume_action(path, provenance(), MANIFEST), Action.COMPUTE)
            write_cache_atomic(path, valid_payload(), lambda arrays: assert_valid_cache_arrays(arrays, provenance(), MANIFEST))
            self.assertTrue(validate_existing_cache(path, provenance(), MANIFEST).valid)
            self.assertEqual(decide_resume_action(path, provenance(), MANIFEST), Action.SKIP_VERIFIED)

    def test_source_and_each_execution_fingerprint_change_stop_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"; write_cache_atomic(path, valid_payload())
            for key in REQUIRED_PROVENANCE:
                if key in {"python_version", "numpy_version", "cache_schema_version"}: continue
                with self.subTest(key=key):
                    expected = provenance(); expected[key] = "f" * 64
                    self.assertEqual(decide_resume_action(path, expected, MANIFEST), Action.STOP_INVALID)

    def test_dtype_shape_corruption_and_unreadable_npz_stop_safely(self):
        mutations = (
            ("float-mask", lambda p: p.__setitem__("unique_valid_mask", p["unique_valid_mask"].astype(np.float64))),
            ("int-mask", lambda p: p.__setitem__("unique_valid_mask", p["unique_valid_mask"].astype(np.int64))),
            ("short-shape", lambda p: p.__setitem__("unique_values", p["unique_values"][:-1])),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, mutate in mutations:
                payload = valid_payload(); mutate(payload); path = root / f"{name}.npz"
                write_cache_atomic(path, payload)
                self.assertEqual(decide_resume_action(path, provenance(), MANIFEST), Action.STOP_INVALID)
            broken = root / "broken.npz"; broken.write_bytes(b"broken")
            self.assertEqual(decide_resume_action(broken, provenance(), MANIFEST), Action.STOP_INVALID)

    def test_support_semantics_reject_synchronized_unique_and_view_tampering(self):
        first = {}
        for row in MANIFEST.rows: first.setdefault(row["unique_feature_id"], row)
        positions = {}
        for position, uid in enumerate(MANIFEST.unique_ids):
            positions.setdefault(first[uid]["family"], position)
        cases = (
            (positions["EEG54_stage_spectral"], True, 0),
            (positions["EEG54_stage_spectral"], False, 5),
            (positions["CAISR20"], True, 0),
            (positions["Demo5"], False, 1),
        )
        for position, is_valid, support in cases:
            payload = valid_payload()
            payload["unique_valid_mask"][position] = is_valid
            payload["unique_values"][position] = float(position) if is_valid else np.nan
            payload["unique_support_count"][position] = support
            synchronize_unique_to_views(payload, position)
            with self.subTest(position=position, valid=is_valid, support=support):
                self.assertFalse(validate_cache_arrays(payload, provenance(), MANIFEST).valid)

    def test_competing_writers_publish_exactly_one_valid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "race.npz"; barrier = threading.Barrier(2)
            def writer(_number):
                barrier.wait()
                try:
                    write_cache_atomic(path, valid_payload(), lambda arrays: assert_valid_cache_arrays(arrays, provenance(), MANIFEST))
                    return True
                except FileExistsError:
                    return False
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                winners = list(pool.map(writer, (1, 2)))
            self.assertEqual(sum(winners), 1)
            self.assertTrue(validate_existing_cache(path, provenance(), MANIFEST).valid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
