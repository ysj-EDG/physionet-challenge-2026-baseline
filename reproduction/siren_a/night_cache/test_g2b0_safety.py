import concurrent.futures
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import g2a_io
from g2a_io import load_manifest, read_cache, write_cache_atomic
from g2b0_batch_safety import Action, decide_resume_action, validate_existing_cache

MANIFEST = load_manifest(ROOT / "docs" / "G1B_FEATURE_MANIFEST_FROZEN.csv")
PROVENANCE = {
    "source_npz_sha256": "1" * 64,
    "manifest_sha256": MANIFEST.sha256,
    "spec_sha256": "2" * 64,
    "g2a_features_sha256": "3" * 64,
    "g2a_io_sha256": "4" * 64,
    "g2b0_batch_safety_sha256": "5" * 64,
    "g2b0_pilot_sha256": "6" * 64,
    "python_version": "3.9.18",
    "numpy_version": "1.26.4",
    "cache_schema_version": "g2b0_pilot_v1",
    "sample_seed": 20261013,
}


def valid_payload():
    values = np.arange(807, dtype=np.float64)
    masks = np.ones(807, dtype=bool)
    support = np.ones(807, dtype=np.int64)
    payload = {
        "unique_feature_ids": np.asarray(MANIFEST.unique_ids),
        "unique_values": values,
        "unique_valid_mask": masks,
        "unique_support_count": support,
        **{key: np.asarray(value) for key, value in PROVENANCE.items()},
    }
    index = {uid: i for i, uid in enumerate(MANIFEST.unique_ids)}
    for view in MANIFEST.view_sizes:
        rows = [row for row in MANIFEST.rows if row["view_id"] == view]
        positions = np.asarray([index[row["unique_feature_id"]] for row in rows])
        payload[f"{view}_values"] = values[positions]
        payload[f"{view}_valid_mask"] = masks[positions]
        payload[f"{view}_support_count"] = support[positions]
    return payload


class G2B0SafetyTests(unittest.TestCase):
    def test_atomic_publish_roundtrip_and_existing_target_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"; payload = valid_payload()
            write_cache_atomic(path, payload)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): write_cache_atomic(path, payload)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(set(read_cache(path)), set(payload))

    def test_two_real_writers_have_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "race.npz"; barrier = threading.Barrier(2)
            def publish(marker):
                payload = {"values": np.full(200000, marker, dtype=np.float64)}
                barrier.wait()
                try:
                    write_cache_atomic(path, payload); return marker
                except FileExistsError:
                    return None
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                winners = list(pool.map(publish, (1.0, 2.0)))
            self.assertEqual(sum(winner is not None for winner in winners), 1)
            winner = next(winner for winner in winners if winner is not None)
            np.testing.assert_array_equal(read_cache(path)["values"], np.full(200000, winner))

    def test_failure_before_link_leaves_no_final_file_or_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interrupted.npz"
            with mock.patch.object(g2a_io.os, "link", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    write_cache_atomic(path, {"values": np.arange(3.0)})
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failure_after_link_removes_only_the_unverified_final(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "post-link-failure.npz"
            with mock.patch.object(
                g2a_io, "_verify_roundtrip",
                side_effect=(None, ValueError("injected final verification failure")),
            ):
                with self.assertRaisesRegex(ValueError, "final verification"):
                    write_cache_atomic(path, {"values": np.arange(3.0)})
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_complete_cache_validates_and_is_safely_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.npz"
            self.assertEqual(decide_resume_action(path, PROVENANCE, MANIFEST), Action.COMPUTE)
            write_cache_atomic(path, valid_payload())
            result = validate_existing_cache(path, PROVENANCE, MANIFEST)
            self.assertTrue(result.valid, result.errors)
            self.assertEqual(decide_resume_action(path, PROVENANCE, MANIFEST), Action.SKIP_VERIFIED)

    def test_every_provenance_mismatch_is_stop_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.npz"; write_cache_atomic(path, valid_payload())
            for key in PROVENANCE:
                with self.subTest(key=key):
                    wrong = dict(PROVENANCE); wrong[key] = 0 if key == "sample_seed" else "wrong"
                    result = validate_existing_cache(path, wrong, MANIFEST)
                    self.assertFalse(result.valid); self.assertIn(key, " ".join(result.errors))
                    self.assertEqual(decide_resume_action(path, wrong, MANIFEST), Action.STOP_INVALID)

    def test_corrupt_partial_shape_and_view_mapping_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt = root / "corrupt.npz"; corrupt.write_bytes(b"not-an-npz")
            self.assertEqual(decide_resume_action(corrupt, PROVENANCE, MANIFEST), Action.STOP_INVALID)
            for name, mutate in (
                ("partial", lambda p: p.pop("V5_values")),
                ("shape", lambda p: p.__setitem__("unique_values", p["unique_values"][:-1])),
                ("mapping", lambda p: p["V1_values"].__setitem__(0, -999.0)),
                ("extra", lambda p: p.__setitem__("y", np.asarray(1))),
            ):
                payload = valid_payload(); mutate(payload); path = root / f"{name}.npz"
                write_cache_atomic(path, payload)
                result = validate_existing_cache(path, PROVENANCE, MANIFEST)
                self.assertFalse(result.valid, name)
                self.assertEqual(decide_resume_action(path, PROVENANCE, MANIFEST), Action.STOP_INVALID)


if __name__ == "__main__":
    unittest.main(verbosity=2)
