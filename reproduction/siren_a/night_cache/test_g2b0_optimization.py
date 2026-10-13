import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import g2a_features
from g2a_features import compute_night_features, pool_values, spatial_value_and_mask
from g2a_io import load_manifest, load_npz_record, read_cache
from test_g2a_features import record

MANIFEST_PATH = ROOT / "docs" / "G1B_FEATURE_MANIFEST_FROZEN.csv"
GOLDEN_ROOT = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g2a-private/g2a_smoke_v1")


class G2B0OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest(MANIFEST_PATH)

    def test_compute_converts_full_xseq_to_float64_once(self):
        rec = record(t=8, w=8)
        original_xseq = rec["X_seq"].astype(np.float32)
        rec["X_seq"] = original_xseq
        real_asarray = np.asarray
        conversions = 0

        def counting_asarray(value, *args, **kwargs):
            nonlocal conversions
            dtype = kwargs.get("dtype", args[0] if args else None)
            if value is original_xseq and dtype in (float, np.float64):
                conversions += 1
            return real_asarray(value, *args, **kwargs)

        with mock.patch.object(g2a_features.np, "asarray", side_effect=counting_asarray):
            result = compute_night_features(rec, self.manifest)
        self.assertEqual(result.unique_feature_values.dtype, np.float64)
        self.assertEqual(conversions, 1)

    def test_public_math_and_spatial_compatibility(self):
        value, valid, support = pool_values(np.arange(5.0), np.ones(5, bool), "Mean")
        self.assertEqual((value, valid, support), (2.0, True, 5))
        value, valid, support = pool_values(np.array([0.0, 1.0, 2.0, 3.0, 8.0]), np.ones(5, bool), "IQR")
        self.assertEqual((value, valid, support), (2.0, True, 5))
        row = next(r for r in self.manifest.rows if r["unique_feature_id"] == "spatial_asym_N2_F3-F4_delta_mean")
        rec = record(); indices = [int(x) for x in row["source_indices"].split(";")]
        rec["X_seq"][:, indices[0]] = 2.0; rec["X_seq"][:, indices[1]] = 1.0
        self.assertEqual(spatial_value_and_mask(row, rec), (1.0, True, 5))

    @unittest.skipUnless(GOLDEN_ROOT.is_dir(), "local G2-A Golden Cache unavailable")
    def test_all_18_golden_caches_match_exactly(self):
        mapping = json.loads((GOLDEN_ROOT / "private_sample_map.json").read_text())
        self.assertEqual(len(mapping), 18)
        for item in mapping:
            current = compute_night_features(load_npz_record(Path(item["source"])), self.manifest)
            golden = read_cache(GOLDEN_ROOT / item["cache"])
            self.assertEqual(current.unique_feature_values.dtype, golden["unique_values"].dtype)
            np.testing.assert_array_equal(current.unique_feature_values, golden["unique_values"])
            np.testing.assert_array_equal(current.unique_feature_valid_mask, golden["unique_valid_mask"])
            np.testing.assert_array_equal(current.unique_feature_support_count, golden["unique_support_count"])
            for view in self.manifest.view_sizes:
                np.testing.assert_array_equal(current.view_values[view], golden[f"{view}_values"])
                np.testing.assert_array_equal(current.view_valid_masks[view], golden[f"{view}_valid_mask"])
                np.testing.assert_array_equal(current.view_support_counts[view], golden[f"{view}_support_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
