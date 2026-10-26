import unittest

import numpy as np
import pandas as pd


class SirenMatchaFusionTests(unittest.TestCase):
    def test_average_tie_rank_matches_existing_siren_definition(self):
        from extensions.siren_matcha_ab_fusion import average_tie_percentile_rank

        values = np.array([3.0, 1.0, 1.0, 2.0], dtype=np.float64)
        np.testing.assert_allclose(
            average_tie_percentile_rank(values),
            np.array([0.875, 0.25, 0.25, 0.625]),
        )

    def test_exact_join_rejects_duplicate_and_missing_keys(self):
        from extensions.siren_matcha_ab_fusion import FusionStop, join_sources

        a = pd.DataFrame({
            "global_index": [0, 1], "site": ["I0002", "I0002"],
            "record_id": ["r0", "r1"], "label": [0, 1], "age": [50.0, 60.0],
            "A_final": [0.2, 0.8],
        })
        b = pd.DataFrame({
            "site": ["I0002", "I0002"], "record_id": ["r0", "r1"],
            "label": [0, 1], "age": [50.0, 60.0],
            "B_seed_20260806_logit": [0.1, 0.2],
            "B_seed_20260807_logit": [0.1, 0.2],
            "B_seed_20260808_logit": [0.1, 0.2],
        })
        joined = join_sources(a, b)
        self.assertEqual(list(joined["record_id"]), ["r0", "r1"])
        self.assertEqual(list(joined["global_index"]), [0, 1])
        with self.assertRaises(FusionStop):
            join_sources(a, pd.concat([b, b.iloc[[0]]], ignore_index=True))
        with self.assertRaises(FusionStop):
            join_sources(a, b.iloc[[0]].copy())

    def test_b_final_uses_native_logit_mean(self):
        from extensions.siren_matcha_ab_fusion import build_b_final

        frame = pd.DataFrame({
            "B_seed_20260806_logit": [1.0, -1.0],
            "B_seed_20260807_logit": [2.0, -2.0],
            "B_seed_20260808_logit": [3.0, -3.0],
        })
        np.testing.assert_array_equal(build_b_final(frame), np.array([2.0, -2.0]))

    def test_fixed_primary_and_control_weights_have_no_search_surface(self):
        from extensions.siren_matcha_ab_fusion import build_fusions

        a = np.array([0.2, 0.8], dtype=np.float64)
        b = np.array([0.6, 0.4], dtype=np.float64)
        result = build_fusions(a, b)
        np.testing.assert_array_equal(result["AB60"], np.array([0.36, 0.64]))
        np.testing.assert_allclose(result["AB50"], np.array([0.4, 0.6]))
        self.assertEqual(set(result), {"AB60", "AB50"})

    def test_paired_bootstrap_uses_shared_multiplicities(self):
        from extensions.siren_matcha_ab_fusion import paired_bootstrap_delta

        scores = {
            "A": np.array([0.1, 0.9, 0.2, 0.8]),
            "B": np.array([0.2, 0.8, 0.3, 0.7]),
        }
        labels = np.array([0, 1, 0, 1], dtype=np.int8)
        ages = np.array([50.0, 50.0, 60.0, 60.0])
        result = paired_bootstrap_delta(labels, ages, scores["A"], scores["B"], n_bootstrap=5, seed=20261023)
        self.assertEqual(result["valid"] + result["invalid"], 5)
        self.assertGreaterEqual(result["valid"], 1)
        self.assertEqual(len(result["draws"]), result["valid"])


if __name__ == "__main__":
    unittest.main()
