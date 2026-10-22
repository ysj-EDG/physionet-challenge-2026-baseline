import unittest
import numpy as np

from extensions.compare_e0_e3_v2 import (
    EXPERIMENT_SOURCES, fixed_column_keys, fuse_prediction_columns,
    normalize_score_type,
)


class CompareE0E3Tests(unittest.TestCase):
    def test_fixed_experiment_sources(self):
        self.assertEqual(EXPERIMENT_SOURCES["E0"], {v: "old" for v in ("V1", "V2", "V3", "V4", "V5")})
        self.assertEqual(EXPERIMENT_SOURCES["E1"]["V5"], "old")
        self.assertEqual(EXPERIMENT_SOURCES["E2"]["V5"], "v2")
        self.assertEqual(EXPERIMENT_SOURCES["E3"]["V1"], "v2")
        self.assertEqual(len(fixed_column_keys()), 225)

    def test_score_type_bridge_is_explicit(self):
        self.assertEqual(normalize_score_type("raw_ranking_score"), "lambdarank_raw")
        self.assertEqual(normalize_score_type("lambdarank_raw"), "lambdarank_raw")
        with self.assertRaises(ValueError):
            normalize_score_type("unknown")

    def test_average_ties_then_mean(self):
        base = np.array([[0., 1., 0.], [1., 1., 2.], [2., 0., 2.]], dtype=np.float64)
        values = np.tile(base, 75)
        indices, fused, ranks = fuse_prediction_columns(values, np.array([4, 5, 6], dtype=np.int64))
        self.assertTrue(np.array_equal(indices, np.array([4, 5, 6])))
        expected = np.array([[1/6, 2/3, 1/6], [1/2, 2/3, 2/3], [5/6, 1/6, 2/3]])
        np.testing.assert_allclose(ranks[:, :3], expected)
        np.testing.assert_allclose(fused, expected.mean(axis=1))


if __name__ == "__main__":
    unittest.main()
