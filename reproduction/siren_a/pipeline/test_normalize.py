import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from normalize import (apply_rank_reference, fit_rank_reference, quantize_rank,
                       rank_by_site)


class NormalizeTests(unittest.TestCase):
    def test_average_ties_match_frozen_example(self):
        x = np.array([[2.0], [2.0], [5.0], [9.0]])
        mask = np.ones_like(x, bool)
        result, valid = rank_by_site(x, mask, np.array(["A"] * 4))
        np.testing.assert_array_equal(valid, mask)
        np.testing.assert_allclose(result[:, 0], [0.25, 0.25, 0.625, 0.875])

    def test_continuous_binary_constant_singleton_and_zero(self):
        x = np.array([[0., 0., 4., 7.], [1., 0., 4., np.nan], [2., 1., 4., np.nan]])
        m = np.isfinite(x)
        p, ok = rank_by_site(x, m, np.array(["A"] * 3))
        np.testing.assert_allclose(p[:, 0], [1/6, 0.5, 5/6])
        np.testing.assert_allclose(p[:, 1], [1/3, 1/3, 5/6])
        np.testing.assert_allclose(p[:, 2], 0.5)
        self.assertEqual(p[0, 3], 0.5); self.assertFalse(ok[1:, 3].any())

    def test_all_missing_and_source_rank_masks_are_separate(self):
        train = np.array([[np.nan], [np.nan]])
        ref = fit_rank_reference(train, np.zeros_like(train, bool))
        test = np.array([[3.0]])
        source = np.ones_like(test, bool)
        p, rank = apply_rank_reference(test, source, ref)
        self.assertTrue(source[0, 0]); self.assertFalse(rank[0, 0]); self.assertTrue(np.isnan(p[0, 0]))

    def test_inductive_endpoints_and_outside_values(self):
        ref = fit_rank_reference(np.array([[2.], [2.], [5.], [9.]]), np.ones((4, 1), bool))
        p, ok = apply_rank_reference(np.array([[1.], [2.], [9.], [10.]]), np.ones((4, 1), bool), ref)
        np.testing.assert_allclose(p[:, 0], [0., 0.25, 0.875, 1.]); self.assertTrue(ok.all())

    def test_negative_float_and_inf_rejection(self):
        x = np.array([[-2.5], [0.0], [3.25]])
        p, _ = rank_by_site(x, np.ones_like(x, bool), np.array(["A"] * 3))
        self.assertTrue(np.all(np.diff(p[:, 0]) > 0))
        with self.assertRaises(ValueError):
            rank_by_site(np.array([[np.inf]]), np.ones((1, 1), bool), np.array(["A"]))

    def test_order_invariance(self):
        x = np.array([[4.], [1.], [4.], [8.]])
        m = np.ones_like(x, bool); sites = np.array(["A"] * 4)
        p, _ = rank_by_site(x, m, sites)
        order = np.array([2, 0, 3, 1]); inverse = np.argsort(order)
        shuffled, _ = rank_by_site(x[order], m[order], sites[order])
        np.testing.assert_array_equal(p, shuffled[inverse])

    def test_site_isolation_and_inductive_reference_is_immutable(self):
        x = np.array([[0.], [10.], [100.], [200.]])
        sites = np.array(["train", "train", "test", "test"]); m = np.ones_like(x, bool)
        p, _ = rank_by_site(x, m, sites)
        np.testing.assert_allclose(p[:, 0], [0.25, 0.75, 0.25, 0.75])
        ref = fit_rank_reference(x[:2], m[:2]); before = ref.fingerprint
        apply_rank_reference(x[2:], m[2:], ref)
        altered = x[2:] * 99; apply_rank_reference(altered, m[2:], ref)
        self.assertEqual(before, ref.fingerprint)

    def test_quantization_half_up_grid_error_and_missing(self):
        u = np.array([[0., 0.5, 1., np.nan], [0.1, 127.5/255, 0.9, np.nan]])
        mask = np.isfinite(u)
        q, levels = quantize_rank(u, mask)
        self.assertEqual(levels[0, 1], 128); self.assertEqual(q[0, 1], 128/255)
        self.assertEqual(levels[~mask].tolist(), [-1, -1]); self.assertTrue(np.isnan(q[~mask]).all())
        self.assertTrue(np.all((levels[mask] >= 0) & (levels[mask] <= 255)))
        self.assertLessEqual(float(np.max(np.abs(q[mask] - u[mask]))), 1/510 + 1e-15)
        self.assertTrue(np.all(np.diff(q[0, :3]) >= 0)); self.assertLessEqual(len(np.unique(levels[mask])), 256)

    def test_quantization_rejects_bad_percentile_and_mask(self):
        with self.assertRaises(ValueError): quantize_rank(np.array([[1.1]]), np.ones((1,1), bool))
        with self.assertRaises(ValueError): quantize_rank(np.array([[np.nan]]), np.ones((1,1), bool))


if __name__ == "__main__": unittest.main(verbosity=2)
