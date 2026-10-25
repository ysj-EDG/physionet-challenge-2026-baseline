import inspect
import unittest

import numpy as np


class RouteFusion50Tests(unittest.TestCase):
    def test_summary_and_bootstrap_rows_are_aggregate_only(self):
        from extensions.route_fusion_50_50 import build_bootstrap_rows, build_seed_rows

        scopes = ("I0002", "I0006", "S0001", "Macro", "Worst-site", "Small-site mean")
        seed_rows = build_seed_rows({"FUSION_MINUS_CORE": {scope: np.array([0.1, 0.2]) for scope in scopes}}, seeds=(42, 43))
        self.assertEqual(seed_rows[0]["comparison"], "FUSION_MINUS_CORE")
        self.assertEqual(seed_rows[0]["n_seeds"], 2)
        boot_rows = build_bootstrap_rows({"FUSION_MINUS_CORE": {scope: np.array([0.1, 0.2, 0.3]) for scope in scopes}}, {"FUSION_MINUS_CORE": {scope: 0.2 for scope in scopes}}, bootstrap_seed=20261023)
        self.assertEqual(boot_rows[0]["scope"], "I0002")
        self.assertEqual(boot_rows[0]["n_bootstrap"], 3)
        self.assertNotIn("scores", boot_rows[0])

    def test_fixed_arithmetic_fusion_does_not_rerank(self):
        from extensions.route_fusion_50_50 import fuse_scores

        core = np.array([0.1, 0.8, 0.3], dtype=np.float64)
        caisr = np.array([0.5, 0.2, 0.7], dtype=np.float64)
        np.testing.assert_array_equal(fuse_scores(core, caisr), np.array([0.3, 0.5, 0.5]))
        self.assertFalse(np.array_equal(fuse_scores(core, caisr), np.array([1.0 / 6.0, 5.0 / 6.0, 0.5])))

    def test_fusion_api_has_no_weight_search_surface(self):
        from extensions.route_fusion_50_50 import fuse_scores

        self.assertEqual(tuple(inspect.signature(fuse_scores).parameters), ("core_scores", "caisr_scores"))

    def test_parent_identity_rejects_count_order_seed_and_nonfinite_drift(self):
        from extensions.route_fusion_50_50 import FusionStop, validate_parent_pair

        indices = np.array([10, 11, 12], dtype=np.int64)
        core = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        caisr = np.array([0.3, 0.2, 0.1], dtype=np.float64)
        validate_parent_pair(indices, indices, core, caisr, holdout="I0002", seed=42)
        cases = [
            (indices[:2], indices, core[:2], caisr, 42),
            (indices, indices[[1, 0, 2]], core, caisr, 42),
            (indices, indices, core, np.array([0.3, np.nan, 0.1]), 42),
            (np.array([10, 10, 12], dtype=np.int64), np.array([10, 10, 12], dtype=np.int64), core, caisr, 42),
            (indices, indices, core, caisr, 42, 43),
        ]
        for case in cases:
            left_idx, right_idx, left, right, seed = case[:5]
            peer_seed = case[5] if len(case) == 6 else None
            with self.assertRaises(FusionStop):
                validate_parent_pair(left_idx, right_idx, left, right, holdout="I0002", seed=seed, caisr_seed=peer_seed)

    def test_seed_fusion_is_arithmetic_and_not_ranked(self):
        from extensions.route_fusion_50_50 import build_seed_fusion

        core = np.array([0.1, 0.8, 0.3], dtype=np.float64)
        caisr = np.array([0.5, 0.2, 0.7], dtype=np.float64)
        np.testing.assert_array_equal(build_seed_fusion(core, caisr), np.array([0.3, 0.5, 0.5]))

    def test_route_worst_site_subtracts_route_minima(self):
        from extensions.route_fusion_50_50 import route_scope_delta

        fusion = {"I0002": 0.70, "I0006": 0.80, "S0001": 0.75}
        core = {"I0002": 0.60, "I0006": 0.82, "S0001": 0.78}
        self.assertAlmostEqual(route_scope_delta(fusion, core, "Worst-site"), 0.10)
        self.assertNotAlmostEqual(route_scope_delta(fusion, core, "Worst-site"), min(fusion[s] - core[s] for s in fusion))

    def test_route_scope_delta_batch_uses_route_minima(self):
        from extensions.route_fusion_50_50 import route_scope_delta_batch

        left = np.array([[0.70, 0.80, 0.75], [0.60, 0.90, 0.95]], dtype=np.float64)
        right = np.array([[0.60, 0.82, 0.78], [0.55, 0.85, 0.90]], dtype=np.float64)
        np.testing.assert_allclose(route_scope_delta_batch(left, right, "Worst-site"), np.array([0.10, 0.05]))
        np.testing.assert_allclose(route_scope_delta_batch(left, right, "Macro"), left.mean(axis=1) - right.mean(axis=1))
        np.testing.assert_allclose(route_scope_delta_batch(left, right, "Small-site mean"), left[:, :2].mean(axis=1) - right[:, :2].mean(axis=1))

    def test_shared_bootstrap_series_uses_one_multiplicity_matrix_for_all_routes(self):
        from extensions.route_fusion_50_50 import bootstrap_route_series

        pairs = {
            "CORE": {"positive": np.array([0]), "negative": np.array([1]), "contribution": np.array([1.0])},
            "CAISR20": {"positive": np.array([0]), "negative": np.array([1]), "contribution": np.array([0.0])},
            "FUSION_50_50": {"positive": np.array([0]), "negative": np.array([1]), "contribution": np.array([0.5])},
        }
        multiplicities = np.array([[1, 1], [2, 1], [1, 2]], dtype=np.int64)
        series = bootstrap_route_series(pairs, multiplicities)
        np.testing.assert_array_equal(series["CORE"], np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(series["CAISR20"], np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(series["FUSION_50_50"], np.array([0.5, 0.5, 0.5]))

    def test_private_score_publish_is_atomic_and_no_clobber(self):
        import tempfile
        from pathlib import Path
        from extensions.route_fusion_50_50 import FusionStop, publish_score_unit

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "final" / "I0002"
            indices = np.array([10, 11], dtype=np.int64)
            scores = np.array([0.2, 0.4], dtype=np.float64)
            self.assertEqual(publish_score_unit(target, indices, scores, {"holdout": "I0002"}), "BUILD_NEW")
            self.assertEqual(publish_score_unit(target, indices, scores, {"holdout": "I0002"}), "SKIP_VERIFIED")
            with self.assertRaises(FusionStop):
                publish_score_unit(target, indices, np.array([0.9, 0.4]), {"holdout": "I0002"})

    def test_scope_rows_include_only_two_fixed_comparisons(self):
        from extensions.route_fusion_50_50 import build_scope_rows

        routes = {
            "CAISR20": {"I0002": 0.70, "I0006": 0.60, "S0001": 0.50},
            "CORE": {"I0002": 0.65, "I0006": 0.65, "S0001": 0.55},
            "FUSION_50_50": {"I0002": 0.72, "I0006": 0.66, "S0001": 0.56},
        }
        rows = build_scope_rows(routes)
        self.assertEqual({row["row_type"] for row in rows}, {"route_summary", "comparison"})
        self.assertEqual({row.get("comparison") for row in rows if row["row_type"] == "comparison"}, {"FUSION_MINUS_CORE", "FUSION_MINUS_CAISR"})
        worst = [row for row in rows if row.get("comparison") == "FUSION_MINUS_CORE" and row["scope"] == "Worst-site"][0]
        self.assertAlmostEqual(worst["delta"], 0.01)

    def test_route_scope_delta_linear_scopes_are_explicit(self):
        from extensions.route_fusion_50_50 import route_scope_delta

        fusion = {"I0002": 0.70, "I0006": 0.80, "S0001": 0.75}
        core = {"I0002": 0.60, "I0006": 0.82, "S0001": 0.78}
        self.assertAlmostEqual(route_scope_delta(fusion, core, "Macro"), np.mean([0.10, -0.02, -0.03]))
        self.assertAlmostEqual(route_scope_delta(fusion, core, "Small-site mean"), np.mean([0.10, -0.02]))


if __name__ == "__main__":
    unittest.main()
