import unittest


class Caisr20CoreProtocolTests(unittest.TestCase):
    def test_frozen_manifest_and_model_contract(self):
        from extensions.caisr20_core_protocol import CAISR20_FEATURE_IDS, CAISR20_X_STATIC_INDICES, model_config
        self.assertEqual(len(CAISR20_FEATURE_IDS), 20)
        self.assertEqual(len(set(CAISR20_FEATURE_IDS)), 20)
        self.assertEqual(CAISR20_X_STATIC_INDICES, (11, 12, 13, 14, 16, 17, 18, 19, 20, 32, 34, 87, 96, 97, 98, 126, 127, 141, 168, 173))
        self.assertNotIn("demo_", " ".join(CAISR20_FEATURE_IDS).lower())
        self.assertEqual(model_config(42)["random_state"], 42)
        self.assertEqual(model_config(56)["C"], 0.03)

    def test_selected_row_preprocessing_does_not_use_holdout(self):
        import numpy as np
        from extensions.caisr20_core_protocol import fit_selected_preprocessor, apply_selected_preprocessor
        train = np.full((2, 20), 1.0, dtype=np.float64)
        train[1, 0] = 3.0
        train[0, 1] = np.nan
        holdout = np.full((1, 20), 1000.0, dtype=np.float64)
        state = fit_selected_preprocessor(train)
        transformed = apply_selected_preprocessor(holdout, state)
        self.assertEqual(state.median[:2].tolist(), [2.0, 1.0])
        self.assertNotEqual(float(transformed[0, 0]), 0.0)

    def test_rank_mean_and_deltas_are_deterministic(self):
        import numpy as np
        from extensions.caisr20_core_protocol import mean_rank_fusion, route_delta, interaction_delta
        scores = np.tile(np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64), (1, 8))[:, :15]
        fused = mean_rank_fusion(scores)
        np.testing.assert_allclose(fused, [0.48333333333333334, 0.5166666666666666])
        self.assertAlmostEqual(route_delta(0.7, 0.6), 0.1)
        self.assertAlmostEqual(interaction_delta(0.8, 0.7, 0.6, 0.5), 0.0)

    def test_typed_transform_preserves_valid_zero_and_marks_nan(self):
        import numpy as np
        from extensions.caisr20_core_protocol import typed_caisr20
        rules = []
        for i in range(20):
            rules.append({"unit_scale": 1.0, "nonlinear": "identity", "missing_rule": "nonfinite", "expected_range": {"hard": False}})
        values = np.zeros((2, 20), dtype=np.float64)
        values[0, 3] = np.nan
        out = typed_caisr20(values, rules)
        self.assertEqual(out[1, 3], 0.0)
        self.assertTrue(np.isnan(out[0, 3]))

    def test_historical_typed_fit_uses_rules_and_sentinel(self):
        import numpy as np
        from extensions.caisr20_core_protocol import (
            apply_historical_typed_caisr20,
            fit_historical_typed_caisr20,
        )
        rules = []
        for i in range(20):
            rules.append({
                "unit_scale": 1.0,
                "nonlinear": "identity",
                "scaling": "robust" if i == 0 else "identity",
                "missing_rule": "nonfinite",
                "expected_range": {"hard": False},
                "identity_fill": 0.0,
            })
        full_static = np.ones((4, 196), dtype=np.float64)
        full_static[:, 11:31] = np.array([
            np.arange(20), np.arange(20) + 1, np.arange(20) + 2,
            np.zeros(20),
        ], dtype=np.float64)
        full_static[3, 10:] = 0.0
        state = fit_historical_typed_caisr20(full_static[:3], rules)
        transformed = apply_historical_typed_caisr20(full_static, rules, state)
        self.assertEqual(transformed.shape, (4, 20))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.all(transformed[3, 1:] == 0.0))
        self.assertEqual(state.fit_valid_counts[0], 3)

    def test_historical_typed_scaler_uses_selected_rows_only(self):
        import numpy as np
        from extensions.caisr20_core_protocol import fit_historical_typed_caisr20
        rules = [{
            "unit_scale": 1.0, "nonlinear": "identity", "scaling": "robust",
            "missing_rule": "nonfinite", "expected_range": {"hard": False},
            "identity_fill": 0.0,
        } for _ in range(20)]
        static = np.ones((3, 196), dtype=np.float64)
        static[:, 11] = [1.0, 2.0, 10000.0]
        state = fit_historical_typed_caisr20(static[:2], rules)
        self.assertLess(state.center[0], 10.0)
        self.assertEqual(state.fit_valid_counts[0], 2)

    def test_historical_typed_log1p_and_hard_range_are_exact(self):
        import numpy as np
        from extensions.caisr20_core_protocol import apply_historical_typed_caisr20, fit_historical_typed_caisr20
        rules = [{
            "unit_scale": 1.0, "nonlinear": "log1p", "scaling": "robust",
            "missing_rule": "nonfinite", "expected_range": {"hard": True, "min": 0.0, "max": 10.0},
            "range_tolerance": 1e-6, "identity_fill": 0.0,
        }] + [{
            "unit_scale": 1.0, "nonlinear": "identity", "scaling": "identity",
            "missing_rule": "nonfinite", "expected_range": {"hard": False},
            "identity_fill": 0.0,
        } for _ in range(19)]
        static = np.ones((3, 196), dtype=np.float64)
        static[:, 11] = [0.0, 2.0, 2.0 + 5e-7]
        state = fit_historical_typed_caisr20(static, rules)
        out = apply_historical_typed_caisr20(static, rules, state)
        self.assertTrue(np.isfinite(out).all())
        self.assertAlmostEqual(float(out[1, 0]), 0.0)
        self.assertEqual(state.scale_method[0], "iqr")

    def test_historical_typed_all_missing_has_explicit_fallback(self):
        import numpy as np
        from extensions.caisr20_core_protocol import fit_historical_typed_caisr20
        rules = [{
            "unit_scale": 1.0, "nonlinear": "identity", "scaling": "robust",
            "missing_rule": "nonfinite", "expected_range": {"hard": False},
            "identity_fill": 0.0,
        } for _ in range(20)]
        static = np.ones((2, 196), dtype=np.float64)
        static[:, 11] = np.nan
        state = fit_historical_typed_caisr20(static, rules)
        self.assertEqual(state.scale_method[0], "all_missing_fallback")
        self.assertTrue(state.all_missing[0])
        self.assertEqual(int(state.fit_valid_counts[0]), 0)


if __name__ == "__main__":
    unittest.main()
