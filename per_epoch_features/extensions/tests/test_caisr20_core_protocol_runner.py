import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class Caisr20CoreProtocolRunnerTests(unittest.TestCase):
    def test_route_scope_delta_uses_difference_of_route_aggregates(self):
        from extensions.caisr20_core_protocol_runner import _route_scope_delta

        core = {"I0002": 0.70, "I0006": 0.80, "S0001": 0.90}
        caisr = {"I0002": 0.75, "I0006": 0.60, "S0001": 0.85}
        self.assertAlmostEqual(_route_scope_delta(core, caisr, "Worst-site"), 0.10)
        self.assertAlmostEqual(_route_scope_delta(core, caisr, "Macro"), 0.06666666666666667)
        self.assertAlmostEqual(_route_scope_delta(core, caisr, "Small-site mean"), 0.075)

    def test_route_scope_delta_matches_frozen_formal_point_estimates(self):
        from extensions.caisr20_core_protocol_runner import _route_scope_delta

        core = {"I0002": 0.6961823966065748, "I0006": 0.6786724625681363, "S0001": 0.6709153732243733}
        caisr = {"I0002": 0.7343584305408272, "I0006": 0.6571448285379149, "S0001": 0.6116437229199803}
        self.assertAlmostEqual(_route_scope_delta(core, caisr, "Macro"), 0.014207750133453878)
        self.assertAlmostEqual(_route_scope_delta(core, caisr, "Worst-site"), 0.059271650304393075)
        self.assertAlmostEqual(_route_scope_delta(core, caisr, "Small-site mean"), -0.00832419995201561)

    def test_route_scope_delta_batch_preserves_macro_and_small_site_semantics(self):
        from extensions.caisr20_core_protocol_runner import _route_scope_delta_batch

        core = np.array([[0.70, 0.80, 0.90], [0.60, 0.90, 0.95]], dtype=np.float64)
        caisr = np.array([[0.75, 0.60, 0.85], [0.55, 0.85, 0.90]], dtype=np.float64)
        np.testing.assert_allclose(_route_scope_delta_batch(core, caisr, "Macro"), np.array([0.06666666666666667, 0.05]))
        np.testing.assert_allclose(_route_scope_delta_batch(core, caisr, "Small-site mean"), np.array([0.075, 0.05]))
        np.testing.assert_allclose(_route_scope_delta_batch(core, caisr, "Worst-site"), np.array([0.10, 0.05]))

    def test_private_unit_publish_is_no_clobber_and_resume_verified(self):
        from extensions.caisr20_core_protocol_runner import publish_unit
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unit"
            payload = {"scores": np.array([1.0, 2.0], dtype=np.float64)}
            meta = {"route_id": "CAISR20_CORE_PROTOCOL", "seed": 42}
            self.assertEqual(publish_unit(path, payload, meta), "BUILD_NEW")
            self.assertEqual(publish_unit(path, payload, meta), "SKIP_VERIFIED")
            with self.assertRaises(Exception):
                publish_unit(path, {"scores": np.array([9.0, 2.0], dtype=np.float64)}, meta)

    def test_source_validity_drift_is_not_called_semantic_equivalence(self):
        from extensions.caisr20_core_protocol import CAISR20_FEATURE_IDS
        from extensions.caisr20_core_protocol_runner import source_value_bridge
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = np.array(list(CAISR20_FEATURE_IDS) + [f"extra_{i}" for i in range(1444 - 20)], dtype="<U100")
            np.save(root / "unique_feature_ids.npy", ids)
            values = np.zeros((2, 1444), dtype=np.float64)
            values[:, :20] = 1.0
            values[0, 0] = np.nan
            mask = np.ones((2, 1444), dtype=bool)
            mask[0, 0] = False
            np.save(root / "X_raw.npy", values)
            np.save(root / "valid_mask.npy", mask)
            np.save(root / "row_sites.npy", np.array(["I0002", "I0002"]))
            raw = np.ones((2, 20), dtype=np.float64)
            manifest = pd.DataFrame({"site": ["I0002", "I0002"]})
            result = source_value_bridge(pool_root=root, raw_caisr=raw, manifest=manifest)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["value_mismatch_total"], 0)
            self.assertEqual(result["validity_mismatch_total"], 1)
            self.assertEqual(result["finite_overlap_status"], "PASS")
            self.assertEqual(result["validity_policy_status"], "DIAGNOSTIC")


    def test_source_value_mismatch_remains_blocking(self):
        from extensions.caisr20_core_protocol import CAISR20_FEATURE_IDS
        from extensions.caisr20_core_protocol_runner import source_value_bridge
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = np.array(list(CAISR20_FEATURE_IDS) + [f"extra_{i}" for i in range(1444 - 20)], dtype="<U100")
            np.save(root / "unique_feature_ids.npy", ids)
            values = np.ones((2, 1444), dtype=np.float64)
            values[0, 0] = 2.0
            np.save(root / "X_raw.npy", values)
            np.save(root / "valid_mask.npy", np.ones((2, 1444), dtype=bool))
            np.save(root / "row_sites.npy", np.array(["I0002", "I0002"]))
            manifest = pd.DataFrame({"site": ["I0002", "I0002"]})
            result = source_value_bridge(pool_root=root, raw_caisr=np.ones((2, 20)), manifest=manifest)
            self.assertEqual(result["status"], "DRIFT")
            self.assertEqual(result["value_mismatch_total"], 1)

    def test_source_identity_mismatch_is_blocking(self):
        from extensions.caisr20_core_protocol import CAISR20_FEATURE_IDS
        from extensions.caisr20_core_protocol_runner import RouteStop, source_value_bridge
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = np.array(list(CAISR20_FEATURE_IDS) + [f"extra_{i}" for i in range(1444 - 20)], dtype="<U100")
            np.save(root / "unique_feature_ids.npy", ids)
            np.save(root / "X_raw.npy", np.ones((2, 1444)))
            np.save(root / "valid_mask.npy", np.ones((2, 1444), dtype=bool))
            np.save(root / "row_sites.npy", np.array(["I0002", "I0006"]))
            manifest = pd.DataFrame({"site": ["I0002", "S0001"]})
            with self.assertRaises(RouteStop):
                source_value_bridge(pool_root=root, raw_caisr=np.ones((2, 20)), manifest=manifest)


if __name__ == "__main__":
    unittest.main()
