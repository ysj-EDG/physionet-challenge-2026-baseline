import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class Caisr20CoreProtocolRunnerTests(unittest.TestCase):
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
            self.assertEqual(result["status"], "DRIFT")
            self.assertEqual(result["value_mismatch_total"], 0)
            self.assertEqual(result["validity_mismatch_total"], 1)


if __name__ == "__main__":
    unittest.main()
