import tempfile
import unittest
from pathlib import Path
import numpy as np

from extensions.multiseed_runner_v2 import publish_prediction_unit_multiseed, UnitIdentityError


class MultiSeedRunnerTests(unittest.TestCase):
    def test_prediction_no_clobber_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unit"
            payload = {"test_global_indices": np.array([1, 2], dtype=np.int64),
                       "scores": np.array([0.1, 0.9], dtype=np.float64)}
            metadata = {"source_version": "v2", "seed": 43, "view": "V1", "model_type": "lr"}
            self.assertEqual(publish_prediction_unit_multiseed(path, payload, metadata), "BUILD_NEW")
            original = (path / "prediction.npz").read_bytes()
            self.assertEqual(publish_prediction_unit_multiseed(path, payload, metadata), "SKIP_VERIFIED")
            self.assertEqual((path / "prediction.npz").read_bytes(), original)
            with self.assertRaises(UnitIdentityError):
                publish_prediction_unit_multiseed(path, payload, {**metadata, "seed": 44})


if __name__ == "__main__":
    unittest.main()
