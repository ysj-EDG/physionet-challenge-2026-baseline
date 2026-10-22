import unittest
import numpy as np

from extensions.train_multiseed_v2 import (
    MODEL_TYPES, prepare_training_v2, fit_multiseed_model_v2,
)
from extensions.train_inputs_v2 import V2FoldInputs


def _fold():
    n, p = 24, 4
    x = np.arange(n * p, dtype=np.float64).reshape(n, p) / 10.0
    y = np.array([0, 1] * (n // 2), dtype=np.int8)
    ages = np.array([40 + 5 * (i % 3) for i in range(n)], dtype=np.float64)
    return V2FoldInputs(x, x[:8].copy(), np.ones_like(x, bool), np.ones((8, p), bool), y,
                        ages, np.floor(ages / 5).astype(np.int64), tuple(f"f{i}" for i in range(p)),
                        np.arange(n, dtype=np.int64), np.arange(8, dtype=np.int64),
                        "I0002", "S0001", "transductive", "V1", {})


class MultiSeedTrainingTests(unittest.TestCase):
    def test_seed_is_restricted_and_changes_sampling_identity(self):
        fold = _fold()
        a = prepare_training_v2(fold, 43)
        b = prepare_training_v2(fold, 44)
        self.assertNotEqual(a.sample.index_sha256, b.sample.index_sha256)
        with self.assertRaises(ValueError):
            prepare_training_v2(fold, 42)

    def test_actual_estimator_random_state_is_seed(self):
        fold = _fold()
        prepared = prepare_training_v2(fold, 43)
        trained = fit_multiseed_model_v2("lr", prepared)
        self.assertEqual(trained.model.get_params()["random_state"], 43)
        self.assertEqual(trained.parameters["random_state"], 43)

    def test_frozen_config_is_not_mutated(self):
        fold = _fold()
        prepared = prepare_training_v2(fold, 44)
        fit_multiseed_model_v2("lr", prepared)
        self.assertEqual(fold.X_train.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
