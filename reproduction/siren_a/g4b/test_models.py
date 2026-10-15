import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproduction.siren_a.g4.models import load_checkpoint, predict_scores, save_checkpoint
from reproduction.siren_a.g4.queries import build_queries
from reproduction.siren_a.g4b.models import fit_seed_model


class SeedModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(902)
        self.X = rng.normal(size=(80, 7)).astype(np.float64)
        self.X[::6, 2] = np.nan
        self.X[:, 6] = np.nan
        self.y = np.tile(np.array([0, 1], dtype=np.int8), 40)
        self.ages = np.r_[np.full(40, 55.0), np.full(40, 60.0)]
        self.ids = tuple(f"f{i}" for i in range(self.X.shape[1]))
        self.indices = np.arange(1000, 1080, dtype=np.int64)
        positions = np.arange(80, dtype=np.int64)
        self.query = build_queries(self.ages, self.y, positions, self.indices)

    def _fit_and_roundtrip(self, model_type, seed):
        trained = fit_seed_model(
            model_type,
            seed,
            self.X,
            self.y,
            self.ids,
            self.indices,
            self.query,
        )
        self.assertEqual(trained.parameters["random_state"], seed)
        self.assertEqual(trained.adapter.fit_scope, "seed_subset")
        expected_index_sha = hashlib.sha256()
        expected_index_sha.update(str(self.indices.shape).encode("ascii"))
        expected_index_sha.update(self.indices.tobytes(order="C"))
        self.assertEqual(trained.adapter.fit_indices_sha256, expected_index_sha.hexdigest())
        scores = predict_scores(model_type, trained.model, trained.adapter, self.X)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(np.ptp(scores), 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            files = save_checkpoint(path, trained)
            loaded = load_checkpoint(path, model_type, files)
            replay = predict_scores(model_type, loaded.model, loaded.adapter, self.X)
            np.testing.assert_allclose(scores, replay, rtol=0.0, atol=1e-12)
        return trained

    def test_lr_uses_selected_subset_adapter_and_current_seed(self):
        trained = self._fit_and_roundtrip("lr", 43)
        self.assertFalse(np.isnan(trained.train_matrix).any())
        self.assertEqual(trained.adapter.finite_counts[-1], 0)
        self.assertEqual(trained.adapter.fill_values[-1], 128.0 / 255.0)
        self.assertEqual(trained.model.random_state, 43)

    def test_hgb_preserves_native_nan_and_current_seed(self):
        trained = self._fit_and_roundtrip("hgb", 44)
        self.assertTrue(np.isnan(trained.train_matrix).any())
        self.assertEqual(trained.model.random_state, 44)
        self.assertFalse(trained.parameters["early_stopping"])

    def test_lambdarank_preserves_nan_query_and_current_seed(self):
        trained = self._fit_and_roundtrip("lambdarank", 56)
        self.assertTrue(np.isnan(trained.train_matrix).any())
        self.assertEqual(trained.parameters["lambdarank_truncation_level"], 40)
        self.assertEqual(trained.model.get_params()["random_state"], 56)
        self.assertEqual(trained.parameters["label_gain"], [0, 1])

    def test_bad_seed_or_query_length_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_seed_model("lr", 42, self.X, self.y, self.ids, self.indices, self.query)
        short_query = build_queries(
            self.ages[:-2],
            self.y[:-2],
            np.arange(78, dtype=np.int64),
            self.indices[:-2],
        )
        with self.assertRaises(ValueError):
            fit_seed_model("lambdarank", 43, self.X, self.y, self.ids, self.indices, short_query)


if __name__ == "__main__":
    unittest.main()
