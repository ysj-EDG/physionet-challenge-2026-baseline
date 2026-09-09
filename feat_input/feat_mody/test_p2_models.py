#!/usr/bin/env python
"""Focused P2 integration tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

import team_code
from feature_scaling import FeatureScaler, SCALER_VERSION
from static_logistic import MODEL_TYPE, STATIC_DIM, StaticLogisticModel
from train_lstm import LSTMModel, resolve_pos_weight, seed_everything

B_PATH = ROOT / "output" / "p0_ab" / "seed7" / "B_typed_v1" / "lstm_model.pt"


class P2ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b_checkpoint = torch.load(B_PATH, map_location="cpu", weights_only=False)

    def test_loss_control_preserves_initial_weights(self):
        seed_everything(7)
        model_a = LSTMModel()
        seed_everything(7)
        model_b = LSTMModel()
        for key, value in model_a.state_dict().items():
            torch.testing.assert_close(value, model_b.state_dict()[key])
        logits = torch.tensor([0.5, -0.25, 1.0, -1.5])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        empirical = resolve_pos_weight("empirical", 53, 680)
        unit = resolve_pos_weight("unit", 53, 680)
        self.assertAlmostEqual(empirical, 680 / 53)
        self.assertEqual(unit, 1.0)
        for weight in (empirical, unit):
            loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([weight])
            )(logits, labels)
            self.assertTrue(torch.isfinite(loss))

    def test_old_b_defaults_to_lstm(self):
        loaded = team_code.load_model(B_PATH.parent, verbose=False)
        self.assertEqual(loaded["model_type"], "lstm")
        self.assertIsInstance(loaded["network"], team_code.LSTMModel)
        self.assertEqual(loaded["preprocessor"].mode, SCALER_VERSION)
        self.assertIsNone(loaded["preprocessor"].clip_z)
        self.assertEqual(
            loaded["preprocessor"].mask_config,
            {"x_static_zero_indices": [], "x_seq_zero_ranges": []},
        )

    def test_static_model_matches_sklearn_logit(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(80, STATIC_DIM))
        y = np.asarray([0, 1] * 40)
        selected = [0, 3, 9, 11, 32]
        scaler = StandardScaler().fit(X)
        scaled = scaler.transform(X)
        classifier = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.4, C=0.03,
            class_weight="balanced", max_iter=10000, tol=1e-4,
            random_state=7, fit_intercept=True,
        ).fit(scaled[:, selected], y)
        model = StaticLogisticModel(
            selected, scaler.mean_, scaler.scale_,
            classifier.coef_[0], classifier.intercept_[0],
        )
        actual = model(
            torch.zeros(len(X), 1, 483),
            torch.zeros(len(X), 1, 12),
            torch.from_numpy(X).float(),
            torch.ones(len(X), dtype=torch.long),
        ).detach().numpy()
        expected = classifier.decision_function(scaled[:, selected])
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_static_checkpoint_keeps_states_separate(self):
        selected = [0, 9, 11]
        mean = np.linspace(-1.0, 1.0, STATIC_DIM)
        scale = np.linspace(1.0, 2.0, STATIC_DIM)
        coef = [0.25, -0.5, 0.75]
        network = StaticLogisticModel(selected, mean, scale, coef, -0.1)
        checkpoint = {
            "model_type": MODEL_TYPE,
            "state_dict": network.state_dict(),
            "feature_dims": {"X_seq": 483, "X_ecg": 12, "x_static": 196},
            "input_preprocessing": self.b_checkpoint["input_preprocessing"],
            "selected_indices": selected,
            "linear_scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
            "linear_model": {"coefficients": coef, "intercept": -0.1},
            "threshold": 0.4,
        }
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "lstm_model.pt"
            torch.save(checkpoint, path)
            loaded = team_code.load_model(path.parent, verbose=False)
        self.assertEqual(loaded["model_type"], MODEL_TYPE)
        self.assertIsInstance(loaded["network"], StaticLogisticModel)
        self.assertEqual(loaded["preprocessor"].mode, SCALER_VERSION)
        self.assertEqual(
            loaded["network"].selected_indices.cpu().tolist(), selected
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
