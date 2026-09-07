#!/usr/bin/env python
"""Engineering tests for typed_v1 shared input preprocessing."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys
sys.path.insert(0, str(ROOT))

from feature_scaling import (
    FeatureScaler, FeatureScalingError, feature_order_hash, load_feature_rules,
)
from train_lstm import LSTMModel as TrainModel, PSGDataset, collate_fn
import team_code


def unit_scaler() -> FeatureScaler:
    rules = load_feature_rules()
    params = []
    for rule in rules:
        center = 0.0
        scale = 1.0
        if rule["branch"] == "x_static" and rule["index"] == 9:
            center, scale = 25.0, 5.0
        params.append({
            "branch": rule["branch"], "index": rule["index"], "name": rule["name"],
            "center": center, "scale": scale, "scale_method": "test",
            "fit_valid_observations": 1, "fit_contributing_records": 1,
            "is_constant": False, "p05": None, "p25": None,
            "p50": center, "p75": None, "p95": None,
        })
    return FeatureScaler("typed_v1", rules, params, {
        "rules_sha256": __import__("feature_scaling").stable_hash(rules),
    })


def arrays(length=12):
    seq = np.zeros((length, 483), dtype=np.float32)
    ecg = np.ones((max(0, length - 10), 12), dtype=np.float32)
    static = np.ones(196, dtype=np.float32)
    static[0] = 60
    static[9] = 25
    static[10:] = 1
    mask = np.ones(length, dtype=bool)
    return seq, ecg, static, mask


class FeatureScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_feature_rules()
        cls.scaler = unit_scaler()

    def test_01_exact_691_mapping(self):
        self.assertEqual(len(self.rules), 691)
        self.assertEqual(sum(r["branch"] == "X_seq" for r in self.rules), 483)
        self.assertEqual(sum(r["branch"] == "X_ecg" for r in self.rules), 12)
        self.assertEqual(sum(r["branch"] == "x_static" for r in self.rules), 196)
        broken = [dict(r) for r in self.rules]
        broken[-1]["index"] = 194
        with self.assertRaises(FeatureScalingError):
            FeatureScaler("typed_v1", broken, self.scaler.parameters)
        self.assertEqual(483 + 12, 495)

    def test_02_age_order_and_raw_copy(self):
        outputs = []
        raw = []
        for age in (60.0, 70.0, 80.0):
            seq, ecg, static, _ = arrays()
            static[0] = age
            raw.append(float(static[0]))
            outputs.append(float(self.scaler.transform_arrays(seq, ecg, static)[2][0]))
        self.assertEqual(raw, [60.0, 70.0, 80.0])
        self.assertTrue(outputs[0] < outputs[1] < outputs[2])

    def test_03_fixed_units_and_bounded_features(self):
        seq, ecg, static, _ = arrays()
        seq[:, 414:417] = np.array([0.0, 50.0, 100.0])
        ecg[:, 0] = 1000.0
        ecg[:, 4] = 50.0
        static[18] = 0.5
        t_seq, t_ecg, t_static = self.scaler.transform_arrays(seq, ecg, static)
        np.testing.assert_allclose(t_seq[0, 414:417], [0.0, 0.5, 1.0])
        self.assertAlmostEqual(float(t_ecg[0, 0]), 1.0)
        self.assertAlmostEqual(float(t_ecg[0, 4]), 0.5)
        self.assertAlmostEqual(float(t_static[18]), 0.5)

    def test_04_legacy_clip_and_new_information(self):
        seq, ecg, static, _ = arrays()
        ecg = np.tile(ecg[:1], (3, 1))
        ecg[:, 0] = [800.0, 1000.0, 1200.0]
        typed = self.scaler.transform_arrays(seq, ecg, static)[1][:, 0]
        legacy = FeatureScaler.legacy_clip().transform_arrays(seq, ecg, static)[1][:, 0]
        np.testing.assert_allclose(typed, [0.8, 1.0, 1.2])
        np.testing.assert_allclose(legacy, [50.0, 50.0, 50.0])

    def test_05_existing_log_and_unbounded_ratio(self):
        seq, ecg, static, _ = arrays()
        seq[0, 0], seq[1, 0] = -2.0, 2.0
        ratio_index = 54 + 15
        seq[:3, ratio_index] = [2.0, 100.0, 1e6]
        out = self.scaler.transform_arrays(seq, ecg, static)[0]
        self.assertLess(out[0, 0], out[1, 0])
        np.testing.assert_allclose(out[:2, 0], [-2.0, 2.0])
        self.assertTrue(np.all(np.diff(out[:3, ratio_index]) > 0))
        self.assertGreater(float(out[2, ratio_index]), 1.0)

    def test_06_emg_extreme_is_finite(self):
        seq, ecg, static, _ = arrays()
        seq[:3, 433] = [0.3, 0.6, 1e15]
        out = self.scaler.transform_arrays(seq, ecg, static)[0][:3, 433]
        self.assertTrue(np.isfinite(out).all())
        self.assertNotEqual(float(out[0]), float(out[1]))
        self.assertLess(out[0], out[1])
        self.assertLess(out[1], out[2])

    def test_07_bmi_missing_and_legal_zero(self):
        seq, ecg, static, _ = arrays()
        static[9] = 0.0
        seq[:, 414] = 0.0
        seq[:, 475] = 0.0
        t_seq, _, t_static = self.scaler.transform_arrays(seq, ecg, static)
        self.assertEqual(float(t_static[9]), 0.0)
        self.assertTrue(np.all(t_seq[:, 414] == 0))
        self.assertTrue(np.all(t_seq[:, 475] == 0))
        static[9] = np.nan
        self.assertEqual(float(self.scaler.transform_arrays(seq, ecg, static)[2][9]), 0.0)
        static[9] = 30
        self.assertEqual(float(self.scaler.transform_arrays(seq, ecg, static)[2][9]), 1.0)

    def test_08_sentinels_fallback_and_empty_ecg(self):
        seq, _, static, _ = arrays()
        ecg = np.zeros((2, 12), dtype=np.float32)
        ecg[:, 11] = 0.75
        t_seq, t_ecg, _ = self.scaler.transform_arrays(seq, ecg, static)
        self.assertTrue(np.all(t_ecg[:, :11] == 0))
        np.testing.assert_allclose(t_ecg[:, 11], 0.75)
        fallback = np.zeros((1, 483), dtype=np.float32)
        empty = np.zeros((0, 12), dtype=np.float32)
        out = self.scaler.transform_arrays(fallback, empty, static, fallback_sequence=True)
        self.assertTrue(np.all(out[0] == 0))
        self.assertEqual(out[1].shape, (0, 12))
        static[10:] = 0
        out_static = self.scaler.transform_arrays(seq, empty, static)[2]
        self.assertTrue(np.all(out_static[10:] == 0))

    def test_09_state_round_trip_and_no_inplace(self):
        restored = FeatureScaler.from_state_dict(self.scaler.state_dict())
        seq, ecg, static, _ = arrays()
        originals = (seq.copy(), ecg.copy(), static.copy())
        first = self.scaler.transform_arrays(seq, ecg, static)
        second = restored.transform_arrays(seq, ecg, static)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        for actual, original in zip((seq, ecg, static), originals):
            np.testing.assert_array_equal(actual, original)
        with self.assertRaises(FeatureScalingError):
            FeatureScaler.from_state_dict({})

    def test_10_training_inference_path_and_alignment(self):
        scaler = self.scaler
        seq, ecg, static, mask = arrays(length=14)
        seq[:, 0] = np.linspace(-1, 1, len(seq))
        ecg[:, 0] = [800, 900, 1000, 1100]
        record = {"BidsFolder": "TEST001", "SessionID": "01", "SiteID": "S"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "TEST001_ses-01.npz"
            np.savez(path, X_seq=seq, X_ecg=ecg, x_static=static, y=np.array(1), mask=mask)
            dataset = PSGDataset([record], tmp, None, scaler)
            item = dataset[0]
            sample = team_code._sample(record, None, [Path(tmp)], scaler)
            np.testing.assert_allclose(item["X_seq"].numpy(), sample["X_seq"])
            np.testing.assert_allclose(item["X_ecg"].numpy(), sample["X_ecg"])
            np.testing.assert_allclose(item["x_static"].numpy(), sample["x_static"])
            batch = collate_fn([item])
            X_seq, X_ecg, _, x_static, _, _, lengths = batch
            self.assertTrue(torch.all(X_ecg[0, :10] == 0))
            self.assertTrue(torch.all(X_ecg[0, 10:14] != 0))
            train_model = TrainModel().cpu().eval()
            infer_model = team_code.LSTMModel().cpu().eval()
            infer_model.load_state_dict(train_model.state_dict())
            with torch.no_grad():
                train_logit = train_model(X_seq, X_ecg, torch.ones_like(X_ecg[..., 0], dtype=torch.bool), x_static, lengths)
            old_device = team_code.DEVICE
            team_code.DEVICE = torch.device("cpu")
            try:
                infer_logit = team_code._infer_one(infer_model, sample)
            finally:
                team_code.DEVICE = old_device
            self.assertTrue(np.allclose(train_logit.item(), infer_logit, rtol=1e-6, atol=1e-6))

    def test_11_transform_does_not_refit_and_backward_finite(self):
        before = json.dumps(self.scaler.state_dict(), sort_keys=True)
        seq, ecg, static, mask = arrays()
        seq[:, 69] = 1e12
        t_seq, t_ecg, t_static = self.scaler.transform_arrays(seq, ecg, static)
        after = json.dumps(self.scaler.state_dict(), sort_keys=True)
        self.assertEqual(before, after)
        item = {
            "X_seq": torch.from_numpy(t_seq), "X_ecg": torch.from_numpy(t_ecg),
            "x_static": torch.from_numpy(t_static), "age": torch.tensor(60.0),
            "y": torch.tensor([1.0]), "mask": torch.from_numpy(mask), "length": len(seq),
        }
        batch = collate_fn([item])
        X_seq, X_ecg, mask_ecg, x_static, _, y, lengths = batch
        model = TrainModel().cpu().train()
        logits = model(X_seq, X_ecg, mask_ecg, x_static, lengths)
        loss = torch.nn.BCEWithLogitsLoss()(logits, y.squeeze(-1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits).all())
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main(verbosity=2)
