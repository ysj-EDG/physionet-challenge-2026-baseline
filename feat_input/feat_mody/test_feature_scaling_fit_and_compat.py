#!/usr/bin/env python
"""Fit-boundary and checkpoint compatibility tests for typed_v1."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from feature_scaling import FeatureScaler, FeatureScalingError
import team_code


def write_training_cache(root: Path, record_id: int, sparse_value: float, tail_value: float, bmi: float):
    rid = f"R{record_id:02d}"
    seq = np.zeros((12, 483), dtype=np.float32)
    seq[:, 0] = sparse_value
    ecg = np.ones((5, 12), dtype=np.float32)
    ecg[:2, 0] = [800.0 + record_id, 900.0 + record_id]
    ecg[2:, 0] = tail_value
    static = np.ones(196, dtype=np.float32)
    static[0] = 60 + record_id
    static[9] = bmi
    static[10:] = 1
    np.savez(
        root / f"{rid}_ses-01.npz", X_seq=seq, X_ecg=ecg,
        x_static=static, y=np.array(0), mask=np.ones(12, dtype=bool),
    )
    return {"BidsFolder": rid, "SessionID": "01", "SiteID": "S"}


def parameter(scaler, branch, index):
    return next(p for p in scaler.parameters if p["branch"] == branch and p["index"] == index)


class FitAndCompatibilityTests(unittest.TestCase):
    def _fit(self, root: Path, tail: float):
        bmi_values = [0.0, np.nan, 20.0, 30.0, 25.0]
        records = [
            write_training_cache(root, i, 1.0 if i == 4 else 0.0, tail, bmi_values[i])
            for i in range(5)
        ]
        manifest = root / "train_records.json"
        manifest.write_text(json.dumps(records))
        return FeatureScaler.fit_from_cache(
            records, root, sample_cap=128, seed=7, manifest_path=manifest,
            code_head="test",
        )

    def test_12_iqr_zero_sparse_uses_safe_fallback_and_bmi_excludes_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            scaler = self._fit(Path(tmp), 1e9)
            sparse = parameter(scaler, "X_seq", 0)
            self.assertIn(sparse["scale_method"], {"p95_p05", "std"})
            self.assertGreater(sparse["scale"], 1e-6)
            bmi = parameter(scaler, "x_static", 9)
            self.assertEqual(bmi["fit_contributing_records"], 3)
            self.assertEqual(bmi["fit_valid_observations"], 3)

    def test_13_ecg_tail_not_consumed_by_fit(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            scaler_a = self._fit(Path(one), 1e6)
            scaler_b = self._fit(Path(two), 1e15)
            pa = parameter(scaler_a, "X_ecg", 0)
            pb = parameter(scaler_b, "X_ecg", 0)
            self.assertEqual(pa["center"], pb["center"])
            self.assertEqual(pa["scale"], pb["scale"])
            self.assertEqual(scaler_a.metadata["raw_ecg_windows_fit"], 25)
            self.assertEqual(scaler_a.metadata["consumed_ecg_windows_fit"], 10)

    def test_14_nontraining_extremes_cannot_change_frozen_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            scaler = self._fit(Path(tmp), 1e9)
            before = json.dumps(scaler.state_dict(), sort_keys=True)
            seq = np.full((12, 483), 1e12, dtype=np.float32)
            ecg = np.full((2, 12), 1e12, dtype=np.float32)
            static = np.full(196, 1e12, dtype=np.float32)
            static[0], static[9] = 80, 35
            *_arrays, diagnostics = scaler.transform_arrays(
                seq, ecg, static, record_id="extreme_nontraining",
                return_diagnostics=True,
            )
            self.assertGreater(sum(x["invalid"] for x in diagnostics["per_branch"].values()), 0)
            self.assertTrue(all(np.isfinite(x).all() for x in _arrays))
            after = json.dumps(scaler.state_dict(), sort_keys=True)
            self.assertEqual(before, after)

    def test_15_old_checkpoint_requires_explicit_legacy_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            model = team_code.LSTMModel()
            torch.save({"state_dict": model.state_dict()}, folder / "lstm_model.pt")
            previous = os.environ.pop("LSTM_ALLOW_LEGACY_INPUT", None)
            try:
                with self.assertRaises(RuntimeError):
                    team_code.load_model(folder, False)
                os.environ["LSTM_ALLOW_LEGACY_INPUT"] = "1"
                loaded = team_code.load_model(folder, False)
                self.assertEqual(loaded["preprocessor"].mode, "legacy_clip")
            finally:
                if previous is None:
                    os.environ.pop("LSTM_ALLOW_LEGACY_INPUT", None)
                else:
                    os.environ["LSTM_ALLOW_LEGACY_INPUT"] = previous


if __name__ == "__main__":
    unittest.main(verbosity=2)
