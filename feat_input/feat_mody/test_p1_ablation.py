#!/usr/bin/env python
"""Lightweight regression tests for the four P1 input ablations."""

from __future__ import annotations

import copy
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(ROOT))

from feature_scaling import FeatureScaler
from train_lstm import PSGDataset, collate_fn

B_CHECKPOINT = ROOT / "output" / "p0_ab" / "seed7" / "B_typed_v1" / "lstm_model.pt"


def raw_arrays(length=14):
    seq = np.zeros((length, 483), dtype=np.float32)
    ecg = np.ones((length - 10, 12), dtype=np.float32)
    static = np.ones(196, dtype=np.float32)
    static[0] = 70.0
    static[9] = 25.0
    static[10:] = 1.0
    mask = np.ones(length, dtype=bool)
    return seq, ecg, static, mask


class P1AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = torch.load(B_CHECKPOINT, map_location="cpu", weights_only=False)
        cls.b_state = checkpoint["input_preprocessing"]
        cls.base = FeatureScaler.from_state_dict(cls.b_state)

    def test_default_missing_mask_field_matches_b_exactly(self):
        old_state = copy.deepcopy(self.b_state)
        old_state.pop("mask_config", None)
        old_style = FeatureScaler.from_state_dict(old_state)
        arrays = raw_arrays()[:3]
        for expected, actual in zip(self.base.transform_arrays(*arrays), old_style.transform_arrays(*arrays)):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(old_style.mask_config, {"x_static_zero_indices": [], "x_seq_zero_ranges": []})

    def test_clip5_changes_only_robust_transformed_columns(self):
        seq, ecg, static, _ = raw_arrays()
        rule = next(r for r in self.base.rules if r["branch"] == "X_seq" and r["scaling"] == "robust" and r["nonlinear"] == "identity")
        parameter = next(p for p in self.base.parameters if p["branch"] == "X_seq" and p["index"] == rule["index"] )
        seq[:, rule["index"]] = (parameter["center"] + 10.0 * parameter["scale"]) * rule["unit_scale"]
        base = self.base.transform_arrays(seq, ecg, static)
        clipped_scaler = self.base.with_runtime_config(clip_z=5.0)
        clipped = clipped_scaler.transform_arrays(seq, ecg, static)
        for branch_index, branch in enumerate(("X_seq", "X_ecg", "x_static")):
            expected = base[branch_index].copy()
            robust = [r["index"] for r in self.base.rules if r["branch"] == branch and r["scaling"] == "robust"]
            expected[..., robust] = np.clip(expected[..., robust], -5.0, 5.0)
            np.testing.assert_array_equal(clipped[branch_index], expected)
        self.assertEqual(float(clipped[0][0, rule["index"]]), 5.0)
        restored = FeatureScaler.from_state_dict(clipped_scaler.state_dict())
        self.assertEqual(restored.clip_z, 5.0)

    def test_each_mask_changes_only_its_declared_columns(self):
        arrays = raw_arrays()[:3]
        base = self.base.transform_arrays(*arrays)
        configs = [
            ({"x_static_zero_indices": [0]}, "static_age"),
            ({"x_static_zero_indices": [9]}, "static_bmi"),
            ({"x_seq_zero_ranges": [[54, 414]]}, "coherence"),
        ]
        for config, name in configs:
            with self.subTest(name=name):
                scaler = self.base.with_runtime_config(clip_z=None, mask_config=config)
                out = scaler.transform_arrays(*arrays)
                np.testing.assert_array_equal(out[1], base[1])
                if name == "coherence":
                    np.testing.assert_array_equal(out[0][:, :54], base[0][:, :54])
                    self.assertTrue(np.all(out[0][:, 54:414] == 0))
                    np.testing.assert_array_equal(out[0][:, 414:], base[0][:, 414:])
                    np.testing.assert_array_equal(out[2], base[2])
                else:
                    index = 0 if name == "static_age" else 9
                    np.testing.assert_array_equal(out[0], base[0])
                    np.testing.assert_array_equal(out[2][:index], base[2][:index])
                    self.assertEqual(float(out[2][index]), 0.0)
                    np.testing.assert_array_equal(out[2][index + 1:], base[2][index + 1:])
                restored = FeatureScaler.from_state_dict(scaler.state_dict())
                for expected, actual in zip(out, restored.transform_arrays(*arrays)):
                    np.testing.assert_array_equal(actual, expected)

    def test_raw_age_padding_and_ecg_alignment_are_unchanged(self):
        seq, ecg, static, mask = raw_arrays()
        no_age = self.base.with_runtime_config(clip_z=None, mask_config={"x_static_zero_indices": [0]})
        base_ds = PSGDataset([], ".", None, self.base)
        no_age_ds = PSGDataset([], ".", None, no_age)
        base_item = base_ds._build_tensors(seq, ecg, static, 1, mask, "synthetic")
        no_age_item = no_age_ds._build_tensors(seq, ecg, static, 1, mask, "synthetic")
        self.assertEqual(float(no_age_item["age"]), 70.0)
        self.assertEqual(float(no_age_item["x_static"][0]), 0.0)
        base_batch = collate_fn([base_item])
        no_age_batch = collate_fn([no_age_item])
        for index in (0, 1, 2, 4, 5, 6):
            torch.testing.assert_close(no_age_batch[index], base_batch[index])
        self.assertTrue(torch.all(no_age_batch[1][0, :10] == 0))
        torch.testing.assert_close(no_age_batch[1][0, 10:], no_age_item["X_ecg"])

    def test_raw_age_padding_and_ecg_alignment_are_unchanged(self):
        seq, ecg, static, mask = raw_arrays()
        base_ds = PSGDataset([], ".", None, self.base)
        no_age = self.base.with_runtime_config(
            clip_z=None, mask_config={"x_static_zero_indices": [0]}
        )
        no_age_ds = PSGDataset([], ".", None, no_age)
        base_item = base_ds._build_tensors(seq, ecg, static, 1, mask, "synthetic")
        ablated_item = no_age_ds._build_tensors(seq, ecg, static, 1, mask, "synthetic")
        self.assertEqual(float(base_item["age"]), 70.0)
        self.assertEqual(float(ablated_item["age"]), 70.0)
        self.assertEqual(float(ablated_item["x_static"][0]), 0.0)
        base_batch = collate_fn([base_item])
        ablated_batch = collate_fn([ablated_item])
        for index in (0, 1, 2, 4, 5, 6):
            torch.testing.assert_close(ablated_batch[index], base_batch[index])
        torch.testing.assert_close(ablated_batch[3][:, 1:], base_batch[3][:, 1:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
