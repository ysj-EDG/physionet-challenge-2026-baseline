#!/usr/bin/env python3
"""Synthetic contract tests for the frozen SIREN Branch A G1-B feature spec."""

from __future__ import annotations

import csv
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "G1B_FEATURE_MANIFEST_FROZEN.csv"
SPEC = ROOT / "docs" / "G1B_FEATURE_SPEC_FROZEN.md"
DRAFT = ROOT / "docs" / "G1A_FEATURE_MANIFEST_DRAFT.csv"

MODEL_STAGES = {
    "Wake": {5},
    "REM": {4},
    "N2": {2},
    "N3": {1},
    "NREM": {1, 2, 3},
    "All-stage": {1, 2, 3, 4, 5},
}
MINIMUM = 5


def demo5_from_static(x_static: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation of the frozen five-value demographic contract."""
    values = np.asarray(x_static, dtype=float)
    if values.shape != (196,):
        raise ValueError("x_static must have shape (196,)")
    sex = values[1:4]
    race = values[4:9]
    if not np.isin(sex, (0.0, 1.0)).all() or sex.sum() != 1.0:
        raise ValueError("illegal sex one-hot")
    if not np.isin(race, (0.0, 1.0)).all() or race.sum() != 1.0:
        raise ValueError("illegal race one-hot")

    result = np.full(5, np.nan, dtype=float)
    valid = np.zeros(5, dtype=bool)
    if sex[2] == 0.0:
        result[0] = sex[1]
        valid[0] = True
    if np.isfinite(values[9]) and values[9] != 0.0:
        result[1] = values[9]
        valid[1] = True
    if race[3] == 0.0:
        result[2:] = (race[4], race[1], race[0])
        valid[2:] = True
    return result, valid


def pool_mean_iqr(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(values, dtype=float)[np.asarray(valid, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if len(selected) < MINIMUM:
        return np.array([np.nan, np.nan]), np.array([False, False])
    q25, q75 = np.quantile(selected, [0.25, 0.75], method="linear")
    return np.array([selected.mean(), q75 - q25]), np.array([True, True])


def stage_mask(codes: np.ndarray, stage_valid: np.ndarray, stage: str) -> np.ndarray:
    codes = np.asarray(codes, dtype=float)
    return np.asarray(stage_valid, dtype=bool) & np.isfinite(codes) & np.isin(
        codes, list(MODEL_STAGES[stage])
    )


def relative_power(band: np.ndarray, beta: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return np.asarray(band) + np.asarray(beta) - np.asarray(p0)


class G1BContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.spec = SPEC.read_text(encoding="utf-8")

    def test_demo5_order_missing_and_repeat_contract(self):
        base = np.zeros(196)
        base[2] = 1  # Male.
        base[9] = 24.5
        base[8] = 1  # White.
        values, valid = demo5_from_static(base)
        np.testing.assert_array_equal(values, [1, 24.5, 1, 0, 0])
        self.assertTrue(valid.all())

        demo = [r for r in self.rows if r["family"] == "Demo5"]
        self.assertEqual(len(demo), 25)
        by_view = {}
        for row in demo:
            by_view.setdefault(row["view_id"], []).append(row)
        expected = [
            "demo_male", "demo_bmi", "demo_race_white",
            "demo_race_black", "demo_race_asian",
        ]
        for rows in by_view.values():
            self.assertEqual([r["unique_feature_id"] for r in rows], expected)
        for feature_id in expected:
            signatures = {
                (r["formula"], r["validity_expression"], r["missing_semantics"])
                for r in demo if r["unique_feature_id"] == feature_id
            }
            self.assertEqual(len(signatures), 1)

    def test_demo5_others_unavailable_bmi_unknown_and_illegal_one_hot(self):
        others = np.zeros(196); others[1] = 1; others[6] = 1; others[9] = 0
        values, valid = demo5_from_static(others)
        self.assertEqual(values[0], 0)
        self.assertFalse(valid[1]); self.assertTrue(np.isnan(values[1]))
        np.testing.assert_array_equal(values[2:], [0, 0, 0])
        self.assertTrue(valid[2:].all())

        unavailable = np.zeros(196); unavailable[3] = 1; unavailable[7] = 1
        values, valid = demo5_from_static(unavailable)
        self.assertFalse(valid[0]); self.assertTrue(np.isnan(values[0]))
        self.assertFalse(valid[2:].any()); self.assertTrue(np.isnan(values[2:]).all())

        illegal = unavailable.copy(); illegal[8] = 1
        with self.assertRaises(ValueError):
            demo5_from_static(illegal)

    def test_age_is_not_a_model_feature(self):
        demo = [r for r in self.rows if r["family"] == "Demo5"]
        self.assertFalse(any(r["source_index"] == "0" for r in demo))
        self.assertNotIn("demo_age", {r["unique_feature_id"] for r in self.rows})

    def test_six_model_stages_and_nrem_definition(self):
        self.assertEqual(tuple(MODEL_STAGES), ("Wake", "REM", "N2", "N3", "NREM", "All-stage"))
        codes = np.array([5, 4, 3, 2, 1, 0, 9, np.nan])
        valid = np.ones(8, dtype=bool)
        self.assertEqual(stage_mask(codes, valid, "NREM").sum(), 3)
        self.assertEqual(stage_mask(codes, valid, "All-stage").sum(), 5)
        eeg = [r for r in self.rows if r["family"] == "EEG54_stage_spectral"]
        self.assertEqual({r["stage_group"] for r in eeg}, set(MODEL_STAGES))
        self.assertNotIn("N1", {r["stage_group"] for r in eeg})

    def test_five_observation_boundary_and_linear_iqr(self):
        values, mask = pool_mean_iqr(np.arange(4.0), np.ones(4, dtype=bool))
        self.assertFalse(mask.any()); self.assertTrue(np.isnan(values).all())
        values, mask = pool_mean_iqr(np.arange(5.0), np.ones(5, dtype=bool))
        np.testing.assert_allclose(values, [2.0, 2.0])
        self.assertTrue(mask.all())
        expected = np.quantile(np.array([0., 1., 4., 8., 10., 13.]), [0.25, 0.75], method="linear")
        values, _ = pool_mean_iqr(np.array([0., 1., 4., 8., 10., 13.]), np.ones(6, dtype=bool))
        self.assertAlmostEqual(values[1], expected[1] - expected[0])

    def test_spatial_formula_and_same_epoch_intersection(self):
        band_l = np.array([2., 2., 2., 2., 2.]); beta_l = np.ones(5); p0_l = np.zeros(5)
        band_r = np.ones(5); beta_r = np.ones(5); p0_r = np.zeros(5)
        diff = relative_power(band_l, beta_l, p0_l) - relative_power(band_r, beta_r, p0_r)
        np.testing.assert_array_equal(diff, np.ones(5))
        left = np.array([1, 1, 1, 1, 1, 0], dtype=bool)
        right = np.array([0, 1, 1, 1, 1, 1], dtype=bool)
        self.assertEqual((left & right).sum(), 4)
        self.assertEqual(min(left.sum(), right.sum()), 5)
        spatial = [r for r in self.rows if r["family"].startswith("spatial_")]
        self.assertEqual(len(spatial), 36)
        with DRAFT.open(newline="", encoding="utf-8") as handle:
            draft_spatial = [
                r for r in csv.DictReader(handle)
                if r["family"].startswith("spatial_")
            ]
        self.assertEqual(
            [(r["unique_feature_id"], r["source_indices"], r["formula"]) for r in spatial],
            [(r["unique_feature_id"], r["source_indices"], r["formula"]) for r in draft_spatial],
        )
        for row in spatial:
            n_indices = len(row["source_indices"].split(";"))
            if row["family"] == "spatial_asymmetry_candidate":
                self.assertEqual(n_indices, 6)
            else:
                self.assertEqual(n_indices, 12)
                self.assertNotIn("C3", row["source_indices"])
                self.assertNotIn("C4", row["source_indices"])

    def test_modality_specific_validity_and_lawful_zero(self):
        hrv_success = np.array([1, 1, 0], dtype=bool)
        hrv_feature_valid = np.array([[1], [0], [1]], dtype=bool)
        self.assertEqual((hrv_success[:, None] & hrv_feature_valid).ravel().tolist(), [True, False, False])
        self.assertTrue(bool((np.array([0.0]) == 0)[0]))  # A valid value may be zero.
        emg_success = np.array([[1, 0, 1]], dtype=bool)
        self.assertEqual(emg_success[:, 1].tolist(), [False])
        resp_valid = np.array([[True, False]])
        self.assertEqual(resp_valid.tolist(), [[True, False]])

        hrv = [r for r in self.rows if r["family"] == "HRV11"]
        emg = [r for r in self.rows if r["family"] == "EMG24"]
        resp = [r for r in self.rows if r["family"] == "Resp14"]
        self.assertTrue(all("hrv_success" in r["validity_expression"] and "hrv_feature_valid" in r["validity_expression"] for r in hrv))
        self.assertTrue(all("emg_epoch_success" in r["validity_expression"] and "emg_channel_available" in r["validity_expression"] for r in emg))
        self.assertTrue(all("resp_feature_valid" in r["validity_expression"] for r in resp))
        self.assertTrue(all(r["min_valid_observations"] == "5 windows" for r in hrv))
        self.assertTrue(all(r["min_valid_observations"] == "5 epochs" for r in emg + resp))

    def test_manifest_dimensions_positions_required_fields_and_exclusions(self):
        self.assertEqual(len(self.rows), 827)
        self.assertEqual(len({r["unique_feature_id"] for r in self.rows}), 807)
        self.assertEqual(Counter(r["view_id"] for r in self.rows), {"V1": 113, "V2": 129, "V3": 349, "V4": 113, "V5": 123})
        required = {
            "source_npz_field", "source_index", "source_indices", "formula", "unit",
            "aggregation", "stage_group", "validity_expression", "min_valid_observations",
            "missing_semantics", "output_missing_value", "specification_status",
            "source_code_reference",
        }
        self.assertTrue(required.issubset(self.rows[0]))
        for view in ("V1", "V2", "V3", "V4", "V5"):
            rows = [r for r in self.rows if r["view_id"] == view]
            self.assertEqual([int(r["view_position"]) for r in rows], list(range(len(rows))))
        self.assertTrue(all(r["specification_status"] == "FROZEN_G1B" for r in self.rows))
        self.assertTrue(all(r["output_missing_value"] == "raw NaN; valid_mask=False" for r in self.rows))
        text = " ".join(r["feature_name"].lower() for r in self.rows)
        for excluded in ("coherence", "bsr", "circadian"):
            self.assertNotIn(excluded, text)
        self.assertFalse(any(r["source_npz_field"] == "X_ecg" and r["source_index"] == "11" for r in self.rows))

    def test_caisr_validity_does_not_treat_individual_zero_as_missing(self):
        caisr = [r for r in self.rows if r["family"] == "CAISR20"]
        self.assertEqual(len(caisr), 20)
        self.assertEqual(
            [int(r["source_index"]) for r in caisr],
            [11, 12, 13, 14, 16, 17, 18, 19, 20, 32, 34, 87,
             96, 97, 98, 126, 127, 141, 168, 173],
        )
        self.assertTrue(all("individual zero is valid" in r["missing_semantics"] for r in caisr))
        self.assertTrue(all("all-zero x_static[10:196] sentinel" in r["missing_semantics"] for r in caisr))

    def test_no_record_exclusion_or_patient_data_in_contract(self):
        lower = self.spec.lower()
        self.assertIn("retain all 6,600 records", lower)
        self.assertIn("no record-level dropna", lower)
        self.assertIn("night-level feature cache was not generated", lower)
        prohibited = ("bdsp_patientid", "bidsfolder", "sessionid", "diagnosis")
        manifest_text = MANIFEST.read_text(encoding="utf-8").lower()
        self.assertFalse(any(term in manifest_text for term in prohibited))


if __name__ == "__main__":
    unittest.main(verbosity=2)
