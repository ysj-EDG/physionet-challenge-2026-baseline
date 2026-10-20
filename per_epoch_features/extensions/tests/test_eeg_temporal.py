from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from extensions.eeg_temporal import EEG_TEMPORAL_SPEC, extract_family
from extensions.recipe import build_recipe_identity
from extensions.sources import load_source_npz
from extensions.tests.fixtures import source_payload, write_source_npz


class EegTemporalTests(unittest.TestCase):
    def _result(self, root: Path, payload):
        source_root = root / "source"
        source = load_source_npz(
            write_source_npz(source_root, payload=payload), source_root
        )
        return extract_family(source, EEG_TEMPORAL_SPEC.default_config)

    @staticmethod
    def _set_delta(payload, values):
        payload["X_seq"][:] = 0.0
        payload["X_seq"][:, 4] = np.asarray(values, dtype=np.float32)

    def test_dimensions_order_and_recipe(self) -> None:
        definitions = EEG_TEMPORAL_SPEC.feature_definitions
        self.assertEqual(len(definitions), 96)
        self.assertEqual(len({item.feature_id for item in definitions}), 96)
        self.assertEqual(
            [item.formula_id for item in definitions[:4]],
            ["range80_v1", "raw_thirds_drift_v1", "clock_hour_tail_v1", "tail_entry_rate_v1"],
        )
        self.assertEqual(
            [item.unit for item in definitions[:4]],
            ["log10_power", "log10_power", "fraction", "per_hour"],
        )
        root = Path(__file__).resolve().parents[2]
        baseline = build_recipe_identity(EEG_TEMPORAL_SPEC, {}, root)
        changed = build_recipe_identity(
            EEG_TEMPORAL_SPEC, {"minimum_reference_support": 21}, root
        )
        self.assertNotEqual(baseline.recipe_id, changed.recipe_id)

    def test_drift_uses_raw_thirds_without_compressing_wake_or_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=120)
            payload["stage_code_aligned"][:] = 2.0
            payload["stage_code_aligned"][20:40] = 5.0
            payload["stage_valid"][60:80] = False
            values = np.concatenate((np.zeros(40), np.full(40, 100.0), np.full(40, 10.0)))
            self._set_delta(payload, values)
            result = self._result(Path(directory), payload)
            self.assertTrue(result.valid_mask[1])
            self.assertEqual(result.values[1], 10.0)
            self.assertEqual(result.support_count[1], 20)

        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=120)
            payload["stage_code_aligned"][:] = 2.0
            payload["stage_valid"][:21] = False
            self._set_delta(payload, np.arange(120, dtype=float))
            result = self._result(Path(directory), payload)
            self.assertFalse(result.valid_mask[1])
            self.assertEqual(result.support_count[1], 19)
            self.assertEqual(result.missing_reason[1], "INSUFFICIENT_DRIFT_SIDE_SUPPORT")

    def test_real_hour_requires_120_contiguous_positions_and_96_sleep_epochs(self) -> None:
        cases = (
            (119, 119, False, False),
            (120, 95, False, False),
            (120, 96, False, True),
            (120, 96, True, False),
        )
        for epochs, sleep_count, broken, expected_valid in cases:
            with self.subTest(epochs=epochs, sleep=sleep_count, broken=broken):
                with tempfile.TemporaryDirectory() as directory:
                    payload = source_payload(epochs=epochs)
                    payload["stage_code_aligned"][:] = 5.0
                    payload["stage_code_aligned"][:sleep_count] = 2.0
                    self._set_delta(payload, np.arange(epochs, dtype=float))
                    if broken:
                        payload["epoch_start_sec"][60:] += 1.0
                    result = self._result(Path(directory), payload)
                    self.assertEqual(bool(result.valid_mask[2]), expected_valid)
                    if expected_valid:
                        threshold = np.quantile(
                            np.arange(sleep_count, dtype=float), 0.88, method="linear"
                        )
                        expected = np.mean(np.arange(sleep_count) > threshold)
                        self.assertAlmostEqual(result.values[2], expected)
                        self.assertEqual(result.support_count[2], sleep_count)
                    else:
                        self.assertTrue(np.isnan(result.values[2]))
                        self.assertEqual(result.missing_reason[2], "NO_QUALIFYING_HOUR")

    def test_duration_break_rejects_hour_and_entries_use_only_legal_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=120)
            payload["stage_code_aligned"][:] = 2.0
            self._set_delta(payload, np.zeros(120))
            result = self._result(Path(directory), payload)
            self.assertTrue(result.valid_mask[2])
            self.assertEqual(result.values[2], 0.0)
            self.assertTrue(result.valid_mask[3])
            self.assertEqual(result.values[3], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=120)
            payload["stage_code_aligned"][:] = 2.0
            self._set_delta(payload, np.zeros(120))
            payload["epoch_duration_sec"][50] = 29.0
            result = self._result(Path(directory), payload)
            self.assertFalse(result.valid_mask[2])
            self.assertTrue(result.valid_mask[3])
            self.assertEqual(result.values[3], 0.0)
            self.assertEqual(result.support_count[3], 119)

        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=24)
            payload["stage_code_aligned"][:] = 2.0
            values = np.arange(24, dtype=float)
            self._set_delta(payload, values)
            payload["stage_code_aligned"][10] = 5.0
            payload["stage_valid"][15] = False
            payload["epoch_start_sec"][20:] += 30.0
            result = self._result(Path(directory), payload)
            valid_sleep = (
                payload["stage_valid"]
                & np.isin(payload["stage_code_aligned"], (1, 2, 3, 4))
            )
            threshold = np.quantile(values[valid_sleep], 0.88, method="linear")
            high = values > threshold
            legal = (
                valid_sleep[:-1]
                & valid_sleep[1:]
                & (np.diff(payload["epoch_start_sec"]) == 30.0)
            )
            entries = np.sum(legal & (~high[:-1]) & high[1:])
            expected_rate = entries / (np.sum(legal) * 30.0 / 3600.0)
            self.assertAlmostEqual(result.values[3], expected_rate)
            self.assertEqual(result.support_count[3], int(np.sum(legal)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
