from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from extensions.eeg_tail import (
    BANDS,
    EEG_TAIL_SPEC,
    LEADS,
    STAGE_SCOPES,
    extract_family,
)
from extensions.recipe import build_recipe_identity
from extensions.sources import load_source_npz
from extensions.tests.fixtures import source_payload, write_source_npz


class EegTailTests(unittest.TestCase):
    def _source(self, root: Path, payload):
        source_root = root / "source"
        return load_source_npz(
            write_source_npz(source_root, payload=payload), source_root
        )

    def test_dimensions_order_and_exact_source_offsets(self) -> None:
        definitions = EEG_TAIL_SPEC.feature_definitions
        self.assertEqual(len(definitions), 288)
        self.assertEqual(len({item.feature_id for item in definitions}), 288)
        self.assertEqual(
            [item.source_columns for item in definitions[:12]],
            [(4, 7, 6), (4, 7, 6), (4, 7, 6), (4, 7, 6),
             (4, 7, 6), (4, 7, 6), (4, 7, 6), (4, 7, 6),
             (4, 7, 6), (4, 7, 6), (4, 7, 6), (4, 7, 6)],
        )
        self.assertEqual((len(LEADS), len(BANDS), len(STAGE_SCOPES)), (6, 4, 6))

    def test_recipe_builds_and_parameter_change_invalidates_only_this_spec(self) -> None:
        code_root = Path(__file__).resolve().parents[2]
        baseline = build_recipe_identity(EEG_TAIL_SPEC, {}, code_root)
        changed = build_recipe_identity(
            EEG_TAIL_SPEC, {"minimum_support": 21}, code_root
        )
        self.assertNotEqual(baseline.recipe_id, changed.recipe_id)

    def test_linear_quantiles_support_and_validity_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=20)
            payload["stage_code_aligned"][:] = 5.0
            payload["X_seq"][:] = 0.0
            values = np.arange(20, dtype=np.float32)
            payload["X_seq"][:, 4] = values
            payload["X_seq"][:, 7] = 10.0
            payload["X_seq"][:, 6] = 3.0
            payload["eeg_clean_subsegment_count"][-1, 1] = 0
            payload["eeg_channel_available"][2] = False
            source = self._source(Path(directory), payload)
            result = extract_family(source, EEG_TAIL_SPEC.default_config)

            expected = values + 7.0
            self.assertAlmostEqual(
                result.values[0], np.quantile(expected, 0.88, method="linear"), places=5
            )
            self.assertAlmostEqual(
                result.values[1], np.quantile(expected, 0.95, method="linear"), places=5
            )
            self.assertEqual(result.support_count[0], 20)
            self.assertTrue(result.valid_mask[0])

            f4_delta_wake = 48
            self.assertFalse(result.valid_mask[f4_delta_wake])
            self.assertTrue(np.isnan(result.values[f4_delta_wake]))
            self.assertEqual(result.support_count[f4_delta_wake], 19)
            self.assertEqual(result.missing_reason[f4_delta_wake], "INSUFFICIENT_SUPPORT")

            c3_delta_wake = 96
            self.assertFalse(result.valid_mask[c3_delta_wake])
            self.assertEqual(result.support_count[c3_delta_wake], 0)
            self.assertEqual(result.missing_reason[c3_delta_wake], "CHANNEL_UNAVAILABLE")

            rem_index = 2
            self.assertFalse(result.valid_mask[rem_index])
            self.assertEqual(result.support_count[rem_index], 0)
            self.assertEqual(result.missing_reason[rem_index], "INSUFFICIENT_SUPPORT")

    def test_valid_zero_is_retained_and_invalid_fill_zero_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=20)
            payload["stage_code_aligned"][:] = 5.0
            payload["X_seq"][:] = 0.0
            payload["eeg_clean_subsegment_count"][:, 0] = 0
            payload["eeg_clean_subsegment_count"][:19, 0] = 1
            source = self._source(Path(directory), payload)
            result = extract_family(source, EEG_TAIL_SPEC.default_config)
            self.assertEqual(result.support_count[0], 19)
            self.assertFalse(result.valid_mask[0])

            f4_delta_wake = 48
            self.assertTrue(result.valid_mask[f4_delta_wake])
            self.assertEqual(result.values[f4_delta_wake], 0.0)
            self.assertEqual(result.values[f4_delta_wake + 1], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
