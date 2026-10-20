from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from extensions.eeg_coherence_stage import EEG_COHERENCE_STAGE_SPEC, extract_family
from extensions.recipe import build_recipe_identity
from extensions.sources import load_source_npz
from extensions.tests.fixtures import source_payload, write_source_npz


class EegCoherenceStageTests(unittest.TestCase):
    def _result(self, root: Path, payload):
        source_root = root / "source"
        source = load_source_npz(write_source_npz(source_root, payload=payload), source_root)
        return extract_family(source, EEG_COHERENCE_STAGE_SPEC.default_config)

    def test_dimensions_pair_positions_band_columns_and_recipe(self) -> None:
        definitions = EEG_COHERENCE_STAGE_SPEC.feature_definitions
        self.assertEqual(len(definitions), 90)
        self.assertEqual(len({item.feature_id for item in definitions}), 90)
        self.assertEqual(definitions[0].source_columns, (54,))
        self.assertEqual(definitions[30].source_columns, (54 + 9 * 24,))
        self.assertEqual(definitions[60].source_columns, (54 + 14 * 24,))
        self.assertEqual(
            [definitions[index].source_columns for index in (0, 6, 12, 18, 24)],
            [(54,), (55,), (56,), (57,), (58,)],
        )
        root = Path(__file__).resolve().parents[2]
        baseline = build_recipe_identity(EEG_COHERENCE_STAGE_SPEC, {}, root)
        changed = build_recipe_identity(
            EEG_COHERENCE_STAGE_SPEC, {"minimum_support": 6}, root
        )
        self.assertNotEqual(baseline.recipe_id, changed.recipe_id)

    def test_support_boundary_common_quality_and_linear_iqr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=5)
            payload["stage_code_aligned"][:] = 4.0
            payload["X_seq"][:] = 0.0
            payload["X_seq"][:, 54] = np.arange(5, dtype=np.float32)
            payload["eeg_common_clean_subsegment_count"][-1] = 0
            result = self._result(Path(directory), payload)
            self.assertFalse(result.valid_mask[0])
            self.assertEqual(result.support_count[0], 4)
            self.assertTrue(np.isnan(result.values[0]))

            payload["eeg_common_clean_subsegment_count"][-1] = 1
            result = self._result(Path(directory), payload)
            self.assertTrue(result.valid_mask[0])
            self.assertEqual(result.values[0], 2.0)
            self.assertEqual(
                result.values[1],
                np.quantile(np.arange(5), 0.75, method="linear")
                - np.quantile(np.arange(5), 0.25, method="linear"),
            )

    def test_both_leads_required_and_invalid_zero_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=5)
            payload["stage_code_aligned"][:] = 4.0
            payload["X_seq"][:] = 0.0
            payload["eeg_channel_available"][1] = False
            result = self._result(Path(directory), payload)
            self.assertFalse(result.valid_mask[0])
            self.assertEqual(result.missing_reason[0], "PAIR_UNAVAILABLE")

            c3c4_rem_mean = 30
            self.assertTrue(result.valid_mask[c3c4_rem_mean])
            self.assertEqual(result.values[c3c4_rem_mean], 0.0)

            payload["eeg_common_clean_subsegment_count"][:] = 0
            result = self._result(Path(directory), payload)
            self.assertFalse(result.valid_mask[c3c4_rem_mean])
            self.assertEqual(result.support_count[c3c4_rem_mean], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
