from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from extensions.emg_stage import EMG_STAGE_SPEC, extract_family
from extensions.recipe import build_recipe_identity
from extensions.sources import load_source_npz
from extensions.tests.fixtures import source_payload, write_source_npz


class EmgStageTests(unittest.TestCase):
    def _result(self, root: Path, payload):
        source_root = root / "source"
        source = load_source_npz(write_source_npz(source_root, payload=payload), source_root)
        return extract_family(source, EMG_STAGE_SPEC.default_config)

    def test_dimensions_offsets_stage_order_and_recipe(self) -> None:
        definitions = EMG_STAGE_SPEC.feature_definitions
        self.assertEqual(len(definitions), 72)
        self.assertEqual(len({item.feature_id for item in definitions}), 72)
        self.assertEqual(
            [definitions[index].source_columns for index in (0, 6, 12, 18)],
            [(432,), (435,), (436,), (438,)],
        )
        self.assertEqual(definitions[24].source_columns, (440,))
        self.assertEqual(definitions[48].source_columns, (448,))
        root = Path(__file__).resolve().parents[2]
        baseline = build_recipe_identity(EMG_STAGE_SPEC, {}, root)
        changed = build_recipe_identity(EMG_STAGE_SPEC, {"minimum_support": 6}, root)
        self.assertNotEqual(baseline.recipe_id, changed.recipe_id)

    def test_support_channel_preprocessing_and_epoch_validity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=5)
            payload["stage_code_aligned"][:] = 5.0
            payload["X_seq"][:] = 0.0
            payload["X_seq"][:, 432] = np.arange(5, dtype=np.float32)
            payload["emg_epoch_success"][-1, 0] = False
            payload["emg_channel_available"][1] = False
            payload["emg_preprocessing_success"][2] = False
            result = self._result(Path(directory), payload)
            self.assertFalse(result.valid_mask[0])
            self.assertEqual(result.support_count[0], 4)
            self.assertEqual(result.missing_reason[0], "INSUFFICIENT_SUPPORT")
            self.assertFalse(result.valid_mask[24])
            self.assertEqual(result.missing_reason[24], "CHANNEL_UNAVAILABLE")
            self.assertFalse(result.valid_mask[48])
            self.assertEqual(result.missing_reason[48], "PREPROCESSING_FAILED")

            payload["emg_epoch_success"][-1, 0] = True
            result = self._result(Path(directory), payload)
            self.assertTrue(result.valid_mask[0])
            self.assertEqual(result.values[0], 2.0)
            self.assertEqual(result.values[1], 2.0)

    def test_valid_zero_and_independent_stage_missingness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=5)
            payload["stage_code_aligned"][:] = 4.0
            payload["X_seq"][:] = 0.0
            result = self._result(Path(directory), payload)
            chin_wake_mean = 0
            chin_rem_mean = 2
            self.assertFalse(result.valid_mask[chin_wake_mean])
            self.assertTrue(result.valid_mask[chin_rem_mean])
            self.assertEqual(result.values[chin_rem_mean], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
