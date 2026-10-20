from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from extensions.recipe import build_recipe_identity
from extensions.resp_stage import RESP_STAGE_SPEC, extract_family
from extensions.sources import load_source_npz
from extensions.tests.fixtures import source_payload, write_source_npz


class RespStageTests(unittest.TestCase):
    def _result(self, root: Path, payload):
        source_root = root / "source"
        source = load_source_npz(write_source_npz(source_root, payload=payload), source_root)
        return extract_family(source, RESP_STAGE_SPEC.default_config)

    def test_dimensions_all_columns_order_and_recipe(self) -> None:
        definitions = RESP_STAGE_SPEC.feature_definitions
        self.assertEqual(len(definitions), 56)
        self.assertEqual(len({item.feature_id for item in definitions}), 56)
        self.assertEqual(
            [definitions[index].source_columns for index in range(0, 56, 4)],
            [(column,) for column in range(456, 470)],
        )
        root = Path(__file__).resolve().parents[2]
        baseline = build_recipe_identity(RESP_STAGE_SPEC, {}, root)
        changed = build_recipe_identity(RESP_STAGE_SPEC, {"minimum_support": 6}, root)
        self.assertNotEqual(baseline.recipe_id, changed.recipe_id)

    def test_feature_mask_support_boundary_and_linear_iqr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=5)
            payload["stage_code_aligned"][:] = 4.0
            payload["X_seq"][:] = 0.0
            payload["X_seq"][:, 456] = np.arange(5, dtype=np.float32)
            payload["resp_feature_valid"][-1, 0] = False
            result = self._result(Path(directory), payload)
            self.assertFalse(result.valid_mask[0])
            self.assertEqual(result.support_count[0], 4)

            payload["resp_feature_valid"][-1, 0] = True
            result = self._result(Path(directory), payload)
            self.assertTrue(result.valid_mask[0])
            self.assertEqual(result.values[0], 2.0)
            self.assertEqual(result.values[1], 2.0)

    def test_independent_feature_missingness_valid_zero_and_stage_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=5)
            payload["stage_code_aligned"][:] = 2.0
            payload["X_seq"][:] = 0.0
            payload["resp_feature_valid"][:, 0] = False
            result = self._result(Path(directory), payload)
            self.assertFalse(result.valid_mask[0])
            self.assertFalse(result.valid_mask[2])
            second_feature_nrem_mean = 6
            self.assertTrue(result.valid_mask[second_feature_nrem_mean])
            self.assertEqual(result.values[second_feature_nrem_mean], 0.0)
            self.assertFalse(result.valid_mask[4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
