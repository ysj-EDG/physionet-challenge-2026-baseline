from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from extensions.caisr_stage import (
    CAISR_STAGE_SPEC,
    STAGE_CODES,
    bout_lengths_by_stage,
    extract_family,
)
from extensions.recipe import build_recipe_identity
from extensions.sources import load_source_npz
from extensions.tests.fixtures import source_payload, write_source_npz


STATIC_P50 = {5: 44, 3: 45, 2: 46, 1: 47, 4: 48}
STATIC_P95 = {5: 69, 3: 70, 2: 71, 1: 72, 4: 73}


class CaisrStageTests(unittest.TestCase):
    def _source(self, root: Path, payload):
        source_root = root / "source"
        return load_source_npz(write_source_npz(source_root, payload=payload), source_root)

    def test_dimensions_order_static_indices_and_recipe(self) -> None:
        definitions = CAISR_STAGE_SPEC.feature_definitions
        self.assertEqual(len(definitions), 35)
        self.assertEqual(len({item.feature_id for item in definitions}), 35)
        self.assertEqual(STAGE_CODES, (("N3", 1), ("N2", 2), ("N1", 3), ("REM", 4), ("Wake", 5)))
        self.assertEqual(
            [item.source_columns for item in definitions[25:]],
            [(47,), (72,), (46,), (71,), (45,), (70,), (48,), (73,), (44,), (69,)],
        )
        root = Path(__file__).resolve().parents[2]
        baseline = build_recipe_identity(CAISR_STAGE_SPEC, {}, root)
        changed = build_recipe_identity(CAISR_STAGE_SPEC, {"static_tolerance": 1e-4}, root)
        self.assertNotEqual(baseline.recipe_id, changed.recipe_id)

    def test_transition_rows_normalize_and_gaps_do_not_connect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=8)
            payload["stage_code_aligned"] = np.asarray([1, 1, 2, 1, 3, 1, 4, 5], dtype=np.float32)
            payload["stage_valid"][:] = True
            payload["stage_valid"][3] = False
            payload["epoch_start_sec"][6:] += 30.0
            source = self._source(Path(directory), payload)
            result = extract_family(source, CAISR_STAGE_SPEC.default_config)
            n3_row = result.values[:5]
            np.testing.assert_allclose(n3_row, [0.5, 0.5, 0.0, 0.0, 0.0])
            self.assertAlmostEqual(float(np.sum(n3_row)), 1.0)
            self.assertTrue(np.all(result.valid_mask[:5]))
            self.assertTrue(np.all(np.isnan(result.values[5:10])))
            self.assertTrue(np.all(~result.valid_mask[5:10]))
            self.assertTrue(np.all(result.support_count[:5] == 2))

    def test_static_bouts_match_direct_grid_semantics_and_reorder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stages = np.asarray(
                [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                dtype=np.float32,
            )
            payload = source_payload(epochs=len(stages))
            payload["stage_code_aligned"] = stages
            payload["stage_valid"][:] = True
            payload["stage_valid"][10] = False
            payload["epoch_start_sec"][16:] += 30.0
            bouts = bout_lengths_by_stage(
                stages, payload["stage_valid"], payload["epoch_start_sec"]
            )
            for code in (1, 2, 3, 4, 5):
                payload["x_static"][STATIC_P50[code]] = np.quantile(
                    bouts[code], 0.50, method="linear"
                )
                payload["x_static"][STATIC_P95[code]] = np.quantile(
                    bouts[code], 0.95, method="linear"
                )
            source = self._source(Path(directory), payload)
            result = extract_family(source, CAISR_STAGE_SPEC.default_config)
            expected = []
            supports = []
            for _name, code in STAGE_CODES:
                expected.extend(
                    [
                        np.quantile(bouts[code], 0.50, method="linear"),
                        np.quantile(bouts[code], 0.95, method="linear"),
                    ]
                )
                supports.extend([len(bouts[code]), len(bouts[code])])
            np.testing.assert_allclose(result.values[25:], expected)
            np.testing.assert_array_equal(result.support_count[25:], supports)
            self.assertTrue(np.all(result.valid_mask[25:]))

    def test_absent_stage_zero_fill_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = source_payload(epochs=6)
            payload["stage_code_aligned"][:] = 2.0
            payload["x_static"][:] = 0.0
            source = self._source(Path(directory), payload)
            result = extract_family(source, CAISR_STAGE_SPEC.default_config)
            n3_p50 = 25
            self.assertFalse(result.valid_mask[n3_p50])
            self.assertTrue(np.isnan(result.values[n3_p50]))
            self.assertEqual(result.support_count[n3_p50], 0)
            self.assertEqual(result.missing_reason[n3_p50], "NO_STAGE_BOUT_SUPPORT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
