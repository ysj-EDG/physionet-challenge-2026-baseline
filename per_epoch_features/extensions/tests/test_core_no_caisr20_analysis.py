import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from extensions.core_no_caisr20_analysis import _rank_fuse, _write_csv


class CoreNoCAISRAnalysisTests(unittest.TestCase):
    def test_rank_fusion_is_mean_tie_rank(self):
        values = np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float64)
        np.testing.assert_allclose(_rank_fuse(values), np.asarray([0.5, 0.5]))

    def test_rank_fusion_preserves_exact_column_count(self):
        values = np.arange(225.0, dtype=np.float64)[None, :]
        fused = _rank_fuse(values)
        self.assertEqual(fused.shape, (1,))
        self.assertTrue(np.isfinite(fused).all())

    def test_public_writer_accepts_route_and_comparison_rows(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _write_csv(path, [{"row_type": "route_summary", "route": "CORE", "scope": "Macro"}, {"row_type": "comparison", "comparison": "DEDUP-CORE", "scope": "Macro", "delta": 0.1}])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertIn("route", lines[0])
            self.assertIn("comparison", lines[0])
            self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
