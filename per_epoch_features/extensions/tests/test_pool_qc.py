from __future__ import annotations
import unittest
import numpy as np
from extensions.pool_qc import summarize_matrix, public_feature_rows

class PoolQCTests(unittest.TestCase):
    def test_qc_preserves_zero_and_reports_missing(self):
        values = np.asarray([[0.0, np.nan, 2.0], [1.0, np.nan, np.nan]], dtype=np.float64)
        valid = np.asarray([[True, False, True], [True, False, False]])
        support = np.asarray([[5, 0, 3], [4, 0, 0]], dtype=np.int64)
        summary = summarize_matrix(values, valid, support, np.asarray([[0, 7, 0], [0, 7, 9]], dtype=np.int64), feature_ids=('a','b','c'))
        self.assertEqual(summary['overall']['valid_count'], 3)
        self.assertEqual(summary['overall']['missing_count'], 3)
        self.assertEqual(summary['features']['a']['zero_count'], 1)
        self.assertEqual(summary['features']['b']['missing_reason_counts']['7'], 2)
        grouped = summarize_matrix(values, valid, support, np.asarray([[0, 7, 0], [0, 7, 9]], dtype=np.int64), feature_ids=('a','b','c'), groups=('I0002','S0001'))
        self.assertEqual(grouped['groups']['I0002']['valid_count'], 2)
        self.assertEqual(grouped['groups']['I0002']['zero_count'], 1)
        public_rows = public_feature_rows(summary)
        self.assertNotIn('support_min', public_rows[0])
        self.assertNotIn('support_median', public_rows[0])
        self.assertNotIn('support_max', public_rows[0])

if __name__ == '__main__': unittest.main()
