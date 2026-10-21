from __future__ import annotations
import unittest
import numpy as np
from extensions.pool_qc import summarize_matrix, public_feature_rows, summarize_scopes, public_scope_feature_rows

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

    def test_scope_qc_closes_and_keeps_no_valid_as_not_computable(self):
        values = np.asarray([[0.0, np.nan], [np.nan, 2.0], [np.nan, np.nan]], dtype=np.float64)
        valid = np.asarray([[True, False], [False, True], [False, False]])
        support = np.asarray([[2, 0], [0, 3], [0, 0]], dtype=np.int64)
        reasons = np.asarray([[0, 7], [8, 0], [9, 9]], dtype=np.int64)
        scoped = summarize_scopes(values, valid, support, reasons,
                                  feature_ids=('a','b'), groups=('I0002','S0001','S0001'),
                                  reason_dictionary={0:'VALID',7:'LOW_SUPPORT',8:'CHANNEL_UNAVAILABLE',9:'NO_SUPPORT'})
        self.assertEqual(set(scoped), {'Overall','I0002','I0006','S0001'})
        self.assertEqual(scoped['I0002']['features']['a']['valid_count'], 1)
        self.assertTrue(scoped['I0006']['features']['a']['all_missing'])
        self.assertIsNone(scoped['I0006']['features']['a']['valid_rate'])
        rows = public_scope_feature_rows(scoped)
        self.assertEqual(len(rows), 8)
        self.assertEqual({row['scope'] for row in rows}, {'Overall','I0002','I0006','S0001'})
        self.assertEqual(sum(row['n_total'] for row in rows if row['feature_id']=='a' and row['scope']=='Overall'), 3)

if __name__ == '__main__': unittest.main()
