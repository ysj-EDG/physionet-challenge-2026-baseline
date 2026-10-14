import unittest
import numpy as np

from reproduction.siren_a.g4.queries import audit_ac_pair_coverage, build_queries


class QueryTests(unittest.TestCase):
    def test_stable_grouping_and_dynamic_truncation(self):
        ages=np.array([60,55,60,55,64,59],dtype=np.float64)
        labels=np.array([1,0,0,1,1,0],dtype=np.int8)
        local=np.arange(6,dtype=np.int64); global_idx=local+10
        q=build_queries(ages,labels,local,global_idx)
        np.testing.assert_array_equal(q.permutation,np.array([1,3,5,0,2,4]))
        np.testing.assert_array_equal(q.group_sizes,np.array([3,3]))
        self.assertEqual(q.truncation_level,3)
        self.assertEqual(int(q.group_sizes.sum()),6)
        np.testing.assert_array_equal(q.sorted_labels,labels[q.permutation])

    def test_single_class_query_is_rejected(self):
        with self.assertRaises(ValueError):
            build_queries(np.array([55.,56.,60.,61.]),np.array([0,0,0,1],dtype=np.int8),np.arange(4),np.arange(4))

    def test_pair_coverage(self):
        ages=np.array([59.,58.,60.,62.])
        labels=np.array([1,0,0,1],dtype=np.int8)
        groups=np.floor(ages/5).astype(np.int64)
        audit=audit_ac_pair_coverage(ages,labels,groups)
        self.assertEqual(audit["all_valid_pairs"],3)
        self.assertEqual(audit["within_group_valid_pairs"],2)
        self.assertAlmostEqual(audit["pair_coverage"],2/3)

    def test_zero_pairs_rejected(self):
        with self.assertRaises(ValueError):
            audit_ac_pair_coverage(np.array([50.,80.]),np.array([0,1]),np.array([10,16]))


if __name__ == "__main__": unittest.main()
