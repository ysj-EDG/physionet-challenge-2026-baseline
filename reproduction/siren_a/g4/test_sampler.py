import unittest
import numpy as np

from reproduction.siren_a.g4.sampler import allocate_quotas, sample_training_rows, target_size


class SamplerTests(unittest.TestCase):
    def test_half_up_target_and_largest_remainder(self):
        self.assertEqual(target_size(5139), 3597)
        self.assertEqual(target_size(1142), 799)
        strata={(10,0):3,(10,1):3,(11,0):4,(11,1):4}
        quotas=allocate_quotas(strata, target_size(14))
        self.assertEqual(sum(quotas.values()), 10)
        self.assertTrue(all(1 <= quotas[k] <= strata[k] for k in strata))

    def test_tie_break_seed_and_order_are_deterministic(self):
        groups=np.array([10]*8+[11]*8, dtype=np.int64)
        labels=np.array(([0,1]*4)*2, dtype=np.int8)
        global_indices=np.arange(100,116,dtype=np.int64)
        first=sample_training_rows("S0001",42,groups,labels,global_indices)
        second=sample_training_rows("S0001",42,groups,labels,global_indices)
        other=sample_training_rows("S0001",43,groups,labels,global_indices)
        np.testing.assert_array_equal(first.local_positions, second.local_positions)
        self.assertFalse(np.array_equal(first.local_positions, other.local_positions))
        self.assertTrue(np.all(first.local_positions[:-1] < first.local_positions[1:]))
        self.assertEqual(len(np.unique(first.global_indices)), len(first.global_indices))
        self.assertEqual(first.target_n, target_size(len(groups)))

    def test_every_nonempty_stratum_is_retained_and_infeasible_rejected(self):
        groups=np.arange(20,dtype=np.int64)
        labels=np.zeros(20,dtype=np.int8)
        with self.assertRaises(ValueError):
            sample_training_rows("X",42,groups,labels,np.arange(20,dtype=np.int64))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            sample_training_rows("X",42,np.array([10,10]),np.array([0,2]),np.array([1,2]))
        with self.assertRaises(ValueError):
            allocate_quotas({(10,0):0}, 1)


if __name__ == "__main__": unittest.main()
