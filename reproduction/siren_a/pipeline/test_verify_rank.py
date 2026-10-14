import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from verify_rank import (
    build_reference,
    check_quantized,
    check_reference,
    quantize_values,
    rank_values,
)


class VerifyRankTests(unittest.TestCase):
    def test_average_ties_use_literal_expected_values(self):
        values=np.array([[2.0],[2.0],[5.0],[9.0]])
        mask=np.ones_like(values,dtype=bool)
        ref=build_reference(values,mask)
        got,valid=rank_values(values,mask,ref)
        np.testing.assert_array_equal(valid,mask)
        np.testing.assert_array_equal(got[:,0],np.array([0.25,0.25,0.625,0.875]))

    def test_continuous_binary_constant_singleton_and_zero(self):
        values=np.array([
            [-2.0,0.0,7.0,4.0],
            [0.0,1.0,7.0,np.nan],
            [3.0,1.0,7.0,np.nan],
        ])
        mask=np.array([
            [1,1,1,1],
            [1,1,1,0],
            [1,1,1,0],
        ],dtype=bool)
        got,valid=rank_values(values,mask,build_reference(values,mask))
        np.testing.assert_array_equal(got[:,0],np.array([1/6,0.5,5/6]))
        np.testing.assert_array_equal(got[:,1],np.array([1/6,2/3,2/3]))
        np.testing.assert_array_equal(got[:,2],np.array([0.5,0.5,0.5]))
        self.assertEqual(got[0,3],0.5)
        self.assertTrue(np.isnan(got[1:,3]).all())
        np.testing.assert_array_equal(valid,mask)

    def test_all_missing_reference_preserves_source_mask_but_rank_invalid(self):
        reference_values=np.array([[np.nan],[np.nan]])
        reference_mask=np.zeros((2,1),dtype=bool)
        query=np.array([[3.0],[np.nan]])
        source=np.array([[1],[0]],dtype=bool)
        got,rank_mask=rank_values(query,source,build_reference(reference_values,reference_mask))
        self.assertTrue(source[0,0])
        self.assertFalse(rank_mask.any())
        self.assertTrue(np.isnan(got).all())

    def test_below_and_above_reference_map_to_zero_and_one(self):
        reference=build_reference(np.array([[2.0],[5.0],[9.0]]),np.ones((3,1),bool))
        got,mask=rank_values(np.array([[1.0],[10.0]]),np.ones((2,1),bool),reference)
        np.testing.assert_array_equal(got[:,0],np.array([0.0,1.0]))
        self.assertTrue(mask.all())

    def test_order_invariance(self):
        values=np.array([[9.0],[2.0],[5.0],[2.0]])
        mask=np.ones((4,1),bool)
        a,_=rank_values(values,mask,build_reference(values,mask))
        order=np.array([2,0,3,1])
        b,_=rank_values(values[order],mask[order],build_reference(values[order],mask[order]))
        inverse=np.argsort(order)
        np.testing.assert_array_equal(a,b[inverse])

    def test_valid_inf_is_rejected(self):
        with self.assertRaises(ValueError):
            build_reference(np.array([[np.inf]]),np.ones((1,1),bool))
        ref=build_reference(np.array([[1.0]]),np.ones((1,1),bool))
        with self.assertRaises(ValueError):
            rank_values(np.array([[-np.inf]]),np.ones((1,1),bool),ref)

    def test_reference_serialization_and_fingerprint(self):
        values=np.array([[2.0,7.0],[2.0,np.nan],[5.0,np.nan]])
        mask=np.array([[1,1],[1,0],[1,0]],bool)
        ref=build_reference(values,mask)
        saved={
            "train_values":np.array([2.0,2.0,5.0,7.0]),
            "train_offsets":np.array([0,3,4],np.int64),
            "train_counts":np.array([3,1],np.int64),
            "train_fingerprint":np.asarray(ref.fingerprint),
        }
        self.assertTrue(check_reference(saved,ref,"train"))

    def test_reference_mutations_are_rejected(self):
        values=np.array([[2.0],[2.0],[5.0]])
        ref=build_reference(values,np.ones((3,1),bool))
        base={
            "train_values":np.array([2.0,2.0,5.0]),
            "train_offsets":np.array([0,3],np.int64),
            "train_counts":np.array([3],np.int64),
            "train_fingerprint":np.asarray(ref.fingerprint),
        }
        mutations=[
            {**base,"train_values":np.array([2.0,5.0])},
            {**base,"train_values":np.array([2.0,2.0,2.0,5.0])},
            {**base,"train_values":np.array([5.0,2.0,2.0])},
            {**base,"train_offsets":np.array([0,2],np.int64)},
            {**base,"train_counts":np.array([2],np.int64)},
            {**base,"train_fingerprint":np.asarray("0"*64)},
        ]
        for mutation in mutations:
            with self.subTest(keys=[k for k in mutation if not np.array_equal(mutation[k],base.get(k,mutation[k]))]):
                with self.assertRaises(ValueError):
                    check_reference(mutation,ref,"train")

    def test_quantization_half_up_and_missing(self):
        percentile=np.array([[0.0,0.3,0.5,1.0,np.nan]])
        mask=np.array([[1,1,1,1,0]],bool)
        q,k=quantize_values(percentile,mask)
        np.testing.assert_array_equal(k,np.array([[0,77,128,255,-1]],np.int16))
        np.testing.assert_array_equal(q[mask],k[mask].astype(float)/255.0)
        self.assertTrue(np.isnan(q[0,4]))
        self.assertEqual(q[0,2],128/255)

    def test_quantization_formula_rejects_adjacent_self_consistent_level(self):
        percentile=np.array([[0.3,0.5,np.nan]])
        mask=np.array([[1,1,0]],bool)
        expected_q,expected_k=quantize_values(percentile,mask)
        self.assertTrue(check_quantized(expected_k,expected_q,percentile,mask))
        wrong_k=expected_k.copy(); wrong_k[0,0]-=1
        wrong_q=expected_q.copy(); wrong_q[0,0]=wrong_k[0,0]/255
        with self.assertRaises(ValueError):
            check_quantized(wrong_k,wrong_q,percentile,mask)

    def test_quantization_rejects_missing_as_zero_and_off_grid(self):
        percentile=np.array([[0.2,np.nan]])
        mask=np.array([[1,0]],bool)
        q,k=quantize_values(percentile,mask)
        bad_k=k.copy(); bad_k[0,1]=0
        with self.assertRaises(ValueError):
            check_quantized(bad_k,q,percentile,mask)
        bad_q=q.copy(); bad_q[0,0]+=1e-6
        with self.assertRaises(ValueError):
            check_quantized(k,bad_q,percentile,mask)

    def test_quantization_bounds_monotonicity_and_error(self):
        percentile=np.linspace(0,1,1001).reshape(-1,1)
        mask=np.ones_like(percentile,dtype=bool)
        q,k=quantize_values(percentile,mask)
        self.assertTrue(np.all(np.diff(k[:,0])>=0))
        self.assertLessEqual(np.max(np.abs(q-percentile)),1/510+1e-14)
        self.assertLessEqual(len(np.unique(k)),256)


if __name__=="__main__":
    unittest.main()
