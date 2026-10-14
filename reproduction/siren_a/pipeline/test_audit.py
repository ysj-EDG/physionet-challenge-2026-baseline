import sys
import unittest
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

from audit import audit_rank, audit_raw, compare_protocols
from normalize import quantize_rank, rank_by_site


class AuditTests(unittest.TestCase):
    def test_raw_audit_counts_ties_zero_and_missing(self):
        x=np.array([[0.,np.nan],[0.,2.],[1.,2.]])
        m=np.isfinite(x)
        rows=audit_raw(x,m,['a','b'],['F','F'],'A')
        self.assertEqual(rows[0]['n_valid'],3); self.assertEqual(rows[0]['lawful_zero_count'],2)
        self.assertEqual(rows[0]['max_tie_count'],2); self.assertEqual(rows[1]['n_missing'],1)

    def test_rank_audit_enforces_grid_mask_and_monotonicity(self):
        raw=np.array([[2.],[2.],[5.],[9.]])
        source=np.ones_like(raw,bool); pct,rank=rank_by_site(raw,source,np.array(['A']*4)); q,k=quantize_rank(pct,rank)
        rows=audit_rank(raw,pct,q,source,rank,{'levels':k,'feature_ids':['x'],'families':['F'],'reference_n':[4]})
        self.assertEqual(rows[0]['n_percentile_unique'],3)
        self.assertLessEqual(rows[0]['max_abs_quantization_error'],1/510+1e-15)
        bad=q.copy(); bad[0,0]=0.123
        with self.assertRaises(ValueError): audit_rank(raw,pct,bad,source,rank,{'levels':k,'feature_ids':['x'],'families':['F'],'reference_n':[4]})

    def test_protocol_comparison_reports_difference_and_train_identity(self):
        train=np.array([[0.2],[0.8]])
        test_t=np.array([[0.25],[0.75]]); test_i=np.array([[0.],[1.]])
        report=compare_protocols(train,train.copy(),test_t,test_i,np.ones_like(train,bool),np.ones_like(test_t,bool))
        self.assertTrue(report['train_exact']); self.assertEqual(report['test_different_count'],2)
        with self.assertRaises(ValueError): compare_protocols(train,train+0.1,test_t,test_i,np.ones_like(train,bool),np.ones_like(test_t,bool))

    def test_rank_audit_rejects_finite_source_missing_and_inf(self):
        raw=np.array([[1.],[np.nan]]); source=np.array([[True],[False]])
        pct=np.array([[0.5],[0.0]]); rank=np.array([[True],[False]]); q=np.array([[128/255],[np.nan]]); k=np.array([[128],[-1]],np.int16)
        with self.assertRaises(ValueError): audit_rank(raw,pct,q,source,rank,{'levels':k,'feature_ids':['x'],'families':['F'],'reference_n':[1]})


if __name__ == '__main__': unittest.main(verbosity=2)
