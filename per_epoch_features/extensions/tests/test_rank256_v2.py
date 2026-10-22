from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from extensions.rank256_v2 import (
    fit_reference, rank_against_reference, quantize_rank256,
    build_loso_indices, build_experiment_selection,
    map_ranked_views, publish_unit_atomic, validate_unit,
    UnitPublishError, build_rank_unit,
)


class Rank256V2Tests(unittest.TestCase):
    def test_average_ties_endpoints_singleton_and_empty_reference(self):
        values=np.array([[0.,1.,2.,np.nan],[0.,2.,2.,5.],[1.,3.,4.,6.]],np.float64)
        mask=np.array([[True,True,True,False],[True,True,True,True],[True,True,True,True]])
        ref=fit_reference(values,mask)
        pct,rank=rank_against_reference(values,mask,ref)
        np.testing.assert_allclose(pct[:3,:3], [[1/3,1/6,1/3],[1/3,0.5,1/3],[5/6,5/6,5/6]])
        self.assertFalse(rank[0,3]); self.assertTrue(np.isnan(pct[0,3]))
        singleton=fit_reference(np.array([[7.]],np.float64),np.ones((1,1),bool))
        p,r=rank_against_reference(np.array([[7.],[6.],[8.]]),np.ones((3,1),bool),singleton)
        np.testing.assert_allclose(p[:,0],[0.5,0.,1.]); self.assertTrue(r.all())
        empty=fit_reference(np.empty((0,1)),np.empty((0,1),bool))
        p,r=rank_against_reference(np.array([[2.]]),np.ones((1,1),bool),empty)
        self.assertFalse(r[0,0]); self.assertTrue(np.isnan(p[0,0]))

    def test_quantization_half_up_grid_and_masks(self):
        p=np.array([[0.,0.5,1.,np.nan,127.5/255.]],np.float64)
        source=np.array([[True,True,True,False,True]])
        q,k=quantize_rank256(p,source)
        np.testing.assert_array_equal(k,[[0,128,255,-1,128]])
        np.testing.assert_allclose(q[0,[0,1,2,4]],[0.,128/255,1.,128/255],rtol=0,atol=0)
        self.assertTrue(np.isnan(q[0,3]))

    def test_source_and_rank_masks_are_separate(self):
        values=np.array([[1.,np.nan]],np.float64); source=np.array([[True,False]])
        ref=fit_reference(np.array([[2.,3.]],np.float64),np.ones((1,2),bool))
        pct,rank=rank_against_reference(values,source,ref)
        self.assertTrue(source[0,0]); self.assertFalse(source[0,1]); self.assertTrue(rank[0,0]); self.assertFalse(rank[0,1])
        self.assertTrue(np.isnan(pct[0,1]))

    def test_inductive_reference_does_not_see_holdout(self):
        train=np.array([[0.],[1.]],np.float64); tm=np.ones_like(train,bool)
        ref=fit_reference(train,tm); before=ref.fingerprint
        test=np.array([[0.5],[100.]],np.float64); p1,_=rank_against_reference(test,np.ones_like(test,bool),ref)
        altered=np.array([[0.5],[-100.]],np.float64); p2,_=rank_against_reference(altered,np.ones_like(altered,bool),ref)
        self.assertEqual(before,ref.fingerprint); self.assertEqual(p1[0,0],p2[0,0])

    def test_loso_indices_are_site_isolated_and_order_preserving(self):
        sites=np.array(['S0001','I0002','S0001','I0006','S0001'])
        fold=build_loso_indices(sites,'I0002')
        np.testing.assert_array_equal(fold['train_indices'],[0,2,4]); np.testing.assert_array_equal(fold['test_indices'],[1]); np.testing.assert_array_equal(fold['unused_indices'],[3])
        self.assertEqual(fold['train_site'],'S0001'); self.assertEqual(fold['unused_site'],'I0006')

    def test_experiment_selection_e2_is_not_first_1060(self):
        ids=tuple(f'f{i}' for i in range(1444)); selections=build_experiment_selection(ids)
        self.assertEqual(selections['E2']['indices'][:3],[0,1,2]); self.assertEqual(selections['E2']['indices'][-3:],[1441,1442,1443]); self.assertNotEqual(selections['E2']['indices'],list(range(1060)))
        self.assertEqual(selections['E2']['feature_ids'][-1],'f1443')

    def test_view_mapping_reuses_unique_columns_and_dimensions(self):
        values=np.arange(1444,dtype=np.float64).reshape(1,-1); masks=np.ones_like(values,bool)
        rows=[]
        dims={'V1':161,'V2':177,'V3':493,'V4':257,'V5':376}
        for view,width in dims.items():
            for pos in range(width):
                idx=pos if pos<807 else 807+(pos % 637)
                rows.append({'view_id':view,'view_position':pos,'feature_id':f'f{idx}','unique_column_index':idx,'source_block':'old807' if idx<807 else 'new'})
        mapped=map_ranked_views(values,masks,tuple(f'f{i}' for i in range(1444)),rows)
        self.assertEqual({k:v[0].shape[1] for k,v in mapped.items()},dims)
        self.assertEqual(mapped['V1'][0][0,0],0.)

    def test_build_unit_preserves_train_protocol_equality_and_empty_reference(self):
        values=np.full((4,1444), np.nan, np.float64)
        values[:, 0] = [0., 1., 3., 2.]
        values[:, 1] = [np.nan, np.nan, 4., 3.]
        mask=np.isfinite(values); sites=np.array(['S0001','I0002','S0001','I0006'])
        ids=tuple(f'f{i}' for i in range(1444))
        unit=build_rank_unit(values,mask,ids,sites,'I0002','inductive')
        self.assertEqual(unit['train_percentile'].shape,(2,1444)); self.assertFalse(unit['test_rank_valid_mask'][0,1])

    def test_atomic_unit_publish_readback_and_no_clobber(self):
        payload={'x':np.array([[1.,np.nan]]), 'mask':np.array([[True,False]])}
        metadata={'schema':'rank256_v2','unit':'synthetic','identity':'abc'}
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'unit'
            result=publish_unit_atomic(target,payload,metadata)
            self.assertEqual(result['status'],'BUILD_NEW'); self.assertTrue(validate_unit(target,metadata))
            reused=publish_unit_atomic(target,payload,metadata)
            self.assertEqual(reused['status'],'SKIP_VERIFIED')
            with self.assertRaises(UnitPublishError): publish_unit_atomic(target,payload,dict(metadata,identity='different'))
            (target/'arrays.npz').write_bytes(b'bad')
            self.assertFalse(validate_unit(target,metadata))

    def test_reference_is_column_isolated(self):
        train_a=np.array([[0., 10.], [1., 20.], [2., 30.]], np.float64)
        train_b=np.array([[0., 100.], [1., 200.], [2., 300.]], np.float64)
        mask=np.ones_like(train_a, bool)
        ref_a=fit_reference(train_a, mask)
        ref_b=fit_reference(train_b, mask)
        probe=np.array([[1., 25.]], np.float64)
        pa,_=rank_against_reference(probe, np.ones_like(probe,bool), ref_a)
        pb,_=rank_against_reference(probe, np.ones_like(probe,bool), ref_b)
        self.assertEqual(pa[0,0], pb[0,0])

    def test_invalid_percentile_mask_semantics_are_rejected(self):
        with self.assertRaises(ValueError):
            quantize_rank256(np.array([[np.nan]], np.float64), np.array([[True]], bool))

    def test_manifest_and_input_identity_mismatch_rejected(self):
        with self.assertRaises(ValueError): map_ranked_views(np.zeros((1,2)),np.ones((1,2),bool),('a','b'),[{'view_id':'V1','view_position':0,'feature_id':'wrong','unique_column_index':0,'source_block':'old807'}])

if __name__=='__main__': unittest.main()
