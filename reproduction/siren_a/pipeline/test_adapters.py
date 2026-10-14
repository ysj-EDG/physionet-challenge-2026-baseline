import unittest

import numpy as np

from reproduction.siren_a.pipeline.adapters import (
    NEUTRAL_RANK256,
    AdapterState,
    check_adapter,
    fit_adapter,
    transform_adapter,
)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[0.0,np.nan,np.nan],[0.5,0.2,np.nan],[1.0,0.8,np.nan],[np.nan,0.4,np.nan]],np.float64)
        self.ids=("a","b","c")

    def test_lr_median_and_all_missing_fallback(self):
        state=fit_adapter(self.X,"lr",self.ids,fit_indices=np.arange(4),fit_scope="full_train_preview")
        self.assertIsInstance(state,AdapterState)
        np.testing.assert_array_equal(state.finite_counts,[3,3,0])
        np.testing.assert_allclose(state.fill_values,[0.5,0.4,NEUTRAL_RANK256])
        np.testing.assert_array_equal(state.all_missing,[False,False,True])
        out=transform_adapter(self.X,state)
        self.assertFalse(np.isnan(out).any()); self.assertEqual(out.shape,self.X.shape)
        np.testing.assert_array_equal(out[np.isfinite(self.X)],self.X[np.isfinite(self.X)])
        self.assertTrue(check_adapter(self.X,out,state))

    def test_lr_fit_subset_only_and_test_never_changes_state(self):
        selected=np.array([0,1],np.int64)
        first=fit_adapter(self.X[selected],"lr",self.ids,fit_indices=selected,fit_scope="seed_subset")
        changed=self.X.copy(); changed[2:]=123.0
        second=fit_adapter(changed[selected],"lr",self.ids,fit_indices=selected,fit_scope="seed_subset")
        np.testing.assert_array_equal(first.fill_values,second.fill_values)
        test=np.array([[np.nan,0.9,np.nan]],np.float64)
        before=first.fill_values.copy(); transform_adapter(test,state=first); np.testing.assert_array_equal(before,first.fill_values)
        altered=self.X[selected].copy(); altered[1,0]=0.75
        third=fit_adapter(altered,"lr",self.ids,fit_indices=selected,fit_scope="seed_subset")
        self.assertNotEqual(first.fill_values[0],third.fill_values[0])

    def test_passthrough_models_preserve_nan_and_dimensions(self):
        for model in ("hgb","lambdarank"):
            with self.subTest(model=model):
                state=fit_adapter(self.X,model,self.ids,fit_indices=np.arange(4))
                out=transform_adapter(self.X,state)
                np.testing.assert_array_equal(out,self.X)
                self.assertEqual(out.dtype,np.float64); self.assertTrue(check_adapter(self.X,out,state))

    def test_invalid_input_and_feature_order_drift_rejected(self):
        with self.assertRaises(ValueError): fit_adapter(self.X.astype(np.float32),"lr",self.ids)
        with self.assertRaises(ValueError): fit_adapter(self.X,"unknown",self.ids)
        with self.assertRaises(ValueError): fit_adapter(np.array([[np.inf]],np.float64),"lr",("x",))
        state=fit_adapter(self.X,"lr",self.ids)
        changed=AdapterState(state.model_type,("x","b","c"),state.feature_ids_sha256,state.fit_indices_sha256,state.fit_scope,state.finite_counts,state.fill_values,state.all_missing)
        with self.assertRaises(ValueError): check_adapter(self.X,transform_adapter(self.X,state),changed)

    def test_lr_median_may_be_off_grid_without_rescaling(self):
        X=np.array([[0.0],[1.0/255.0]],np.float64)
        state=fit_adapter(X,"lr",("x",))
        self.assertEqual(state.fill_values[0],0.5/255.0)
        out=transform_adapter(np.array([[np.nan],[1.0/255.0]],np.float64),state)
        self.assertEqual(out[0,0],0.5/255.0); self.assertEqual(out[1,0],1.0/255.0)


if __name__=="__main__": unittest.main()
