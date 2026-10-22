import unittest
import numpy as np

from extensions.train_inputs_v2 import V2FoldInputs
from extensions.train_seed42_v2 import (
    prepare_seed42_training, compare_training_protocols_v2, fit_seed42_adapter_v2,
)
from reproduction.siren_a.g4.sampler import sample_training_rows
from reproduction.siren_a.g4.queries import build_queries
from reproduction.siren_a.pipeline.adapters import check_adapter


def _fold(protocol="transductive"):
    n=20; width=161
    ages=np.array([40+5*(i//4)+0.2*(i%4) for i in range(n)], dtype=np.float64)
    y=np.array([0,1,0,1]*5, dtype=np.int8)
    X=np.tile(np.linspace(0.0,0.9,width,dtype=np.float64),(n,1))
    X[1,3]=np.nan; X[7,11]=np.nan
    mask=np.isfinite(X)
    return V2FoldInputs(X_train=X, X_test=np.zeros((2,width),dtype=np.float64),
        train_valid_mask=mask, test_valid_mask=np.ones((2,width),bool),
        y_train=y, age_train=ages, train_age_groups=np.floor(ages/5).astype(np.int64),
        feature_ids=tuple(f"f{i}" for i in range(width)), train_indices=np.arange(n,dtype=np.int64),
        test_indices=np.array([20,21],dtype=np.int64), holdout_site="I0002", train_site="S0001",
        protocol=protocol, view="V1", source_fingerprints={"source":"synthetic"})


class Seed42TrainingV2Tests(unittest.TestCase):
    def test_sampler_and_query_match_frozen_functions(self):
        fold=_fold(); prepared=prepare_seed42_training(fold)
        expected=sample_training_rows("S0001",42,fold.train_age_groups,fold.y_train,fold.train_indices)
        np.testing.assert_array_equal(prepared.sample.local_positions, expected.local_positions)
        np.testing.assert_array_equal(prepared.sample.global_indices, expected.global_indices)
        ages=fold.age_train[expected.local_positions]; labels=fold.y_train[expected.local_positions]
        query=build_queries(ages,labels,expected.local_positions,expected.global_indices)
        self.assertEqual(prepared.query.fingerprint, query.fingerprint)
        np.testing.assert_array_equal(prepared.query.permutation, query.permutation)

    def test_protocol_training_inputs_are_equal(self):
        first=prepare_seed42_training(_fold("transductive")); second=prepare_seed42_training(_fold("inductive"))
        self.assertTrue(compare_training_protocols_v2(first, second))

    def test_adapter_preserves_native_nan_and_checks_lr_fill(self):
        fold=_fold(); prepared=prepare_seed42_training(fold)
        for model_type in ("lr", "hgb", "lambdarank"):
            state, transformed = fit_seed42_adapter_v2(fold, prepared, model_type)
            source=fold.X_train[prepared.sample.local_positions]
            self.assertTrue(check_adapter(source, transformed, state))
            if model_type == "lr": self.assertFalse(np.isnan(transformed).any())
            else: self.assertEqual(np.isnan(transformed).sum(), np.isnan(source).sum())

if __name__ == "__main__": unittest.main()
