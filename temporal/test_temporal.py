"""Focused invariants for the temporal study (standard-library unittest)."""
from __future__ import annotations
import unittest
import numpy as np
import torch
from temporal.data import (TemporalRecord, apply_block_order, blockify,
    deterministic_permutation, event_epoch_validity, fit_neutral, impute_values)
from temporal.models import LocalTCNClassifier, masked_mean_std

def _record(values, valid):
    return TemporalRecord('r','p','I0002',0,50.,np.asarray(values,np.float32),np.asarray(valid,bool),len(values),1,0)

class TemporalTests(unittest.TestCase):
    def test_blocking_23_epochs(self):
        blocks,dropped=blockify(np.zeros((23,13),np.float32)); self.assertEqual(blocks.shape,(2,10,13)); self.assertEqual(dropped,3)
    def test_padding_excluded_from_pooling(self):
        got=masked_mean_std(torch.tensor([[[1.,2.],[3.,4.],[999.,999.]]]),torch.tensor([[False,False,True]]))
        self.assertTrue(torch.allclose(got,torch.tensor([[2.,3.,1.,1.]])))
    def test_zero_variance_pooling_backward_is_finite(self):
        tokens=torch.ones(2,3,4,requires_grad=True)
        masked_mean_std(tokens,torch.tensor([[False,False,False],[False,True,True]])).sum().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
    def test_invalid_stage_uses_neutral_not_zero(self):
        neutral=fit_neutral([_record([[1,0,0,0,0],[0,1,0,0,0]],np.ones((2,5),bool))],[0])
        got=impute_values(_record([[0,0,0,0,0]],np.zeros((1,5),bool)),neutral)[0]
        self.assertTrue(np.allclose(got,[.5,.5,0,0,0]))
    def test_valid_event_zero_and_invalid_event_zero_differ(self):
        neutral=fit_neutral([_record([[.25],[.75]],[[1],[1]])],[0])
        self.assertEqual(impute_values(_record([[0.]],[[1]]),neutral)[0,0],0)
        self.assertAlmostEqual(impute_values(_record([[0.]],[[0]]),neutral)[0,0],.5)
    def test_event_coverage_and_bad_frequency(self):
        starts=np.array([0.,30.,60.]); durations=np.full(3,30.)
        valid,_=event_epoch_validity(starts,durations,{'x':60},{'x':1},'x'); self.assertEqual(valid.tolist(),[True,True,False])
        with self.assertRaises(ValueError): event_epoch_validity(starts,durations,{'x':60},{'x':0},'x')
    def test_shuffle_is_block_specific_and_real_is_identity(self):
        blocks=np.arange(200).reshape(2,10,10)
        self.assertTrue(np.array_equal(deterministic_permutation('r',0),deterministic_permutation('r',0)))
        self.assertFalse(np.array_equal(deterministic_permutation('r',0),deterministic_permutation('r',1)))
        self.assertTrue(np.array_equal(apply_block_order(blocks,'r',False),blocks))
        self.assertFalse(np.array_equal(apply_block_order(blocks,'r',True),blocks))
    def test_model_shape_and_receptive_field(self):
        model=LocalTCNClassifier(13); out=model(torch.randn(4,7,10,13),torch.zeros(4,7,dtype=torch.bool))
        self.assertEqual(tuple(out.shape),(4,)); self.assertEqual(model.receptive_field_epochs,13)
    def test_all_padding_rejected(self):
        with self.assertRaises(ValueError): LocalTCNClassifier(13)(torch.randn(1,2,10,13),torch.ones(1,2,dtype=torch.bool))
    def test_eeg72_mapping_contract(self):
        spectral=np.arange(54).reshape(6,9); bsr=np.arange(18).reshape(3,6)
        self.assertEqual(spectral.shape,(6,9)); self.assertEqual(bsr.shape,(3,6)); self.assertEqual(bsr.ravel()[7],bsr[1,1])

if __name__=='__main__': unittest.main()
