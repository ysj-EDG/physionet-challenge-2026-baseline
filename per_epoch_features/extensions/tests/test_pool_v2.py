from __future__ import annotations
import unittest
import numpy as np
from extensions.pool_v2 import build_column_contract, build_view_manifest, assemble_arrays

class PoolV2Tests(unittest.TestCase):
    def test_column_contract_and_views_have_frozen_dimensions(self):
        old = tuple(f'old_{i}' for i in range(807))
        families = {
            'eeg_tail': tuple(f'tail_{i}' for i in range(288)),
            'eeg_temporal': tuple(f'temporal_{i}' for i in range(96)),
            'eeg_coherence_stage': tuple(f'coh_{i}' for i in range(90)),
            'caisr_stage': tuple(f'caisr_{i}' for i in range(35)),
            'emg_stage': tuple(f'emg_{i}' for i in range(72)),
            'resp_stage': tuple(f'resp_{i}' for i in range(56)),
        }
        contract = build_column_contract(old, families)
        self.assertEqual(len(contract['unique_feature_ids']), 1444)
        self.assertEqual(contract['experiment_sets'], {'E0': 807, 'E1': 1191, 'E2': 1060, 'E3': 1444})
        views = build_view_manifest(contract, old_view_rows=())
        self.assertEqual({k: len(v) for k, v in views.items()}, {'V1':161,'V2':177,'V3':493,'V4':257,'V5':376})

    def test_assemble_keeps_nan_and_masks_without_imputation(self):
        old_ids = tuple(f'old_{i}' for i in range(807))
        family_ids = {'eeg_tail': tuple(f'tail_{i}' for i in range(288)), 'eeg_temporal': tuple(f'temporal_{i}' for i in range(96)), 'eeg_coherence_stage': tuple(f'coh_{i}' for i in range(90)), 'caisr_stage': tuple(f'caisr_{i}' for i in range(35)), 'emg_stage': tuple(f'emg_{i}' for i in range(72)), 'resp_stage': tuple(f'resp_{i}' for i in range(56))}
        blocks = {}
        for family, ids in family_ids.items():
            blocks[family] = {'values': np.zeros((1, len(ids)), dtype=np.float64), 'valid_mask': np.ones((1, len(ids)), dtype=bool), 'support_count': np.ones((1, len(ids)), dtype=np.int64), 'missing_reason': np.full((1, len(ids)), '', dtype='<U1')}
        blocks['eeg_tail']['values'][0,1] = np.nan
        blocks['eeg_tail']['valid_mask'][0,1] = False
        blocks['eeg_tail']['support_count'][0,1] = 0
        blocks['eeg_tail']['missing_reason'][0,1] = 'LOW_SUPPORT'
        result = assemble_arrays(old_values=np.ones((1,807),dtype=np.float64), old_valid=np.ones((1,807),dtype=bool), old_support=np.ones((1,807),dtype=np.int64), family_blocks=blocks, old_feature_ids=old_ids, family_feature_ids=family_ids)
        self.assertEqual(result['values'][0,807], 0.0)
        self.assertTrue(np.isnan(result['values'][0,808]))
        self.assertFalse(result['valid_mask'][0,808])

if __name__ == '__main__': unittest.main()
