from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
import numpy as np
from extensions.pool_v2 import (
    build_column_contract, build_view_manifest, assemble_arrays,
    encode_reason_matrix, decode_reason_codes, write_candidate_pool_atomic,
    PoolPublishError,
)

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
        old_rows=[]; offsets={'V1':0,'V2':113,'V3':242,'V4':591,'V5':704}
        for view,width in {'V1':113,'V2':129,'V3':349,'V4':113,'V5':123}.items():
            for pos in range(width):
                idx=(offsets[view]+pos) % 807
                old_rows.append({'view_id':view,'view_position':pos,'unique_feature_id':old[idx]})
        chunks={'eeg_tail':[('V1',48),('V2',48),('V3',144),('V4',48)],
                'eeg_temporal':[('V4',96)],'eeg_coherence_stage':[('V5',90)],
                'caisr_stage':[('V5',35)],'emg_stage':[('V5',72)],'resp_stage':[('V5',56)]}
        new_rows=[]; cursor=807
        for family,parts in chunks.items():
            pos=0
            for view,count in parts:
                for fid in families[family][pos:pos+count]:
                    new_rows.append({'proposed_view':view,'feature_id':fid,'family_id':family})
                    cursor+=1; pos+=1
        views = build_view_manifest(contract, old_view_rows=old_rows, new_manifest_rows=new_rows)
        self.assertEqual({k: len(v) for k,v in views.items()}, {'V1':161,'V2':177,'V3':493,'V4':257,'V5':376})

    def test_reason_codes_are_code_to_reason_and_preserve_zero(self):
        valid = np.array([[False, True, False], [False, False, True]])
        reasons = np.array([['OLD_SOURCE_MISSING', '', 'LOW_SUPPORT'],
                            ['CHANNEL_UNAVAILABLE', 'LOW_SUPPORT', '']], dtype='<U32')
        codes, dictionary = encode_reason_matrix(valid, reasons)
        self.assertEqual(dictionary, {0:'VALID', 1:'OLD_SOURCE_MISSING',
                                      2:'CHANNEL_UNAVAILABLE', 3:'LOW_SUPPORT'})
        self.assertEqual(codes.tolist(), [[1,0,3],[2,3,0]])
        np.testing.assert_array_equal(decode_reason_codes(codes, valid, dictionary), reasons)

    def test_reason_codes_reject_unknown_empty_and_conflicting_dictionary(self):
        valid = np.array([[False]])
        with self.assertRaises(ValueError):
            encode_reason_matrix(valid, np.array([['']], dtype='<U1'))
        with self.assertRaises(ValueError):
            decode_reason_codes(np.array([[99]]), valid, {0:'VALID', 1:'OLD_SOURCE_MISSING'})
        with self.assertRaises(ValueError):
            decode_reason_codes(np.array([[1]]), valid, {0:'VALID', 1:'VALID'})

    def test_missing_reason_sets_are_record_independent(self):
        valid = np.array([[False, True], [False, False]])
        reasons = np.array([['A', ''], ['B', 'C']], dtype='<U1')
        codes, dictionary = encode_reason_matrix(valid, reasons)
        self.assertEqual(dictionary, {0:'VALID', 1:'OLD_SOURCE_MISSING', 2:'A', 3:'B', 4:'C'})
        self.assertEqual(codes.tolist(), [[2, 0], [3, 4]])

    def test_manifest_is_required_in_production(self):
        contract = build_column_contract(tuple(f'o{i}' for i in range(807)),
            {f: tuple(f'{f}_{i}' for i in range(n)) for f,n in
             zip(('eeg_tail','eeg_temporal','eeg_coherence_stage','caisr_stage','emg_stage','resp_stage'), (288,96,90,35,72,56))})
        with self.assertRaises(ValueError):
            build_view_manifest(contract, old_view_rows=[], new_manifest_rows=[])

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
        old_values=np.ones((1,807),dtype=np.float64); old_valid=np.ones((1,807),dtype=bool); old_support=np.ones((1,807),dtype=np.int64)
        old_values[0,0]=np.nan; old_valid[0,0]=False; old_support[0,0]=0
        old_values[0,1]=0.0; old_valid[0,1]=True
        result = assemble_arrays(old_values=old_values, old_valid=old_valid, old_support=old_support, family_blocks=blocks, old_feature_ids=old_ids, family_feature_ids=family_ids)
        self.assertEqual(result['missing_reason_codes'][0,0], 1)
        self.assertEqual(result['missing_reason_dictionary'][1], 'OLD_SOURCE_MISSING')
        self.assertEqual(result['values'][0,1], 0.0)
        self.assertEqual(result['missing_reason_codes'][0,1], 0)
        self.assertEqual(result['values'][0,807], 0.0)
        self.assertTrue(np.isnan(result['values'][0,808]))
        self.assertFalse(result['valid_mask'][0,808])

    def test_pool_publish_is_no_clobber_and_reloads_exactly(self):
        arrays = {
            'values': np.array([[0.0, np.nan]], dtype=np.float64),
            'valid_mask': np.array([[True, False]], dtype=bool),
            'support_count': np.array([[2, 0]], dtype=np.int64),
            'missing_reason_codes': np.array([[0, 1]], dtype=np.int64),
            'missing_reason_dictionary': {0:'VALID', 1:'OLD_SOURCE_MISSING'},
            'unique_feature_ids': ('f0','f1'),
            'row_keys': ('r0',), 'groups': ('I0002',),
            'view_rows': [{'view_id':'V1','view_position':0,'feature_id':'f0',
                          'unique_column_index':0,'source_block':'old807'}],
        }
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'pool'
            identity = {'source_schema':'synthetic-v1','source_digest':'abc',
                        'recipe_ids':{'synthetic':'r1'}}
            first = write_candidate_pool_atomic(arrays, target, identity)
            self.assertEqual(first['status'], 'BUILD_NEW')
            second = write_candidate_pool_atomic(arrays, target, identity)
            self.assertEqual(second['status'], 'SKIP_VERIFIED')
            before = (target/'X_raw.npy').read_bytes()
            changed = dict(identity); changed['source_digest'] = 'different'
            with self.assertRaises(PoolPublishError):
                write_candidate_pool_atomic(arrays, target, changed)
            self.assertEqual(before, (target/'X_raw.npy').read_bytes())
            incomplete = Path(td) / 'incomplete'
            incomplete.mkdir(); (incomplete/'X_raw.npy').write_bytes(b'not-a-pool')
            with self.assertRaises(PoolPublishError):
                write_candidate_pool_atomic(arrays, incomplete, identity)
            self.assertEqual((incomplete/'X_raw.npy').read_bytes(), b'not-a-pool')

if __name__ == '__main__': unittest.main()
