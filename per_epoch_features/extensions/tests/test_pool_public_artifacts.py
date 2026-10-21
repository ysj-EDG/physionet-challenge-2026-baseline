from __future__ import annotations
import csv
import json
import unittest
from pathlib import Path
from extensions.pool_v2 import build_column_contract, build_view_manifest

class PoolPublicArtifactTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[3]

    def test_public_manifests_have_frozen_counts_and_safe_fields(self):
        with (self.root / 'audit/featurelab_v2_unique_manifest.csv').open(newline='', encoding='utf-8') as handle:
            unique = list(csv.DictReader(handle))
        with (self.root / 'audit/featurelab_v2_view_manifest.csv').open(newline='', encoding='utf-8') as handle:
            views = list(csv.DictReader(handle))
        with (self.root / 'audit/featurelab_v2_feature_qc.csv').open(newline='', encoding='utf-8') as handle:
            qc = list(csv.DictReader(handle))
        self.assertEqual(len(unique), 1444)
        self.assertEqual(len({row['feature_id'] for row in unique}), 1444)
        self.assertEqual(len(views), 1464)
        self.assertEqual(len(qc), 5776)
        self.assertEqual(set(qc[0]), {'scope','site_id','feature_id','family_id','modality','implementation_status','n_total','n_valid','n_missing','valid_rate','n_unique_valid','all_missing','constant_valid','zero_count','zero_rate','missing_reason_counts'})
        self.assertEqual({row['scope'] for row in qc}, {'Overall','I0002','I0006','S0001'})
        self.assertEqual({row['site_id'] for row in qc if row['scope']=='Overall'}, {''})
        self.assertEqual({row['site_id'] for row in qc if row['scope']!='Overall'}, {'I0002','I0006','S0001'})
        forbidden = ('/database/', 'source_sha256', 'record_key', 'sub-S')
        for path in (self.root / 'docs/featurelab/FEATURELAB_V2_INPUT_SPEC.md', self.root / 'docs/featurelab/FEATURELAB_V2_POOL_REPORT.md', self.root / 'audit/featurelab_v2_experiment_manifest.json'):
            text = path.read_text(encoding='utf-8')
            for token in forbidden:
                self.assertNotIn(token, text)

    def test_view_ids_indices_and_frozen_old_prefix_are_exact(self):
        with (self.root / 'audit/featurelab_v2_unique_manifest.csv').open(newline='', encoding='utf-8') as h:
            unique=list(csv.DictReader(h))
        with (self.root / 'audit/featurelab_v2_view_manifest.csv').open(newline='', encoding='utf-8') as h:
            rows=list(csv.DictReader(h))
        old_ids=tuple(row['feature_id'] for row in unique[:807])
        families={family:tuple(row['feature_id'] for row in unique[807:] if row['family_id']==family) for family in ('eeg_tail','eeg_temporal','eeg_coherence_stage','caisr_stage','emg_stage','resp_stage')}
        contract=build_column_contract(old_ids,families)
        old_rows=[{'view_id':r['view_id'],'view_position':r['view_position'],'unique_feature_id':r['feature_id']} for r in rows if r['source_block']=='old807']
        new_rows=[{'proposed_view':r['view_id'],'feature_id':r['feature_id'],'family_id':r['source_block']} for r in rows if r['source_block']!='old807']
        built=build_view_manifest(contract,old_rows,new_rows)
        flat=[row for view in ('V1','V2','V3','V4','V5') for row in built[view]]
        self.assertEqual([(r['view_id'],r['view_position'],r['feature_id'],r['unique_column_index']) for r in flat], [(r['view_id'],int(r['view_position']),r['feature_id'],int(r['unique_column_index'])) for r in rows])
        self.assertEqual([(r['view_id'],r['view_position'],r['unique_feature_id']) for r in old_rows], [(r['view_id'],r['view_position'],r['feature_id']) for r in rows if r['source_block']=='old807'])

    def test_experiment_manifest_has_no_private_paths(self):
        data = json.loads((self.root / 'audit/featurelab_v2_experiment_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(data['candidate_sets'], {'E0':807,'E1':1191,'E2':1060,'E3':1444})
        self.assertEqual(data['experiment_column_selection'], {'E0':[{'start':0,'stop':807}], 'E1':[{'start':0,'stop':1191}], 'E2':[{'start':0,'stop':807},{'start':1191,'stop':1444}], 'E3':[{'start':0,'stop':1444}]})
        self.assertEqual(data['protocols']['holdout_training_site'], {'I0002':'S0001','I0006':'S0001','S0001':'I0006'})
        self.assertTrue(data['protocols']['transductive'])
        self.assertTrue(data['protocols']['inductive_feature'])
        self.assertEqual(data['views'], {'V1':161,'V2':177,'V3':493,'V4':257,'V5':376})
        self.assertFalse(data['imputation'])
        self.assertFalse(data['scaling'])
        self.assertFalse(data['model_training'])
        self.assertFalse(data['loso_scoring'])

if __name__ == '__main__':
    unittest.main()
