from __future__ import annotations
import csv
import json
import unittest
from pathlib import Path

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
        self.assertEqual(len(qc), 1444)
        self.assertEqual(set(qc[0]), {'unique_column_index','feature_id','family_id','modality','implementation_status','valid_count','missing_count','zero_count','constant_valid'})
        forbidden = ('/database/', 'source_sha256', 'record_key', 'sub-S')
        for path in (self.root / 'docs/featurelab/FEATURELAB_V2_INPUT_SPEC.md', self.root / 'docs/featurelab/FEATURELAB_V2_POOL_REPORT.md', self.root / 'audit/featurelab_v2_experiment_manifest.json'):
            text = path.read_text(encoding='utf-8')
            for token in forbidden:
                self.assertNotIn(token, text)

    def test_experiment_manifest_has_no_private_paths(self):
        data = json.loads((self.root / 'audit/featurelab_v2_experiment_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(data['candidate_sets'], {'E0':807,'E1':1191,'E2':1060,'E3':1444})
        self.assertEqual(data['views'], {'V1':161,'V2':177,'V3':493,'V4':257,'V5':376})
        self.assertFalse(data['imputation'])
        self.assertFalse(data['scaling'])
        self.assertFalse(data['model_training'])
        self.assertFalse(data['loso_scoring'])

if __name__ == '__main__':
    unittest.main()
