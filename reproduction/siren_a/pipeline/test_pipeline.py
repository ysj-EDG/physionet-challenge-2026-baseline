import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "night_cache"))

from data import FeatureData, load_features, validate_coverage
from split import make_loso, select_train_site
from views import get_views
from run_prep import (UnitAction, build_parser, decide_unit_action,
                      publish_unit_atomic, validate_unit)
from g2a_io import load_manifest

MANIFEST = load_manifest(ROOT / "docs" / "G1B_FEATURE_MANIFEST_FROZEN.csv")


class PipelineTests(unittest.TestCase):
    def test_loso_is_complete_disjoint_and_selects_largest_site(self):
        sites = np.array(["I0002"] * 2 + ["I0006"] * 3 + ["S0001"] * 5)
        folds = make_loso(sites)
        self.assertEqual(set(folds), {"I0002", "I0006", "S0001"})
        self.assertEqual(select_train_site("I0002", sites), ("S0001", "I0006"))
        self.assertEqual(select_train_site("I0006", sites), ("S0001", "I0002"))
        self.assertEqual(select_train_site("S0001", sites), ("I0006", "I0002"))
        for holdout, fold in folds.items():
            self.assertFalse(set(fold.train_indices) & set(fold.test_indices))
            self.assertTrue(np.all(sites[fold.test_indices] == holdout))
            self.assertEqual(len(np.unique(fold.test_indices)), len(fold.test_indices))

    def test_views_use_frozen_positions_and_repeat_demo_identically(self):
        values = np.arange(2 * 807, dtype=float).reshape(2, 807)
        masks = np.ones_like(values, bool)
        vv, vm = get_views(values, masks, MANIFEST)
        self.assertEqual({k: v.shape for k, v in vv.items()}, {"V1":(2,113),"V2":(2,129),"V3":(2,349),"V4":(2,113),"V5":(2,123)})
        for uid in ("demo_male", "demo_bmi", "demo_race_white", "demo_race_black", "demo_race_asian"):
            positions=[]
            for view in vv:
                rows=[r for r in MANIFEST.rows if r["view_id"]==view]
                for pos,row in enumerate(rows):
                    if row["unique_feature_id"]==uid: positions.append(vv[view][:,pos])
            for item in positions[1:]: np.testing.assert_array_equal(item, positions[0])

    def test_public_coverage_closure_detects_mismatch(self):
        data = FeatureData(np.array([[1., np.nan],[2.,3.]]), np.array([[1,0],[1,1]],bool),
                           ["a","b"], ["cache_0","cache_1"], np.array(["A","A"]), "fp", {})
        rows=[{"site":"A","unique_feature_id":"a","valid_count":"2","missing_count":"0"},
              {"site":"A","unique_feature_id":"b","valid_count":"1","missing_count":"1"}]
        self.assertTrue(validate_coverage(data, rows))
        rows[1]["valid_count"]="2"
        with self.assertRaises(ValueError): validate_coverage(data, rows)

    def test_real_g2b1_load_and_coverage_alignment(self):
        root=Path('/database/home/gaohaojie/workspace/python-example-2026-siren-a-g2b1-private/night_cache_v1')
        coverage=ROOT/'audit'/'g2b1_feature_coverage_aggregate.csv'
        if not root.is_dir(): self.skipTest('private G2-B1 cache unavailable')
        data=load_features(root, MANIFEST)
        self.assertEqual(data.values.shape,(6600,807)); self.assertEqual(data.source_valid_mask.shape,(6600,807))
        self.assertEqual(dict(zip(*np.unique(data.sites,return_counts=True))), {'I0002':319,'I0006':1142,'S0001':5139})
        with coverage.open() as f: rows=[r for r in csv.DictReader(f) if r['site']!='Overall']
        self.assertTrue(validate_coverage(data,rows))

    def test_unit_no_clobber_resume_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); final=root/'unit'; payload={'a':np.array([1.],np.float64),'tag':np.asarray('x')}; metadata={'fingerprint':'x'}
            publish_unit_atomic(final,payload,metadata)
            self.assertEqual(decide_unit_action(final,metadata),UnitAction.SKIP_VERIFIED)
            self.assertTrue(validate_unit(final,metadata).valid)
            self.assertEqual(decide_unit_action(final,{'fingerprint':'y'}),UnitAction.STOP_INVALID)
            with self.assertRaises(FileExistsError): publish_unit_atomic(final,payload,metadata)
            incomplete=root/'other.building'; incomplete.mkdir()
            self.assertEqual(decide_unit_action(root/'other',metadata),UnitAction.STOP_INVALID)

    def test_cli_has_only_explicit_preprocessing_modes(self):
        parser=build_parser()
        with self.assertRaises(SystemExit): parser.parse_args([])
        options={o for a in parser._actions for o in a.option_strings}
        self.assertTrue({'--preflight-only','--smoke','--build','--verify-only'}.issubset(options))
        self.assertFalse(options & {'--train','--labels','--age','--diagnosis'})


if __name__ == "__main__": unittest.main(verbosity=2)
