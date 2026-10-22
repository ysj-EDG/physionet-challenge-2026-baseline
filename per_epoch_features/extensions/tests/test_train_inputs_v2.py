import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproduction.siren_a.pipeline.label_join import MetadataIndex, MetadataRecord
from extensions.rank256_v2 import publish_unit_atomic, _canonical, _sha_bytes
from extensions.train_inputs_v2 import (
    V2_VIEW_DIMS, V2FoldInputs, check_cross_protocol_train_v2, load_v2_fold,
)


def _manifest(path):
    rows=[]
    demo_ids=["demo_male","demo_bmi","demo_race_white","demo_race_black","demo_race_asian"]
    next_index=0
    for view,width in V2_VIEW_DIMS.items():
        for pos in range(width):
            if pos >= width-5:
                idx=156+(pos-(width-5)); feature_id=demo_ids[pos-(width-5)]
            else:
                idx=next_index; feature_id=f"{view}_f{idx}"; next_index += 1
            rows.append({"view_id":view,"view_position":str(pos),"feature_id":feature_id,"unique_column_index":str(idx),"source_block":"synthetic"})
        if view == "V1": next_index = 161
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    return rows


def _metadata(sites):
    records=tuple(MetadataRecord(str(site),f"b{i}",f"s{i}",str(i%2),str(40+i),f"p{i}",i) for i,site in enumerate(sites))
    return MetadataIndex(records=records,cache_names=tuple(f"cache_{i:06d}.npz" for i in range(len(records))),relative_paths=tuple(f"{s}/cache_{i:06d}.npz" for i,s in enumerate(sites)),sites=np.asarray(sites,dtype='U5'),demographics_sha256='demo',join_fingerprint='join',audit={'patient_id_available':True})


def _unit(root, manifest_path, holdout='I0002', protocol='transductive', *, bad_indices=False):
    sites=np.array(['S0001','I0002','S0001','I0006'])
    folds={'I0002':('S0001','I0006',2,1),'I0006':('S0001','I0002',2,1),'S0001':('I0006','I0002',1,2)}
    train_site,unused,train_n,test_n=folds[holdout]
    with manifest_path.open() as handle:
        rows=list(csv.DictReader(handle))
    manifest_hash=_sha_bytes(_canonical(rows).encode())
    payload={'train_indices':np.array([0,2] if holdout!='S0001' else [3],np.int64), 'test_indices':np.array([1] if holdout!='S0001' else [0,2],np.int64), 'unused_indices':np.array([3] if holdout!='S0001' else [1],np.int64)}
    if bad_indices: payload['train_indices']=np.array([2,0],np.int64) if holdout!='S0001' else np.array([3],np.int64)
    for view,width in V2_VIEW_DIMS.items():
        train_idx=payload['train_indices']; test_idx=payload['test_indices']
        base=np.linspace(0.0,0.9,width,dtype=np.float64)
        payload[f'train_{view}_values']=np.tile(base,(len(train_idx),1)) + train_idx[:,None]/1000.0
        payload[f'test_{view}_values']=np.tile(base,(len(test_idx),1)) + test_idx[:,None]/1000.0
        payload[f'train_{view}_rank_valid_mask']=np.ones((len(train_idx),width),bool)
        payload[f'test_{view}_rank_valid_mask']=np.ones((len(test_idx),width),bool)
    metadata={'schema':'featurelab_v2_rank256_v1','holdout_site':holdout,'train_site':train_site,'unused_site':unused,'protocol':protocol,'train_count':train_n,'test_count':test_n,'unused_count':len(payload['unused_indices']),'feature_count':1444,'feature_id_sha256':_sha_bytes(_canonical([next(r['feature_id'] for r in rows if int(r['unique_column_index']) == i) for i in range(1444)]).encode()),'pool_matrix_digest':'pool','formula':'(L+0.5E)/n','levels':256,'rounding':'half_up','numpy_version':np.__version__,'python_version':'synthetic','experiment_selection':{},'view_dimensions':V2_VIEW_DIMS,'view_manifest_sha256':manifest_hash,'source_valid_mask_field':'valid_mask','support_field':'support_count','missing_reason_field':'missing_reason_codes'}
    publish_unit_atomic(root/f'holdout_{holdout}'/protocol,payload,metadata)
    return sites,metadata


class TrainInputsV2Tests(unittest.TestCase):
    def test_train_targets_are_exposed_only_for_training_role(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); manifest=root/'views.csv'; _manifest(manifest); sites,meta=_unit(root,manifest)
            fold=load_v2_fold(root,'I0002','transductive','V1',_metadata(sites),manifest,{'source':'synthetic'},expected_folds={'I0002':('S0001','I0006',2,1)})
            self.assertIsInstance(fold,V2FoldInputs); self.assertTrue(hasattr(fold,'y_train')); self.assertTrue(hasattr(fold,'age_train'))
            self.assertFalse(hasattr(fold,'y_test')); self.assertFalse(hasattr(fold,'age_test')); self.assertEqual(fold.X_test.shape,(1,161))

    def test_view_rows_and_global_order_are_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); manifest=root/'views.csv'; _manifest(manifest); sites,_=_unit(root,manifest)
            fold=load_v2_fold(root,'I0002','transductive','V2',_metadata(sites),manifest,{'source':'synthetic'},expected_folds={'I0002':('S0001','I0006',2,1)})
            self.assertEqual(tuple(fold.feature_ids[:3]),('V2_f161','V2_f162','V2_f163')); np.testing.assert_allclose(fold.X_train[:,0],[0.,0.002]); np.testing.assert_array_equal(fold.train_indices,[0,2]); np.testing.assert_array_equal(fold.test_indices,[1])

    def test_manifest_or_indices_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); manifest=root/'views.csv'; _manifest(manifest); sites,_=_unit(root,manifest,bad_indices=True)
            with self.assertRaises(ValueError): load_v2_fold(root,'I0002','transductive','V1',_metadata(sites),manifest,{'source':'synthetic'},expected_folds={'I0002':('S0001','I0006',2,1)})
            with manifest.open() as handle:
                rows=list(csv.DictReader(handle))
            rows[0]['feature_id']='wrong'; bad=root/'bad.csv'
            with bad.open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
            root2=Path(td)/'other'; root2.mkdir(); _unit(root2,manifest)
            with self.assertRaises(ValueError): load_v2_fold(root2,'I0002','transductive','V1',_metadata(sites),bad,{'source':'synthetic'},expected_folds={'I0002':('S0001','I0006',2,1)})

    def test_cross_protocol_train_and_shared_site_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); manifest=root/'views.csv'; _manifest(manifest); sites,_=_unit(root,manifest,protocol='transductive'); _unit(root,manifest,protocol='inductive')
            metadata=_metadata(sites); kwargs={'expected_folds':{'I0002':('S0001','I0006',2,1)}}
            first=load_v2_fold(root,'I0002','transductive','V3',metadata,manifest,{'source':'synthetic'},**kwargs); second=load_v2_fold(root,'I0002','inductive','V3',metadata,manifest,{'source':'synthetic'},**kwargs)
            self.assertTrue(check_cross_protocol_train_v2(first,second)); self.assertEqual(first.train_site,'S0001'); self.assertEqual(first.train_indices.tolist(),[0,2])

if __name__=='__main__': unittest.main()
