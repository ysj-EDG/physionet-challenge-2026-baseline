import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproduction.siren_a.pipeline.input_data import (
    EXPECTED_FOLDS,
    FoldInputs,
    check_cross_protocol_train,
    check_fold_inputs,
    get_view_feature_ids,
    load_fold,
)
from reproduction.siren_a.pipeline.label_join import MetadataIndex, MetadataRecord
from reproduction.siren_a.pipeline.run_inputs import (
    atomic_json_no_clobber,
    atomic_npz_no_clobber,
    build_parser,
    ensure_public_csv,
    publish_directory_no_clobber,
    verify_private_output,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_index():
    records=(
        MetadataRecord("I0002","sub-a","01","False","55","p1",0),
        MetadataRecord("I0006","sub-b","01","True","60","p2",1),
        MetadataRecord("S0001","sub-c","01","False","65","p3",2),
        MetadataRecord("S0001","sub-d","01","True","70","p4",3),
    )
    return MetadataIndex(records,tuple(f"cache_{i:06d}.npz" for i in range(4)),tuple(f"{r.site}/{r.record_stem}.npz" for r in records),np.array([r.site for r in records]),"csvsha","joinsha",{"matched":4,"patient_id_available":True})


def write_manifest(path, size=113):
    fields=["view_id","view_position","unique_feature_id","feature_name","family"]
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i in range(size):
            w.writerow({"view_id":"V1","view_position":i,"unique_feature_id":f"f{i}","feature_name":f"f{i}","family":"EEG"})


def write_unit(root, protocol, *, corrupt_hash=False):
    unit=Path(root)/"holdout_I0002"/protocol; unit.mkdir(parents=True)
    train=np.array([2,3],np.int64); test=np.array([0],np.int64); unused=np.array([1],np.int64)
    np.savez(unit/"indices.npz",train_indices=train,test_indices=test,unused_indices=unused)
    tv=np.tile(np.arange(113,dtype=np.float64)/255,(2,1)); xv=tv[:1].copy()
    tm=np.ones_like(tv,dtype=bool); xm=np.ones_like(xv,dtype=bool)
    tv[0,4]=np.nan; tm[0,4]=False
    np.savez(unit/"views.npz",train_V1_values=tv,train_V1_rank_valid_mask=tm,test_V1_values=xv,test_V1_rank_valid_mask=xm)
    metadata={"schema":"g3a_rank256_v1","holdout_site":"I0002","train_site":"S0001","unused_site":"I0006","protocol":protocol,"train_count":2,"test_count":1,"run_identity_sha256":"runsha"}
    (unit/"metadata.json").write_text(json.dumps(metadata),encoding="utf-8")
    files={name:sha(unit/name) for name in ("indices.npz","views.npz","metadata.json")}
    if corrupt_hash: files["views.npz"]="0"*64
    (unit/"complete.json").write_text(json.dumps({"schema":"g3a_rank256_v1","files":files}),encoding="utf-8")
    return unit


class FoldInputTests(unittest.TestCase):
    def test_fold_contract_and_no_test_targets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest=root/"manifest.csv"; write_manifest(manifest); write_unit(root/"g3","transductive")
            fold=load_fold(root/"g3","I0002","transductive","V1",metadata_index(),manifest,{"g3a":"abc"},expected_folds={"I0002":("S0001","I0006",2,1)})
            self.assertIsInstance(fold,FoldInputs); self.assertEqual(fold.X_train.shape,(2,113)); self.assertEqual(fold.X_test.shape,(1,113))
            self.assertEqual(fold.y_train.tolist(),[0,1]); self.assertEqual(fold.age_train.tolist(),[65.,70.]); self.assertEqual(fold.train_age_groups.tolist(),[13,14])
            self.assertFalse(hasattr(fold,"y_test")); self.assertFalse(hasattr(fold,"age_test")); self.assertFalse(hasattr(fold,"metadata"))
            self.assertTrue(check_fold_inputs(fold))

    def test_manifest_order_and_forbidden_model_columns(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"m.csv"; write_manifest(path)
            ids=get_view_feature_ids(path,"V1",expected_size=113)
            self.assertEqual(ids[0],"f0"); self.assertEqual(ids[-1],"f112")
            with path.open("a",encoding="utf-8") as f: f.write("V1,113,Age,Age,Demo\n")
            with self.assertRaises(ValueError): get_view_feature_ids(path,"V1",expected_size=114)

    def test_source_hash_drift_and_wrong_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest=root/"manifest.csv"; write_manifest(manifest)
            unit=write_unit(root/"g3","transductive",corrupt_hash=True)
            with self.assertRaises(ValueError): load_fold(root/"g3","I0002","transductive","V1",metadata_index(),manifest,{"g3a":"abc"},expected_folds={"I0002":("S0001","I0006",2,1)})
            complete=json.loads((unit/"complete.json").read_text()); complete["files"]["views.npz"]=sha(unit/"views.npz"); (unit/"complete.json").write_text(json.dumps(complete))
            np.savez(unit/"indices.npz",train_indices=np.array([3,2]),test_indices=np.array([0]),unused_indices=np.array([1]))
            complete["files"]["indices.npz"]=sha(unit/"indices.npz"); (unit/"complete.json").write_text(json.dumps(complete))
            with self.assertRaises(ValueError): load_fold(root/"g3","I0002","transductive","V1",metadata_index(),manifest,{"g3a":"abc"},expected_folds={"I0002":("S0001","I0006",2,1)})

    def test_protocol_train_exact_equality(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest=root/"manifest.csv"; write_manifest(manifest)
            write_unit(root/"g3","transductive"); write_unit(root/"g3","inductive")
            kwargs=dict(g3_root=root/"g3",holdout_site="I0002",view="V1",metadata=metadata_index(),manifest_path=manifest,source_fingerprints={"g3a":"abc"},expected_folds={"I0002":("S0001","I0006",2,1)})
            a=load_fold(protocol="transductive",**kwargs); b=load_fold(protocol="inductive",**kwargs)
            self.assertTrue(check_cross_protocol_train(a,b))
            changed=np.array(b.X_train,copy=True); changed[0,0]+=1
            bad=FoldInputs(changed,b.X_test,b.train_valid_mask,b.test_valid_mask,b.y_train,b.age_train,b.train_age_groups,b.feature_ids,b.train_indices,b.test_indices,b.holdout_site,b.train_site,b.protocol,b.view,b.source_fingerprints)
            with self.assertRaises(ValueError): check_cross_protocol_train(a,bad)

    def test_cli_modes_have_no_training_surface(self):
        parser=build_parser()
        for mode in ("--preflight-only","--smoke","--build","--verify-only"):
            parsed=parser.parse_args([mode]); self.assertTrue(getattr(parsed,mode[2:].replace("-","_")))
        help_text=parser.format_help().lower()
        for token in ("estimator","model-fit","epochs","learning-rate","seed-count"):
            self.assertNotIn(token,help_text)

    def test_no_clobber_completion_and_resume_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); building=root/"run.building"; target=root/"run"; building.mkdir()
            atomic_json_no_clobber(building/"payload.json",{"status":"PASS","source":"abc"})
            with self.assertRaises(FileExistsError): atomic_json_no_clobber(building/"payload.json",{"status":"changed"})
            payload_sha=sha(building/"payload.json")
            atomic_json_no_clobber(building/"complete.json",{"schema":"g3b_inputs_v1","files":{"payload.json":payload_sha},"identity":{"source":"abc"}})
            publish_directory_no_clobber(building,target)
            self.assertTrue(verify_private_output(target,{"source":"abc"}))
            with self.assertRaises(FileExistsError): publish_directory_no_clobber(root/"other",target)
            with self.assertRaises(ValueError): verify_private_output(target,{"source":"drift"})
            (target/"payload.json").write_text("{}")
            with self.assertRaises(ValueError): verify_private_output(target,{"source":"abc"})

    def test_atomic_npz_handles_string_provenance_and_nan_values(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"state.npz"
            atomic_npz_no_clobber(path,model_type=np.asarray("lr"),values=np.asarray([1.0,np.nan]))
            with np.load(path,allow_pickle=False) as z:
                self.assertEqual(str(z["model_type"].item()),"lr")
                np.testing.assert_array_equal(z["values"],np.asarray([1.0,np.nan]))
            with self.assertRaises(FileExistsError): atomic_npz_no_clobber(path,model_type=np.asarray("hgb"))

    def test_public_summary_resume_skips_only_exact_verified_content(self):
        rows=[{"holdout_site":"I0002","train_site":"S0001","protocol":"transductive","view":"V1","model_type":"lr","train_count":2,"test_count":1,"feature_count":113,"source_missing_count":1,"adapter_missing_count":0,"train_all_missing_columns":0,"train_positive_count":1,"train_negative_count":1,"age_group_count":2,"groups_with_both_classes":0,"metadata_join_status":"PASS","leakage_check_status":"PASS","adapter_status":"PASS"}]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"summary.csv"
            ensure_public_csv(path,rows,expected_count=1)
            original=path.read_bytes()
            ensure_public_csv(path,rows,expected_count=1)
            self.assertEqual(path.read_bytes(),original)
            changed=[dict(rows[0],train_count=3)]
            with self.assertRaises(ValueError): ensure_public_csv(path,changed,expected_count=1)


if __name__=="__main__": unittest.main()
