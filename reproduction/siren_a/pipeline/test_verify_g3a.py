import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from verify_g3a import (
    ManifestContract,
    audit_rank_independent,
    atomic_json_no_clobber,
    check_g3_identity,
    check_index_inventory,
    check_indices,
    check_unit_integrity,
    compare_audit_rows,
    compare_matrix,
    independent_views,
    inventory_fingerprint,
    load_manifest_contract,
    rebuild_loso,
    validate_public_payload,
    verify_private_output,
)
from verify_rank import build_reference, quantize_values, rank_values


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class IdentityAndIndexTests(unittest.TestCase):
    def setUp(self):
        self.inventory=[
            {"cache_name":"cache_000000.npz","relative_path":"I0002/a.npz","site":"I0002"},
            {"cache_name":"cache_000001.npz","relative_path":"I0002/b.npz","site":"I0002"},
            {"cache_name":"cache_000002.npz","relative_path":"I0006/c.npz","site":"I0006"},
            {"cache_name":"cache_000003.npz","relative_path":"S0001/d.npz","site":"S0001"},
            {"cache_name":"cache_000004.npz","relative_path":"S0001/e.npz","site":"S0001"},
        ]
        self.index=[dict(x) for x in self.inventory]
        self.counts={"I0002":2,"I0006":1,"S0001":2}

    def test_exact_index_inventory_mapping(self):
        sites=check_index_inventory(self.inventory,self.index,self.counts)
        np.testing.assert_array_equal(sites,np.array(["I0002","I0002","I0006","S0001","S0001"]))

    def test_same_site_swap_is_rejected(self):
        bad=[dict(x) for x in self.index]; bad[0],bad[1]=bad[1],bad[0]
        with self.assertRaises(ValueError): check_index_inventory(self.inventory,bad,self.counts)

    def test_site_cache_order_and_duplicates_are_rejected(self):
        variants=[]
        bad=[dict(x) for x in self.index]; bad[0]["site"]="I0006"; variants.append(bad)
        bad=[dict(x) for x in self.index]; bad[1]["cache_name"]="cache_000009.npz"; variants.append(bad)
        bad=[dict(x) for x in self.index]; bad[1]["relative_path"]=bad[0]["relative_path"]; variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError): check_index_inventory(self.inventory,bad,self.counts)

    def test_loso_arrays_are_exact_ordered_and_complete(self):
        sites=np.array(["I0002","I0002","I0006","S0001","S0001"])
        folds=rebuild_loso(sites)
        fold=folds["I0002"]
        np.testing.assert_array_equal(fold["train_indices"],np.array([3,4],np.int64))
        np.testing.assert_array_equal(fold["test_indices"],np.array([0,1],np.int64))
        np.testing.assert_array_equal(fold["unused_indices"],np.array([2],np.int64))
        arrays={k:fold[k] for k in ("train_indices","test_indices","unused_indices")}
        self.assertTrue(check_indices(arrays,fold,len(sites)))

    def test_same_site_index_swap_and_wrong_site_fail(self):
        sites=np.array(["I0002","I0002","I0006","S0001","S0001"])
        fold=rebuild_loso(sites)["I0002"]
        swapped={k:fold[k].copy() for k in ("train_indices","test_indices","unused_indices")}
        swapped["train_indices"]=swapped["train_indices"][::-1]
        with self.assertRaises(ValueError): check_indices(swapped,fold,len(sites))
        wrong={k:fold[k].copy() for k in ("train_indices","test_indices","unused_indices")}
        wrong["test_indices"][0]=2
        with self.assertRaises(ValueError): check_indices(wrong,fold,len(sites))

    def test_overlap_out_of_range_and_wrong_dtype_fail(self):
        sites=np.array(["I0002","I0002","I0006","S0001","S0001"])
        fold=rebuild_loso(sites)["I0002"]
        cases=[]
        bad={k:fold[k].copy() for k in ("train_indices","test_indices","unused_indices")}; bad["unused_indices"][0]=3; cases.append(bad)
        bad={k:fold[k].copy() for k in ("train_indices","test_indices","unused_indices")}; bad["test_indices"][0]=99; cases.append(bad)
        bad={k:fold[k].copy() for k in ("train_indices","test_indices","unused_indices")}; bad["train_indices"]=bad["train_indices"].astype(float); cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError): check_indices(bad,fold,len(sites))


class IntegrityAndMappingTests(unittest.TestCase):
    REQUIRED=("indices.npz","reference.npz","percentile.npz","quantized.npz","views.npz","rank_audit.csv","metadata.json")

    def make_unit(self,root):
        unit=Path(root)/"unit"; unit.mkdir()
        for name in self.REQUIRED:
            path=unit/name
            if name.endswith(".npz"):
                with path.open("wb") as f: np.savez_compressed(f,x=np.array([1.0]))
            elif name.endswith(".json"): path.write_text(json.dumps({"ok":True}))
            else: path.write_text("a,b\n1,2\n")
        complete={"schema":"g3a_rank256_v1","files":{name:sha(unit/name) for name in self.REQUIRED}}
        (unit/"complete.json").write_text(json.dumps(complete))
        return unit

    def test_complete_unit_integrity_and_rehashed_tamper_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit=self.make_unit(tmp)
            self.assertTrue(check_unit_integrity(unit))
            with (unit/"percentile.npz").open("wb") as f: np.savez_compressed(f,x=np.array([2.0]))
            complete=json.loads((unit/"complete.json").read_text())
            complete["files"]["percentile.npz"]=sha(unit/"percentile.npz")
            (unit/"complete.json").write_text(json.dumps(complete))
            self.assertTrue(check_unit_integrity(unit))
            with self.assertRaises(ValueError):
                compare_matrix(np.array([2.0]),np.array([1.0]),"tampered percentile")

    def test_unknown_file_incomplete_dir_corrupt_npz_and_symlink_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit=self.make_unit(tmp)
            (unit/"extra.txt").write_text("x")
            with self.assertRaises(ValueError): check_unit_integrity(unit)
        with tempfile.TemporaryDirectory() as tmp:
            unit=self.make_unit(tmp)
            (unit.with_name(unit.name+".building")).mkdir()
            with self.assertRaises(ValueError): check_unit_integrity(unit)
        with tempfile.TemporaryDirectory() as tmp:
            unit=self.make_unit(tmp)
            (unit/"reference.npz").write_bytes(b"broken")
            complete=json.loads((unit/"complete.json").read_text()); complete["files"]["reference.npz"]=sha(unit/"reference.npz")
            (unit/"complete.json").write_text(json.dumps(complete))
            with self.assertRaises(ValueError): check_unit_integrity(unit)
        with tempfile.TemporaryDirectory() as tmp:
            unit=self.make_unit(tmp)
            target=unit/"metadata.json"; target.unlink(); target.symlink_to(unit/"rank_audit.csv")
            with self.assertRaises(ValueError): check_unit_integrity(unit)

    def test_independent_views_map_values_masks_and_repeated_features(self):
        contract=ManifestContract(
            rows=(
                {"view_id":"V1","view_position":0,"unique_feature_id":"a","family":"Demo5"},
                {"view_id":"V1","view_position":1,"unique_feature_id":"b","family":"EEG"},
                {"view_id":"V2","view_position":0,"unique_feature_id":"a","family":"Demo5"},
            ),
            unique_ids=("a","b"), families=("Demo5","EEG"),
            view_positions={"V1":np.array([0,1]),"V2":np.array([0])},
            view_sizes={"V1":2,"V2":1}, sha256="x",
        )
        values=np.array([[1.0,np.nan],[0.0,3.0]])
        masks=np.array([[1,0],[1,1]],bool)
        vv,vm=independent_views(values,masks,contract)
        np.testing.assert_array_equal(vv["V1"],values)
        np.testing.assert_array_equal(vm["V1"],masks)
        np.testing.assert_array_equal(vv["V1"][:,0],vv["V2"][:,0])
        np.testing.assert_array_equal(vm["V1"][:,0],vm["V2"][:,0])

    def test_matrix_comparison_checks_nan_mask_and_single_value(self):
        expected=np.array([[1.0,np.nan]])
        self.assertTrue(compare_matrix(expected,expected.copy(),"ok"))
        for bad in (np.array([[1.0,0.0]]),np.array([[1.0001,np.nan]])):
            with self.assertRaises(ValueError): compare_matrix(bad,expected,"bad")


class AuditAndRecoveryTests(unittest.TestCase):
    def test_audit_comparison_rejects_wrong_count_float_and_duplicate(self):
        expected=[{"feature_id":"a","n_total":2,"max_abs_quantization_error":0.1}]
        self.assertTrue(compare_audit_rows(expected,[dict(expected[0])],("feature_id",)))
        for observed in (
            [{"feature_id":"a","n_total":3,"max_abs_quantization_error":0.1}],
            [{"feature_id":"a","n_total":2,"max_abs_quantization_error":0.1001}],
            [dict(expected[0]),dict(expected[0])],
        ):
            with self.assertRaises(ValueError): compare_audit_rows(expected,observed,("feature_id",))

    def test_atomic_json_no_clobber_and_private_output_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/"summary.json"
            atomic_json_no_clobber(path,{"status":"PASS"})
            with self.assertRaises(FileExistsError): atomic_json_no_clobber(path,{"status":"OTHER"})
            files={"summary.json":sha(path)}
            atomic_json_no_clobber(root/"complete.json",{"schema":"g3av_verification_v1","files":files})
            self.assertTrue(verify_private_output(root))
            path.write_text('{"status":"tampered"}')
            with self.assertRaises(ValueError): verify_private_output(root)

    def test_interrupted_private_output_and_identity_drift_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/".building").mkdir()
            with self.assertRaises(ValueError): verify_private_output(root)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"summary.json").write_text('{"source":"wrong"}')
            (root/"complete.json").write_text(json.dumps({"schema":"g3av_verification_v1","files":{"summary.json":sha(root/"summary.json")}}))
            with self.assertRaises(ValueError): verify_private_output(root,expected_identity={"source":"expected"})

    def test_public_payload_rejects_private_identity(self):
        self.assertTrue(validate_public_payload([{"holdout_site":"I0002","status":"PASS"}]))
        for token in ("/database/private","cache_000001.npz","BidsFolder","SessionID","record_id","patient_id"):
            with self.assertRaises(ValueError): validate_public_payload([{"note":token}])


class FullVerifierContractTests(unittest.TestCase):
    MANIFEST_SHA="7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a"

    def test_real_manifest_contract_is_827_807_and_ordered(self):
        manifest=load_manifest_contract(HERE.parent/"docs"/"G1B_FEATURE_MANIFEST_FROZEN.csv",self.MANIFEST_SHA)
        self.assertEqual(len(manifest.rows),827)
        self.assertEqual(len(manifest.unique_ids),807)
        self.assertEqual(manifest.view_sizes,{"V1":113,"V2":129,"V3":349,"V4":113,"V5":123})
        for view,size in manifest.view_sizes.items():
            self.assertEqual(manifest.view_positions[view].shape,(size,))

    def test_inventory_fingerprint_is_order_sensitive(self):
        rows=[
            {"cache_name":"cache_000000.npz","relative_path":"I0002/a.npz","site":"I0002"},
            {"cache_name":"cache_000001.npz","relative_path":"I0002/b.npz","site":"I0002"},
        ]
        first=inventory_fingerprint(rows)
        self.assertEqual(first,inventory_fingerprint([dict(x) for x in rows]))
        self.assertNotEqual(first,inventory_fingerprint(rows[::-1]))

    def test_g3_identity_checks_source_and_recorded_code_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); files={}
            identity={
                "schema":"g3a_rank256_v1","g2b1_cache_fingerprint":"cache",
                "g2b1_inventory_fingerprint":"inventory","manifest_sha256":"manifest",
                "spec_sha256":"spec","formula":"(L+0.5E)/n","levels":256,"rounding":"half_up",
            }
            for name in ("data","split","normalize","views","audit","run_prep"):
                path=root/f"{name}.py"; path.write_text(name,encoding="utf-8")
                files[name+"_sha256"]=path
                identity[name+"_sha256"]=sha(path)
            self.assertTrue(check_g3_identity(identity,files,"cache","inventory","manifest","spec"))
            bad=dict(identity); bad["data_sha256"]="0"*64
            with self.assertRaises(ValueError):
                check_g3_identity(bad,files,"cache","inventory","manifest","spec")

    def test_independent_audit_recomputes_counts_bins_ties_and_overflow(self):
        raw=np.array([[0.0],[1.0],[2.0],[np.nan]])
        source=np.array([[1],[1],[1],[0]],bool)
        reference=build_reference(np.array([[0.0],[1.0]]),np.ones((2,1),bool))
        percentile,rank=rank_values(raw,source,reference)
        quantized,levels=quantize_values(percentile,rank)
        rows=audit_rank_independent(
            raw,percentile,quantized,source,rank,levels,
            ("f",),("Demo5",),reference.counts,
            np.array([0.0]),np.array([1.0]),
        )
        row=rows[0]
        self.assertEqual(row["n_total"],4)
        self.assertEqual(row["n_source_valid"],3)
        self.assertEqual(row["n_rank_valid"],3)
        self.assertEqual(row["reference_n"],2)
        self.assertEqual(row["occupied_bins"],3)
        self.assertEqual(row["above_reference_count"],1)
        self.assertEqual(row["lawful_zero_count"],1)


if __name__=="__main__":
    unittest.main()
