import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
ROOT=HERE.parent
from g2a_io import load_npz_record, write_cache_atomic, read_cache
from g2a_smoke import deterministic_base_sample, build_parser, MAX_OUTPUT_RECORDS, _atomic_json


class G2ASmokeTests(unittest.TestCase):
    def payload(self,t=5,w=5):
        return dict(
            X_seq=np.zeros((t,483)),X_ecg=np.zeros((w,12)),x_static=np.zeros(196),
            extraction_version=np.asarray("timegrid_v2.1.0"),validity_schema_version=np.asarray("validity_v0.2"),
            stage_code_aligned=np.full(t,2.),stage_valid=np.ones(t,bool),eeg_channel_available=np.ones(6,bool),
            eeg_clean_subsegment_count=np.ones((t,6),int),emg_channel_available=np.ones(3,bool),emg_epoch_success=np.ones((t,3),bool),
            resp_feature_valid=np.ones((t,14),bool),hrv_success=np.ones(w,bool),hrv_feature_valid=np.ones((w,11),bool),
            ecg_alignment_valid=np.ones(t,bool),arousal_valid=np.ones(t,bool),resp_event_valid=np.ones(t,bool),limb_event_valid=np.ones(t,bool),
            annotation_status=np.asarray("available"),stage_probability=np.ones((t,5))/5,stage_probability_valid=np.ones(t,bool),
            stage_probability_channels_present=np.ones(5,bool),annotation_signal_lengths_json=np.asarray('{"caisr_prob_n1":5,"caisr_prob_n2":5,"caisr_prob_n3":5,"caisr_prob_r":5,"caisr_prob_w":5}'))

    def test_loader_schema_versions_shapes_missing_and_no_label_requirement(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"r.npz"; np.savez(p,**self.payload())
            r=load_npz_record(p); self.assertNotIn("y",r); self.assertEqual(r["X_seq"].shape,(5,483))
            bad=self.payload(); bad["extraction_version"]=np.asarray("wrong"); np.savez(Path(d)/"bad.npz",**bad)
            with self.assertRaises(ValueError): load_npz_record(Path(d)/"bad.npz")
            bad=self.payload(); del bad["stage_valid"]; np.savez(Path(d)/"missing.npz",**bad)
            with self.assertRaises(ValueError): load_npz_record(Path(d)/"missing.npz")

    def test_atomic_cache_roundtrip_and_no_overwrite(self):
        arrays={"values":np.array([1.,np.nan]),"valid":np.array([1,0],bool),"support":np.array([1,0])}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"cache.npz"; write_cache_atomic(p,arrays)
            got=read_cache(p); np.testing.assert_equal(got["values"],arrays["values"])
            with self.assertRaises(FileExistsError): write_cache_atomic(p,arrays)

    def test_atomic_json_accepts_numpy_aggregate_scalars(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"aggregate.json"
            _atomic_json(p,{"flag":np.bool_(True),"count":np.int64(3),"ratio":np.float64(0.5)})
            self.assertIn('"flag": true',p.read_text())

    def test_deterministic_selector_six_per_site_unique_and_capped(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for site in ("I0002","I0006","S0001"):
                (root/site).mkdir(); [(root/site/f"r{i:03d}.npz").touch() for i in range(20)]
            a=deterministic_base_sample(root,20261013); b=deterministic_base_sample(root,20261013)
            self.assertEqual(a,b); self.assertEqual(len(a),18); self.assertEqual(len(set(a)),18); self.assertLessEqual(len(a),MAX_OUTPUT_RECORDS)
            self.assertEqual({s:sum(p.parent.name==s for p in a) for s in ("I0002","I0006","S0001")},{"I0002":6,"I0006":6,"S0001":6})

    def test_cli_has_no_full_mode(self):
        parser=build_parser(); options={x for a in parser._actions for x in a.option_strings}
        self.assertTrue({"--input-root","--output-root","--manifest","--spec","--report"}.issubset(options))
        self.assertFalse(options & {"--full","--all","--batch-6600"})
        source=(HERE/"g2a_smoke.py").read_text()
        self.assertIn('output_root.name+".building"',source)
        self.assertIn("os.replace(staging,args.output_root)",source)


    def test_public_report_privacy_and_no_forbidden_processing_path(self):
        report=(ROOT/"docs"/"G2A_SMOKE_REPORT.md").read_text()
        for token in ("/database/",".npz","BidsFolder","SessionID","BDSPPatientID"):
            self.assertNotIn(token,report)
        source="\n".join((HERE/name).read_text() for name in ("g2a_io.py","g2a_features.py","g2a_smoke.py"))
        for token in ("dropna(","standardize(","percentile_transform(","train_model(",'["y"]',"['y']"):
            self.assertNotIn(token,source)
        self.assertIn("ENTROPY_COVERAGE_ALERT",report)
        self.assertIn("G2B_FULL_CACHE: **NOT GENERATED**",report)


if __name__ == "__main__": unittest.main(verbosity=2)
