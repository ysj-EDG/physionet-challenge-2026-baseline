import csv
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from g2a_io import EXPECTED_MANIFEST_SHA256, load_manifest
from g2a_features import (
    compute_demo5, compute_night_features, entropy_source_valid,
    pool_values, spatial_value_and_mask,
)

MANIFEST = ROOT / "docs" / "G1B_FEATURE_MANIFEST_FROZEN.csv"


def record(t=5, w=5):
    xseq = np.zeros((t, 483), dtype=float)
    xecg = np.zeros((w, 12), dtype=float)
    static = np.zeros(196, dtype=float)
    static[2] = 1; static[8] = 1; static[9] = 25
    static[10:] = np.arange(186) + 1
    stages = np.full(t, 2.0)
    probability = np.tile([0, 1, 0, 0, 0], (t, 1)).astype(float)
    return {
        "X_seq": xseq, "X_ecg": xecg, "x_static": static,
        "extraction_version": "timegrid_v2.1.0", "validity_schema_version": "validity_v0.2",
        "stage_code_aligned": stages, "stage_valid": np.ones(t, bool),
        "eeg_channel_available": np.ones(6, bool),
        "eeg_clean_subsegment_count": np.ones((t, 6), int),
        "emg_channel_available": np.ones(3, bool), "emg_epoch_success": np.ones((t, 3), bool),
        "resp_feature_valid": np.ones((t, 14), bool),
        "hrv_success": np.ones(w, bool), "hrv_feature_valid": np.ones((w, 11), bool),
        "ecg_alignment_valid": np.ones(t, bool),
        "arousal_valid": np.ones(t, bool), "resp_event_valid": np.ones(t, bool),
        "limb_event_valid": np.ones(t, bool), "annotation_status": "available",
        "stage_probability": probability, "stage_probability_valid": np.ones(t, bool),
        "stage_probability_channels_present": np.ones(5, bool),
        "annotation_signal_lengths": {
            "caisr_prob_n1": t, "caisr_prob_n2": t, "caisr_prob_n3": t,
            "caisr_prob_r": t, "caisr_prob_w": t,
        },
    }


class G2AFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest(MANIFEST)

    def row(self, feature_id):
        return next(r for r in self.manifest.rows if r["unique_feature_id"] == feature_id)

    def test_manifest_sha_dimensions_and_invalid_manifest_rejected(self):
        self.assertEqual(self.manifest.sha256, EXPECTED_MANIFEST_SHA256)
        self.assertEqual(len(self.manifest.rows), 827)
        self.assertEqual(len(self.manifest.unique_ids), 807)
        self.assertEqual(self.manifest.view_sizes, {"V1":113,"V2":129,"V3":349,"V4":113,"V5":123})
        with self.assertRaises(ValueError):
            load_manifest(MANIFEST, expected_sha256="0" * 64)

    def test_pool_four_five_boundary_mean_and_linear_iqr(self):
        v,m,s = pool_values(np.arange(4.0), np.ones(4,bool), "Mean")
        self.assertTrue(np.isnan(v)); self.assertFalse(m); self.assertEqual(s,4)
        v,m,s = pool_values(np.arange(5.0), np.ones(5,bool), "IQR")
        self.assertEqual((v,m,s),(2.0,True,5))

    def test_eeg_independent_lead_and_nrem_includes_n1(self):
        r=record(); r["stage_code_aligned"] = np.array([1,2,3,1,2.])
        r["X_seq"][:,0]=np.arange(5.)
        r["eeg_clean_subsegment_count"][:,1]=0
        result=compute_night_features(r,self.manifest)
        idx=result.unique_ids.index("eeg_NREM_eeg_F3-M2_logP2P1_mean")
        self.assertEqual(result.unique_feature_values[idx],2.0); self.assertTrue(result.unique_feature_valid_mask[idx])
        f4=result.unique_ids.index("eeg_NREM_eeg_F4-M1_logP2P1_mean")
        self.assertFalse(result.unique_feature_valid_mask[f4])
        self.assertFalse(any(x.startswith("eeg_N1_") for x in result.unique_ids))

    def test_spatial_six_and_twelve_indices_same_epoch(self):
        r=record()
        asym=self.row("spatial_asym_N2_F3-F4_delta_mean")
        idx=[int(x) for x in asym["source_indices"].split(";")]
        r["X_seq"][:,idx[0]]=2; r["X_seq"][:,idx[1]]=1
        value,mask,support=spatial_value_and_mask(asym,r)
        self.assertEqual((value,mask,support),(1.0,True,5)); self.assertEqual(len(idx),6)
        grad=self.row("spatial_ap_gradient_N2_delta_mean")
        gidx=[int(x) for x in grad["source_indices"].split(";")]
        r["X_seq"][:,gidx[0:2]]=2; r["X_seq"][:,gidx[2:4]]=1
        value,mask,support=spatial_value_and_mask(grad,r)
        self.assertEqual((value,mask,support),(1.0,True,5)); self.assertEqual(len(gidx),12)
        r["eeg_clean_subsegment_count"][0,0]=0
        self.assertEqual(spatial_value_and_mask(grad,r)[2],4)

    def test_hrv_emg_resp_masks_and_valid_zero(self):
        r=record(); result=compute_night_features(r,self.manifest)
        for uid in ("hrv_model_HRV_MedianNN_mean","emg_chin_log_rms_norm_mean","resp_airflow_rate_bpm_mean"):
            i=result.unique_ids.index(uid); self.assertTrue(result.unique_feature_valid_mask[i]); self.assertEqual(result.unique_feature_values[i],0.0); self.assertEqual(result.unique_feature_support_count[i],5)
        r["hrv_feature_valid"][:2,0]=False
        r["emg_channel_available"][1]=False
        r["resp_feature_valid"][:2,0]=False
        result=compute_night_features(r,self.manifest)
        self.assertFalse(result.unique_feature_valid_mask[result.unique_ids.index("hrv_model_HRV_MedianNN_mean")])
        self.assertFalse(result.unique_feature_valid_mask[result.unique_ids.index("emg_lleg_log_rms_norm_mean")])
        self.assertFalse(result.unique_feature_valid_mask[result.unique_ids.index("resp_airflow_rate_bpm_mean")])

    def test_caisr_entropy_complete_missing_nine_and_other_zero(self):
        r=record(); self.assertTrue(entropy_source_valid(r,float(r["x_static"][87])))
        r["stage_probability"][0,0]=9; self.assertFalse(entropy_source_valid(r,float(r["x_static"][87])))
        r=record(); r["stage_probability_channels_present"][0]=False
        self.assertFalse(entropy_source_valid(r,float(r["x_static"][87])))
        r=record(); r["x_static"][96]=0
        result=compute_night_features(r,self.manifest); i=result.unique_ids.index("caisr_arousal_arousal_index")
        self.assertTrue(result.unique_feature_valid_mask[i]); self.assertEqual(result.unique_feature_values[i],0)

    def test_demo5_others_unavailable_bmi_and_repeat_views(self):
        r=record(); r["x_static"][4:9]=0; r["x_static"][6]=1; r["x_static"][9]=0
        v,m,s=compute_demo5(r["x_static"])
        np.testing.assert_array_equal(v[[2,3,4]],[0,0,0]); self.assertTrue(m[[2,3,4]].all()); self.assertFalse(m[1]); self.assertEqual(s[1],0)
        result=compute_night_features(r,self.manifest)
        for uid in ("demo_male","demo_bmi","demo_race_white","demo_race_black","demo_race_asian"):
            slots=[]
            for view in result.view_values:
                rows=[x for x in self.manifest.rows if x["view_id"]==view]
                pos=next(int(x["view_position"]) for x in rows if x["unique_feature_id"]==uid)
                slots.append((result.view_values[view][pos],result.view_valid_masks[view][pos],result.view_support_counts[view][pos]))
            self.assertTrue(all((np.isnan(a[0]) and np.isnan(slots[0][0]) or a[0]==slots[0][0]) and a[1:]==slots[0][1:] for a in slots))
        r["x_static"][4:9]=0; r["x_static"][7]=1
        self.assertFalse(compute_demo5(r["x_static"])[1][2:].any())

    def test_full_output_invariants_and_views(self):
        result=compute_night_features(record(),self.manifest)
        self.assertEqual(len(result.unique_feature_values),807)
        self.assertEqual({k:len(v) for k,v in result.view_values.items()},self.manifest.view_sizes)
        self.assertEqual(sum(map(len,result.view_values.values())),827)
        self.assertTrue(np.isfinite(result.unique_feature_values[result.unique_feature_valid_mask]).all())
        self.assertTrue(np.isnan(result.unique_feature_values[~result.unique_feature_valid_mask]).all())
        self.assertTrue((result.unique_feature_support_count>=0).all())


if __name__ == "__main__": unittest.main(verbosity=2)
