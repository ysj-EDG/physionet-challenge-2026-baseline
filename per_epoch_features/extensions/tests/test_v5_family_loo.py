import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class V5FamilyLooContractTests(unittest.TestCase):
    def test_protocol_reference_experiment_binding_is_explicit(self):
        from extensions.v5_family_loo_analysis import build_protocol_summary_rows

        site_rows = [
            {"config_id": "LOO_NO_RESP", "protocol": "transductive", "holdout_site": "I0002", "ac_auroc": 0.61},
            {"config_id": "LOO_NO_RESP", "protocol": "transductive", "holdout_site": "I0006", "ac_auroc": 0.70},
            {"config_id": "LOO_NO_RESP", "protocol": "transductive", "holdout_site": "S0001", "ac_auroc": 0.79},
        ]
        references = {
            ("E2", "transductive"): {"macro_ac": 0.68, "worst_site_ac": 0.60, "small_site_mean_ac": 0.65},
            ("E0", "transductive"): {"macro_ac": 0.65, "worst_site_ac": 0.55, "small_site_mean_ac": 0.62},
        }
        row = build_protocol_summary_rows(site_rows, references)[0]
        self.assertAlmostEqual(row["delta_vs_E2_macro"], 0.02)
        self.assertAlmostEqual(row["delta_vs_E0_macro"], 0.05)
        self.assertNotEqual(row["delta_vs_E2_macro"], row["delta_vs_E0_macro"])
        self.assertAlmostEqual(row["delta_vs_E2_worst"], 0.01)
        self.assertAlmostEqual(row["delta_vs_E0_worst"], 0.06)

    def test_protocol_reference_is_separated_by_protocol(self):
        from extensions.v5_family_loo_analysis import build_protocol_summary_rows

        site_rows = [
            {"config_id": "LOO_NO_RESP", "protocol": "transductive", "holdout_site": "I0002", "ac_auroc": 0.61},
            {"config_id": "LOO_NO_RESP", "protocol": "transductive", "holdout_site": "I0006", "ac_auroc": 0.61},
            {"config_id": "LOO_NO_RESP", "protocol": "transductive", "holdout_site": "S0001", "ac_auroc": 0.61},
        ]
        references = {
            ("E2", "transductive"): {"macro_ac": 0.60, "worst_site_ac": 0.60, "small_site_mean_ac": 0.60},
            ("E2", "inductive"): {"macro_ac": 0.80, "worst_site_ac": 0.80, "small_site_mean_ac": 0.80},
            ("E0", "transductive"): {"macro_ac": 0.55, "worst_site_ac": 0.55, "small_site_mean_ac": 0.55},
        }
        row = build_protocol_summary_rows(site_rows, references)[0]
        self.assertAlmostEqual(row["delta_vs_E2_macro"], 0.01)

    def test_frozen_protocol_reference_loader_uses_ac_fields(self):
        from extensions.v5_family_loo_analysis import load_protocol_references

        path = Path(__file__).parents[3] / "audit" / "featurelab_v2_e0_e3_protocol_summary.csv"
        references = load_protocol_references(path)
        self.assertNotEqual(references[("E0", "transductive")]["macro_ac"], references[("E2", "transductive")]["macro_ac"])
        self.assertNotEqual(references[("E0", "inductive")]["macro_ac"], references[("E2", "inductive")]["macro_ac"])
        self.assertIn("worst_site_ac", references[("E2", "transductive")])

    def test_core_manifest_keeps_only_old_coherence_and_emg_blocks(self):
        from extensions.core_coh_emg import CORE_CONFIG_ID, select_core_config, load_core_manifest_payload
        from extensions.v5_family_loo import load_v5_manifest

        manifest = load_v5_manifest(Path(__file__).parents[3] / "audit" / "featurelab_v2_view_manifest.csv")
        selected = select_core_config(manifest)
        self.assertEqual(CORE_CONFIG_ID, "CORE_COH_EMG")
        self.assertEqual(len(selected), 285)
        self.assertEqual({row.source_block for row in selected}, {"old807", "eeg_coherence_stage", "emg_stage"})
        self.assertEqual([row.view_position for row in selected], sorted(row.view_position for row in selected))
        self.assertEqual(len({row.feature_id for row in selected}), 285)
        self.assertEqual(len({row.unique_column_index for row in selected}), 285)
        payload = load_core_manifest_payload(manifest)
        self.assertEqual(payload["retained_family_counts"], {"old807": 123, "eeg_coherence_stage": 90, "emg_stage": 72})
        self.assertEqual(payload["removed_family_counts"], {"caisr_stage": 35, "resp_stage": 56})

    def test_core_runner_requires_explicit_selector_registry(self):
        import inspect
        from extensions.v5_family_loo_runner import run_models_and_predictions

        parameters = inspect.signature(run_models_and_predictions).parameters
        self.assertIn("selector", parameters)
        self.assertIn("allowed_configs", parameters)

    def test_core_projection_is_distinct_from_loo_configuration_registry(self):
        from extensions.core_coh_emg import CORE_CONFIG_ID
        from extensions.v5_family_loo import CONFIGS
        self.assertNotIn(CORE_CONFIG_ID, CONFIGS)

    def test_manifest_defines_four_widths_and_source_blocks(self):
        from extensions.v5_family_loo import CONFIGS, load_v5_manifest
        rows = load_v5_manifest(Path(__file__).parents[3] / "audit" / "featurelab_v2_view_manifest.csv")
        self.assertEqual(tuple(CONFIGS), ("LOO_NO_COHERENCE", "LOO_NO_CAISR", "LOO_NO_EMG", "LOO_NO_RESP"))
        self.assertEqual(len(rows.v5_rows), 376)
        self.assertEqual({name: len(rows.columns(name)) for name in CONFIGS}, {
            "LOO_NO_COHERENCE": 286, "LOO_NO_CAISR": 341,
            "LOO_NO_EMG": 304, "LOO_NO_RESP": 320,
        })

    def test_selection_preserves_repeated_old_feature_identity(self):
        from extensions.v5_family_loo import load_v5_manifest, select_config
        manifest = load_v5_manifest(Path(__file__).parents[3] / "audit" / "featurelab_v2_view_manifest.csv")
        selected = select_config(manifest, "LOO_NO_RESP")
        old = [row for row in manifest.v5_rows if row.source_block == "old807"]
        self.assertEqual([x.unique_column_index for x in selected[:123]], [x.unique_column_index for x in old])
        self.assertEqual(len({x.feature_id for x in selected}), len(selected))

    def test_projection_preserves_nan_masks_and_feature_order(self):
        from extensions.v5_family_loo import load_v5_manifest, project_matrix, select_config
        manifest = load_v5_manifest(Path(__file__).parents[3] / "audit" / "featurelab_v2_view_manifest.csv")
        selected = select_config(manifest, "LOO_NO_COHERENCE")
        source = np.arange(2 * 376, dtype=np.float64).reshape(2, 376)
        source[0, 4] = np.nan
        mask = np.isfinite(source)
        values, projected = project_matrix(source, mask, selected)
        self.assertEqual(values.shape, (2, 286))
        np.testing.assert_array_equal(projected, np.isfinite(values))
        np.testing.assert_array_equal(values[:, 0], source[:, 0])

    def test_config_manifest_is_deterministic_and_order_independent(self):
        from extensions.v5_family_loo import config_manifest, load_v5_manifest
        manifest = load_v5_manifest(Path(__file__).parents[3] / "audit" / "featurelab_v2_view_manifest.csv")
        a = config_manifest(manifest, ["LOO_NO_RESP", "LOO_NO_EMG"])
        b = config_manifest(manifest, ["LOO_NO_EMG", "LOO_NO_RESP"])
        self.assertEqual(a, b)
        self.assertIn("feature_ids_sha256", a["LOO_NO_RESP"])

    def test_atomic_private_unit_no_clobber_and_resume(self):
        from extensions.v5_family_loo_runner import publish_npz_unit, UnitIdentityError
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unit"
            payload = {"scores": np.array([0.1, 0.9], dtype=np.float64), "indices": np.array([3, 4], dtype=np.int64)}
            metadata = {"schema": "synthetic", "identity": "a"}
            self.assertEqual(publish_npz_unit(path, payload, metadata), "BUILD_NEW")
            original = (path / "payload.npz").read_bytes()
            self.assertEqual(publish_npz_unit(path, payload, metadata), "SKIP_VERIFIED")
            self.assertEqual((path / "payload.npz").read_bytes(), original)
            with self.assertRaises(UnitIdentityError):
                publish_npz_unit(path, payload, {"schema": "synthetic", "identity": "b"})

    def test_delta_name_and_bootstrap_are_deterministic(self):
        from extensions.v5_family_loo_runner import bootstrap_delta, comparison_name
        self.assertEqual(comparison_name("LOO_NO_RESP", "E2"), "LOO_NO_RESP-E2")
        baseline = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        loo = np.array([0.2, 0.1, 0.4], dtype=np.float64)
        a = bootstrap_delta(loo, baseline, seed=20261023, n_bootstrap=17)
        b = bootstrap_delta(loo, baseline, seed=20261023, n_bootstrap=17)
        np.testing.assert_array_equal(a, b)

    def test_public_payload_rejects_private_fields(self):
        from extensions.v5_family_loo_runner import validate_public_payload
        with self.assertRaises(ValueError):
            validate_public_payload({"config_id": "LOO_NO_RESP", "record_id": "patient"})
        self.assertTrue(validate_public_payload({"config_id": "LOO_NO_RESP", "ac": 0.5}))


if __name__ == "__main__":
    unittest.main()
