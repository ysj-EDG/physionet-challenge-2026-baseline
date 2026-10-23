import unittest


class CoreCohEmgAnalysisTests(unittest.TestCase):
    def test_two_by_two_effects_and_interaction_close(self):
        from extensions.core_coh_emg_analysis import build_core_2x2_row
        row = build_core_2x2_row(
            protocol="transductive", scope="macro",
            core=0.60, caisr_only=0.65, resp_only=0.70, both=0.80,
        )
        self.assertAlmostEqual(row["caisr_effect_resp_absent"], 0.05)
        self.assertAlmostEqual(row["caisr_effect_resp_present"], 0.10)
        self.assertAlmostEqual(row["resp_effect_caisr_absent"], 0.10)
        self.assertAlmostEqual(row["resp_effect_caisr_present"], 0.15)
        self.assertAlmostEqual(row["interaction"], 0.05)
        self.assertAlmostEqual(row["interaction"], row["caisr_effect_resp_present"] - row["caisr_effect_resp_absent"], places=12)
        self.assertAlmostEqual(row["interaction"], row["resp_effect_caisr_present"] - row["resp_effect_caisr_absent"], places=12)

    def test_two_by_two_row_collection_is_fixed_and_deterministic(self):
        from extensions.core_coh_emg_analysis import build_core_2x2_rows
        metrics = {("transductive", "Macro"): {"CORE_COH_EMG": 0.6, "LOO_NO_RESP": 0.65, "LOO_NO_CAISR": 0.7, "E2": 0.8}}
        rows = build_core_2x2_rows(metrics)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scope"], "Macro")
        with self.assertRaises(ValueError):
            build_core_2x2_rows({("transductive", "Macro"): {"CORE_COH_EMG": 0.6}})

    def test_two_by_two_requires_finite_aligned_cells(self):
        from extensions.core_coh_emg_analysis import build_core_2x2_row
        with self.assertRaises(ValueError):
            build_core_2x2_row(protocol="inductive", scope="worst_site", core=float("nan"), caisr_only=0.2, resp_only=0.3, both=0.4)


if __name__ == "__main__":
    unittest.main()
