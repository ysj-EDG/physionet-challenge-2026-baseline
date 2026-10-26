import unittest

import numpy as np


from extensions.caisr20_core_protocol import CAISR20_FEATURE_IDS
CAISR = tuple(CAISR20_FEATURE_IDS)


def rows():
    from extensions.v5_family_loo import V5ManifestRow
    blocks = [("old807", 123), ("eeg_coherence_stage", 90), ("emg_stage", 72), ("caisr_stage", 35), ("resp_stage", 56)]
    out = []
    pos = 0
    # CORE rows deliberately begin with the exact standalone CAISR IDs.
    for i, feature_id in enumerate(CAISR):
        out.append(V5ManifestRow(i, feature_id, i, "old807"))
    pos = 20
    for block, width in (("old807", 103), ("eeg_coherence_stage", 90), ("emg_stage", 72)):
        for j in range(width):
            if block == "old807":
                prefix = ("hrv_model_" if j < 22 else "emg_" if j < 70 else "resp_" if j < 98 else "demo_")
                feature_id = f"{prefix}{j}"
            else:
                feature_id = f"{block}_{j}"
            out.append(V5ManifestRow(pos, feature_id, 1000 + pos, block)); pos += 1
    return tuple(out)


class CoreNoCAISRContractTests(unittest.TestCase):
    def test_removes_exact_twenty_ids_and_retains_order(self):
        from extensions.core_no_caisr20 import build_no_caisr20_rows

        original = rows()
        retained = build_no_caisr20_rows(original, CAISR)
        self.assertEqual(len(original), 285)
        self.assertEqual(len(retained), 265)
        self.assertEqual(tuple(row.feature_id for row in retained), tuple(row.feature_id for row in original[20:]))
        self.assertEqual(set(row.feature_id for row in original) - set(row.feature_id for row in retained), set(CAISR))
        self.assertEqual(set(row.feature_id for row in retained) - set(row.feature_id for row in original), set())

    def test_rejects_noncanonical_core_prefix(self):
        from extensions.core_no_caisr20 import CoreNoCAISRStop, build_no_caisr20_rows

        original = list(rows()); original[0], original[1] = original[1], original[0]
        with self.assertRaises(CoreNoCAISRStop):
            build_no_caisr20_rows(tuple(original), CAISR)

    def test_rank_subset_is_exact_and_does_not_rebuild(self):
        from extensions.core_no_caisr20 import project_rank_subset

        values = np.arange(3 * 285, dtype=np.float64).reshape(3, 285)
        positions = np.arange(20, 285, dtype=np.int64)
        subset = project_rank_subset(values, positions)
        np.testing.assert_array_equal(subset, values[:, 20:])
        self.assertEqual(float(np.max(np.abs(subset - values[:, 20:]))), 0.0)

    def test_family_counts_and_standalone_disjointness(self):
        from extensions.core_no_caisr20 import family_counts, feature_id_disjoint

        retained = rows()[20:]
        self.assertEqual(family_counts(retained), {"HRV": 22, "EMG": 48, "Resp": 28, "Demo": 5, "Coherence": 90, "StageEMG": 72, "other": 0})
        self.assertTrue(feature_id_disjoint(tuple(row.feature_id for row in retained), CAISR))


if __name__ == "__main__":
    unittest.main()
