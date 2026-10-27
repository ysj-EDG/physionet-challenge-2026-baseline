import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class CrossFoldSelectionTests(unittest.TestCase):
    def test_candidate_grid_is_exact_and_pre_registered(self):
        from extensions.siren_matcha_ab_crossfold import CANDIDATE_WEIGHTS

        self.assertEqual(tuple(CANDIDATE_WEIGHTS), (0.2, 0.3, 0.4, 0.5, 0.6, 0.7))

    def test_target_fold_is_excluded_from_selection(self):
        from extensions.siren_matcha_ab_crossfold import selection_folds

        self.assertEqual(selection_folds("I0002"), ("I0006", "S0001"))
        self.assertEqual(selection_folds("I0006"), ("I0002", "S0001"))
        self.assertEqual(selection_folds("S0001"), ("I0002", "I0006"))

    def test_deterministic_tie_break_prefers_prior_then_lower_weight(self):
        from extensions.siren_matcha_ab_crossfold import select_weight

        self.assertEqual(select_weight({0.3: 0.7, 0.5: 0.7}), 0.3)
        self.assertEqual(select_weight({0.2: 0.7, 0.6: 0.7}), 0.2)
        self.assertEqual(select_weight({0.4: 0.6, 0.5: 0.6000000000005}), 0.4)

    def test_fixed_weights_replay_previous_routes(self):
        from extensions.siren_matcha_ab_crossfold import build_weighted_score

        a = np.array([0.2, 0.8], dtype=np.float64)
        b = np.array([0.6, 0.4], dtype=np.float64)
        np.testing.assert_array_equal(build_weighted_score(a, b, 0.4), np.array([0.36, 0.64]))
        np.testing.assert_allclose(build_weighted_score(a, b, 0.5), np.array([0.4, 0.6]))

    def test_nested_bootstrap_excludes_target_and_frequency_sums(self):
        from extensions.siren_matcha_ab_crossfold import nested_crossfold_bootstrap

        sites = {
            site: {
                "labels": np.tile(np.array([0, 1], dtype=np.int8), 10),
                "ages": np.full(20, 50.0),
                "rA": np.tile(np.array([0.1, 0.9]), 10),
                "rB": np.tile(np.array([0.2, 0.8]), 10),
            }
            for site in ("I0002", "I0006", "S0001")
        }
        result = nested_crossfold_bootstrap(sites, n_bootstrap=17, seed=20261023)
        for target in sites:
            counts = result["selection_frequency"][target]
            self.assertEqual(sum(counts.values()), 17)
            self.assertEqual(result["selection_exclusion"][target], selection_without_target(target))

    def test_parent_manifest_hashes_are_verified(self):
        from extensions.siren_matcha_ab_crossfold import load_frozen_payloads

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root / "fused" / "holdout_TEST" / "payload.npz"
            payload.parent.mkdir(parents=True)
            np.savez(payload, global_index=np.array([1]), record_id=np.array(["r1"]), label=np.array([0]), age=np.array([50.0]), rA=np.array([0.2]), rB=np.array([0.3]), A_final=np.array([0.2]), B_final=np.array([0.3]), B_seed_20260806_logit=np.array([0.3]), B_seed_20260807_logit=np.array([0.3]), B_seed_20260808_logit=np.array([0.3]), AB60=np.array([0.24]), AB50=np.array([0.25]))
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest = {"private_payload_sha256": {"TEST": digest}}
            (root / "manifests").mkdir()
            (root / "manifests" / "private_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_frozen_payloads(root, sites=("TEST",))
            self.assertEqual(len(loaded["TEST"]["rA"]), 1)


def selection_without_target(target):
    return tuple(site for site in ("I0002", "I0006", "S0001") if site != target)


if __name__ == "__main__":
    unittest.main()
