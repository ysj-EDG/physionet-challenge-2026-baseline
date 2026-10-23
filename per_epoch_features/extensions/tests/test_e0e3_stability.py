import unittest
from unittest.mock import patch

import numpy as np
import extensions.e0e3_stability as stability

from extensions.e0e3_stability import (
    assert_public_privacy,
    build_fused_groups,
    build_prediction_source_keys,
    build_replacement_keys,
    bootstrap_multiplicity,
    fuse_score_columns,
    pair_contributions,
    select_seed_columns,
    weighted_ac_batch,
    weighted_ac_from_pairs,
)


class SeedAndFusionTests(unittest.TestCase):
    def test_select_seed_columns_requires_exact_fifteen_and_fuses_mean_ranks(self):
        keys = [(42, v, m) for v in ("V1", "V2", "V3", "V4", "V5")
                for m in ("lr", "hgb", "lambdarank")]
        scores = np.arange(60, dtype=np.float64).reshape(4, 15)
        chosen, matrix = select_seed_columns(keys, scores, 42)
        self.assertEqual(len(chosen), 15)
        indices, fused = fuse_score_columns(matrix)
        self.assertEqual(indices.shape, (4,))
        self.assertTrue(np.allclose(fused, np.array([0.125, 0.375, 0.625, 0.875])))

    def test_fusion_rejects_nonfinite_scores(self):
        with self.assertRaises(ValueError):
            fuse_score_columns(np.array([[0.1, np.nan] * 8], dtype=np.float64)[:, :15])

    def test_seed_pairing_rejects_wrong_or_duplicate_keys(self):
        keys = [(42, "V1", "lr")]
        with self.assertRaises(ValueError):
            select_seed_columns(keys, np.ones((2, 1)), 42)


class BootstrapTests(unittest.TestCase):
    def test_age_gap_boundary_is_inclusive(self):
        labels = np.array([1, 0, 0], dtype=np.int8)
        ages = np.array([30.0, 32.0, 32.1])
        scores = np.array([0.9, 0.1, 0.8])
        pairs = pair_contributions(labels, ages, scores)
        self.assertEqual(pairs["positive"].tolist(), [0])
        self.assertEqual(pairs["negative"].tolist(), [1])

    def test_age_gap_above_two_is_ineligible(self):
        labels = np.array([1, 0], dtype=np.int8)
        ages = np.array([30.0, 32.01])
        scores = np.array([0.9, 0.1])
        self.assertEqual(len(pair_contributions(labels, ages, scores)["positive"]), 0)

    def test_tie_contribution_is_half(self):
        labels = np.array([1, 0], dtype=np.int8)
        ages = np.array([30.0, 30.0])
        scores = np.array([0.5, 0.5])
        pairs = pair_contributions(labels, ages, scores)
        self.assertEqual(pairs["contribution"].tolist(), [0.5])

    def test_bootstrap_seed_is_deterministic(self):
        first = bootstrap_multiplicity(5, np.random.default_rng(20261023))
        second = bootstrap_multiplicity(5, np.random.default_rng(20261023))
        np.testing.assert_array_equal(first, second)

    def test_batch_invalid_replicate_remains_nan(self):
        labels = np.array([1, 1], dtype=np.int8)
        ages = np.array([30.0, 31.0])
        scores = np.array([0.1, 0.2])
        pairs = pair_contributions(labels, ages, scores)
        result = weighted_ac_batch(pairs, np.array([[1, 1], [0, 0]], dtype=np.int64))
        self.assertTrue(np.isnan(result).all())
    def test_weighted_ac_matches_explicit_expansion_with_ties(self):
        labels = np.array([1, 0, 1, 0], dtype=np.int8)
        ages = np.array([30.0, 30.0, 31.0, 40.0])
        scores = np.array([0.5, 0.5, 0.2, 0.1])
        multiplicity = np.array([2, 1, 3, 2], dtype=np.int64)
        pairs = pair_contributions(labels, ages, scores, gap=2.0)
        weighted = weighted_ac_from_pairs(pairs, multiplicity)
        expanded = np.repeat(np.arange(len(labels)), multiplicity)
        self.assertAlmostEqual(weighted, 0.2, places=12)
        self.assertTrue(np.isfinite(weighted))
        self.assertEqual(expanded.tolist(), [0, 0, 1, 2, 2, 2, 3, 3])

    def test_cluster_multiplicity_propagates_to_records(self):
        multiplicity = bootstrap_multiplicity(4, np.random.default_rng(7), [[0, 1], [2], [3]])
        self.assertEqual(multiplicity[0], multiplicity[1])
        self.assertEqual(multiplicity.shape, (4,))

    def test_bootstrap_rejects_nonpartitioned_clusters(self):
        with self.assertRaises(ValueError):
            bootstrap_multiplicity(3, np.random.default_rng(1), [[0, 1], [1, 2]])

    def test_invalid_bootstrap_without_eligible_pairs_is_nan(self):
        labels = np.array([1, 1], dtype=np.int8)
        ages = np.array([20.0, 21.0])
        scores = np.array([0.1, 0.2])
        pairs = pair_contributions(labels, ages, scores)
        self.assertTrue(np.isnan(weighted_ac_from_pairs(pairs, np.ones(2, dtype=np.int64))))

    def test_batch_weighted_ac_matches_scalar_engine(self):
        labels = np.array([1, 0, 1, 0], dtype=np.int8)
        ages = np.array([30.0, 30.0, 31.0, 40.0])
        scores = np.array([0.5, 0.5, 0.2, 0.1])
        pairs = pair_contributions(labels, ages, scores)
        draws = np.array([[1, 1, 1, 1], [2, 1, 3, 2]], dtype=np.int64)
        expected = np.array([weighted_ac_from_pairs(pairs, row) for row in draws])
        np.testing.assert_allclose(weighted_ac_batch(pairs, draws), expected)


class ReplacementTests(unittest.TestCase):
    def test_rv5_reuses_e2_identity_and_view_model_changes_fifteen(self):
        e2 = {(seed, view, model): (f"v2-{seed}-{view}-{model}" if view == "V5"
                                    else f"old-{seed}-{view}-{model}")
              for seed in range(42, 57) for view in ("V1", "V2", "V3", "V4", "V5")
              for model in ("lr", "hgb", "lambdarank")}
        e0 = {key: f"old-{key[0]}-{key[1]}-{key[2]}" for key in e2}
        rv5 = build_replacement_keys(e0, e2, view="V5")
        self.assertEqual(rv5, e2)
        one = build_replacement_keys(e0, e2, view="V5", model="lr")
        self.assertEqual(sum(one[k] != e0[k] for k in one), 15)

    def test_replacement_rejects_unknown_view(self):
        base = {(42, view, model): 0 for view in ("V1", "V2", "V3", "V4", "V5")
                for model in ("lr", "hgb", "lambdarank")}
        with self.assertRaises(ValueError):
            build_replacement_keys(base, base, view="V9")

    def test_each_view_model_replacement_changes_exactly_fifteen(self):
        e0 = {key: f"old-{key}" for key in
              ((seed, view, model) for seed in range(42, 57)
               for view in ("V1", "V2", "V3", "V4", "V5")
               for model in ("lr", "hgb", "lambdarank"))}
        e2 = {key: f"new-{key}" for key in e0}
        for view in ("V1", "V2", "V3", "V4", "V5"):
            for model in ("lr", "hgb", "lambdarank"):
                out = build_replacement_keys(e0, e2, view=view, model=model)
                self.assertEqual(sum(out[k] != e0[k] for k in out), 15)


class ProductionSourceSelectionTests(unittest.TestCase):
    def test_build_fused_groups_passes_key_source_map_to_fuser(self):
        captured = []

        def fake_fuser(arrays, holdout, protocol, source_map, seed=None):
            captured.append(source_map)
            return [], [], np.array([0], dtype=np.int64), np.array([0.5], dtype=np.float64)

        with patch.object(stability, "_fuse_source", side_effect=fake_fuser):
            stability.build_fused_groups({}, view="V5", model="lr")
        self.assertEqual(len(captured), 6)
        for source_map in captured:
            changed = {key for key, source in source_map.items() if source == "v2"}
            self.assertEqual(len(changed), 15)
            self.assertTrue(all(key[1:] == ("V5", "lr") for key in changed))

    def test_view_model_source_selection_is_key_granular(self):
        sources = build_prediction_source_keys(view="V5", model="lr")
        self.assertEqual(len(sources), 225)
        changed = {key for key, source in sources.items() if source == "v2"}
        self.assertEqual(len(changed), 15)
        self.assertTrue(all(key[1:] == ("V5", "lr") for key in changed))
        self.assertEqual(sum(source == "old" for source in sources.values()), 210)

    def test_three_v5_model_replacements_have_disjoint_key_sets(self):
        changed = []
        for model in ("lr", "hgb", "lambdarank"):
            sources = build_prediction_source_keys(view="V5", model=model)
            changed.append({key for key, source in sources.items() if source == "v2"})
        self.assertEqual([len(value) for value in changed], [15, 15, 15])
        self.assertTrue(changed[0].isdisjoint(changed[1]))
        self.assertTrue(changed[0].isdisjoint(changed[2]))
        self.assertTrue(changed[1].isdisjoint(changed[2]))

    def test_full_view_replacement_stays_at_forty_five_columns(self):
        sources = build_prediction_source_keys(view="V5")
        self.assertEqual(sum(source == "v2" for source in sources.values()), 45)
        self.assertEqual(sum(source == "old" for source in sources.values()), 180)

    def test_all_view_model_replacements_cover_exactly_225_keys(self):
        for view in ("V1", "V2", "V3", "V4", "V5"):
            for model in ("lr", "hgb", "lambdarank"):
                sources = build_prediction_source_keys(view=view, model=model)
                self.assertEqual(set(sources), set((seed, v, m)
                                 for seed in range(42, 57)
                                 for v in ("V1", "V2", "V3", "V4", "V5")
                                 for m in ("lr", "hgb", "lambdarank")))
                self.assertEqual(sum(x == "v2" for x in sources.values()), 15)
                self.assertEqual(sum(x == "old" for x in sources.values()), 210)


class PrivacyTests(unittest.TestCase):
    def test_public_rows_reject_record_level_fields(self):
        with self.assertRaises(ValueError):
            assert_public_privacy([{"scope": "Macro", "record_id": "secret"}])

    def test_public_rows_accept_aggregate_fields(self):
        assert_public_privacy([{"scope": "Macro", "delta_ac": 0.1,
                                "valid_replicates": 10}])


if __name__ == "__main__":
    unittest.main()
