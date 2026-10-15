import unittest

from reproduction.siren_a.g4.config import HGB_PARAMS, LGBM_PARAMS, LR_PARAMS
from reproduction.siren_a.g4b.config import (
    ALL_SEEDS,
    NEW_SEEDS,
    full_model_identities,
    full_prediction_identities,
    new_model_identities,
    new_prediction_identities,
    parameters_for_seed,
)


class SeedScheduleTests(unittest.TestCase):
    def test_new_schedule_excludes_seed42_and_closes_expected_counts(self):
        self.assertEqual(NEW_SEEDS, tuple(range(43, 57)))
        self.assertEqual(ALL_SEEDS, tuple(range(42, 57)))
        self.assertNotIn(42, NEW_SEEDS)
        models = new_model_identities()
        predictions = new_prediction_identities()
        self.assertEqual((len(models), len(set(models))), (420, 420))
        self.assertEqual((len(predictions), len(set(predictions))), (1260, 1260))
        self.assertEqual({seed for _, seed, _, _ in models}, set(NEW_SEEDS))
        self.assertTrue(all(sum(item[1] == seed for item in models) == 30 for seed in NEW_SEEDS))
        self.assertTrue(all(sum(item[4] == seed for item in predictions) == 90 for seed in NEW_SEEDS))

    def test_full_schedule_closes_seed42_without_duplicates(self):
        models = full_model_identities()
        predictions = full_prediction_identities()
        self.assertEqual((len(models), len(set(models))), (450, 450))
        self.assertEqual((len(predictions), len(set(predictions))), (1350, 1350))
        self.assertEqual({item[1] for item in models}, set(ALL_SEEDS))
        self.assertEqual({item[4] for item in predictions}, set(ALL_SEEDS))

    def test_parameter_injection_changes_only_seed_and_query_truncation(self):
        snapshots = (dict(LR_PARAMS), dict(HGB_PARAMS), dict(LGBM_PARAMS))
        for seed in (43, 44, 56):
            lr = parameters_for_seed("lr", seed, 17)
            hgb = parameters_for_seed("hgb", seed, 17)
            lgb = parameters_for_seed("lambdarank", seed, 17)
            self.assertEqual(lr, {**snapshots[0], "random_state": seed})
            self.assertEqual(hgb, {**snapshots[1], "random_state": seed})
            self.assertEqual(
                lgb,
                {**snapshots[2], "random_state": seed, "lambdarank_truncation_level": 17},
            )
        self.assertEqual((LR_PARAMS, HGB_PARAMS, LGBM_PARAMS), snapshots)

    def test_invalid_seed_model_or_truncation_is_rejected(self):
        for bad_seed in (42, 57, 43.0):
            with self.assertRaises(ValueError):
                parameters_for_seed("lr", bad_seed, 2)
        with self.assertRaises(ValueError):
            parameters_for_seed("unknown", 43, 2)
        with self.assertRaises(ValueError):
            parameters_for_seed("lambdarank", 43, 0)


if __name__ == "__main__":
    unittest.main()
