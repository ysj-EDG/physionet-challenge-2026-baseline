import unittest

import numpy as np


class CoreNoCAISRRunnerTests(unittest.TestCase):
    def test_model_and_prediction_identity_cardinality_is_frozen(self):
        from extensions.core_no_caisr20_runner import model_identities_no_caisr20, prediction_identities_no_caisr20, validate_no_caisr_run_contract

        models = model_identities_no_caisr20()
        predictions = prediction_identities_no_caisr20()
        self.assertEqual(len(models), 90)
        self.assertEqual(len(predictions), 135)
        self.assertTrue(all(item[2] == "V5" and item[3] == "CORE_NO_CAISR20" for item in models))
        self.assertTrue(all(item[2] == "V5" and item[3] == "CORE_NO_CAISR20" and item[4] == "inductive" for item in predictions))
        self.assertEqual(validate_no_caisr_run_contract(models, predictions), {"models": 90, "predictions": 135})

    def test_final_ensemble_has_180_frozen_plus_45_new_columns(self):
        from extensions.core_no_caisr20_runner import assemble_final_columns

        frozen = np.arange(4 * 180, dtype=np.float64).reshape(4, 180)
        new = np.arange(4 * 45, dtype=np.float64).reshape(4, 45)
        out = assemble_final_columns(frozen, new)
        self.assertEqual(out.shape, (4, 225))
        np.testing.assert_array_equal(out[:, :180], frozen)
        np.testing.assert_array_equal(out[:, 180:], new)

    def test_no_caisr_route_never_accepts_old_core_v5(self):
        from extensions.core_no_caisr20_runner import CoreNoCAISRRunnerStop, validate_prediction_route

        with self.assertRaises(CoreNoCAISRRunnerStop):
            validate_prediction_route({"route_id": "CORE_COH_EMG", "view": "V5"})
        self.assertTrue(validate_prediction_route({"route_id": "CORE_NO_CAISR20", "view": "V5"}))


if __name__ == "__main__":
    unittest.main()
