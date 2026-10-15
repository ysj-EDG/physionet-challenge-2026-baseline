import copy
import unittest

from reproduction.siren_a.g4b.config import (
    full_model_identities,
    full_prediction_identities,
)


def synthetic_entries():
    models = []
    for site, seed, view, model_type in full_model_identities():
        models.append({
            "train_site": site,
            "seed": seed,
            "view": view,
            "model_type": model_type,
            "checkpoint_reference": f"m/{site}/{seed}/{view}/{model_type}",
            "model_sha256": f"model-{site}-{seed}-{view}-{model_type}",
            "adapter_sha256": f"adapter-{site}-{seed}-{view}-{model_type}",
            "feature_order_sha256": f"feature-{view}",
            "sample_sha256": f"sample-{site}-{seed}",
            "completion_sha256": f"complete-{site}-{seed}-{view}-{model_type}",
        })
    predictions = []
    for holdout, protocol, view, model_type, seed in full_prediction_identities():
        train_site = "I0006" if holdout == "S0001" else "S0001"
        predictions.append({
            "holdout_site": holdout,
            "train_site": train_site,
            "protocol": protocol,
            "view": view,
            "model_type": model_type,
            "seed": seed,
            "prediction_reference": f"p/{holdout}/{protocol}/{view}/{model_type}/{seed}",
            "prediction_sha256": f"pred-{holdout}-{protocol}-{view}-{model_type}-{seed}",
            "test_indices_sha256": f"test-{holdout}-{protocol}",
            "score_type": "raw_ranking_score" if model_type == "lambdarank" else "positive_probability",
            "prediction_length": {"I0002": 319, "I0006": 1142, "S0001": 5139}[holdout],
            "model_sha256": f"model-{train_site}-{seed}-{view}-{model_type}",
            "adapter_sha256": f"adapter-{train_site}-{seed}-{view}-{model_type}",
            "feature_order_sha256": f"feature-{view}",
            "completion_sha256": f"pred-complete-{holdout}-{protocol}-{view}-{model_type}-{seed}",
        })
    return models, predictions


class G5ReadyManifestTests(unittest.TestCase):
    def test_complete_manifest_closes_450_models_and_1350_predictions(self):
        from reproduction.siren_a.g4b.manifest import validate_manifest_entries

        models, predictions = synthetic_entries()
        result = validate_manifest_entries(models, predictions)
        self.assertEqual(result["models"], 450)
        self.assertEqual(result["predictions"], 1350)
        self.assertEqual(result["seeds"], 15)
        self.assertEqual(set(result["prediction_units_per_holdout_protocol"].values()), {225})

    def test_duplicate_or_missing_model_and_prediction_are_rejected(self):
        from reproduction.siren_a.g4b.manifest import validate_manifest_entries

        models, predictions = synthetic_entries()
        for broken_models, broken_predictions in (
            (models[:-1], predictions),
            (models + [copy.deepcopy(models[0])], predictions),
            (models, predictions[:-1]),
            (models, predictions + [copy.deepcopy(predictions[0])]),
        ):
            with self.assertRaises(ValueError):
                validate_manifest_entries(broken_models, broken_predictions)

    def test_prediction_model_trace_and_test_index_closure_are_enforced(self):
        from reproduction.siren_a.g4b.manifest import validate_manifest_entries

        models, predictions = synthetic_entries()
        broken = copy.deepcopy(predictions)
        broken[0]["model_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            validate_manifest_entries(models, broken)
        broken = copy.deepcopy(predictions)
        target = next(row for row in broken if row["holdout_site"] == "I0002" and row["protocol"] == "transductive")
        target["test_indices_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            validate_manifest_entries(models, broken)

    def test_score_types_lengths_and_feature_order_must_match_contract(self):
        from reproduction.siren_a.g4b.manifest import validate_manifest_entries

        models, predictions = synthetic_entries()
        for field, value in (("score_type", "probability"),
                             ("prediction_length", 1),
                             ("feature_order_sha256", "wrong")):
            broken = copy.deepcopy(predictions)
            broken[0][field] = value
            with self.assertRaises(ValueError):
                validate_manifest_entries(models, broken)


if __name__ == "__main__":
    unittest.main()
