import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproduction.siren_a.pipeline.label_join import MetadataIndex, MetadataRecord
from reproduction.siren_a.g5.core import (
    aggregate_protocol,
    array_sha256,
    load_eval_targets,
    load_predictions,
    mean_ranks,
    score_site,
)


class RankTests(unittest.TestCase):
    def test_average_tie_percentiles(self):
        scores = np.array([[2.0], [2.0], [5.0], [9.0]])
        ranks, ensemble = mean_ranks(scores)
        expected = np.array([0.25, 0.25, 0.625, 0.875])
        np.testing.assert_array_equal(ranks[:, 0], expected)
        np.testing.assert_array_equal(ensemble, expected)

    def test_rank_each_model_then_average(self):
        scores = np.array([[0.0, 100.0], [10.0, 0.0], [11.0, 1.0]])
        ranks, ensemble = mean_ranks(scores)
        raw_first = np.argsort(np.argsort(scores.mean(axis=1)))
        self.assertFalse(np.array_equal(np.argsort(ensemble), raw_first))
        np.testing.assert_allclose(ensemble, ranks.mean(axis=1), rtol=0, atol=0)

    def test_ranks_are_direction_fixed_and_label_independent(self):
        scores = np.array([[3.0], [1.0], [2.0]])
        ranks, first = mean_ranks(scores)
        labels = np.array([0, 1, 0])
        labels[:] = 1 - labels
        _, second = mean_ranks(scores)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(ranks[0, 0], ranks[2, 0])

    def test_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            mean_ranks(np.array([[0.0], [np.nan]]))


class PredictionLoadingTests(unittest.TestCase):
    def _fixture(self, root):
        approved = root / "approved"
        unit = approved / "unit"
        unit.mkdir(parents=True, exist_ok=True)
        scores = np.array([0.2, 0.8], dtype=np.float64)
        indices = np.array([0, 1], dtype=np.int64)
        np.savez(unit / "prediction.npz", scores=scores, test_global_indices=indices)
        row = {
            "holdout_site": "I0002", "train_site": "S0001",
            "protocol": "transductive", "view": "V1", "model_type": "lr",
            "seed": 42, "prediction_reference": str(unit / "prediction.npz"),
            "prediction_sha256": array_sha256(scores),
            "test_indices_sha256": array_sha256(indices),
            "score_type": "positive_probability", "prediction_length": 2,
            "model_sha256": "m", "adapter_sha256": "a",
            "feature_order_sha256": "f",
        }
        model = {"train_site": "S0001", "seed": 42, "view": "V1",
                 "model_type": "lr", "model_sha256": "m",
                 "adapter_sha256": "a", "feature_order_sha256": "f"}
        prediction_manifest = root / "predictions.json"
        model_manifest = root / "models.json"
        prediction_manifest.write_text(json.dumps({"entries": [row]}), encoding="utf-8")
        model_manifest.write_text(json.dumps({"entries": [model]}), encoding="utf-8")
        return prediction_manifest, model_manifest, row, approved

    def test_loads_fixed_grid_and_rejects_duplicates_or_wrong_site_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, m, row, approved = self._fixture(root)
            groups, _ = load_predictions(
                p, m, {"I0002": np.array([0, 1])}, (approved,),
                holdouts=("I0002",), protocols=("transductive",),
                views=("V1",), model_types=("lr",), seeds=(42,))
            self.assertEqual(groups[("I0002", "transductive")].scores.shape, (2, 1))
            p.write_text(json.dumps({"entries": [row, row]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_predictions(p, m, {"I0002": np.array([0, 1])}, (approved,),
                                 holdouts=("I0002",), protocols=("transductive",),
                                 views=("V1",), model_types=("lr",), seeds=(42,))
            p.write_text(json.dumps({"entries": [row]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_predictions(p, m, {"I0002": np.array([1, 2])}, (approved,),
                                 holdouts=("I0002",), protocols=("transductive",),
                                 views=("V1",), model_types=("lr",), seeds=(42,))

    def test_rejects_hash_nonfinite_probability_and_unapproved_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, m, row, approved = self._fixture(root)
            for mutation in ("hash", "nonfinite", "range", "path"):
                current = dict(row)
                if mutation == "hash":
                    current["prediction_sha256"] = "bad"
                elif mutation in {"nonfinite", "range"}:
                    values = np.array([np.nan, 0.8]) if mutation == "nonfinite" else np.array([1.2, 0.8])
                    path = Path(current["prediction_reference"])
                    np.savez(path, scores=values.astype(np.float64),
                             test_global_indices=np.array([0, 1], dtype=np.int64))
                    current["prediction_sha256"] = array_sha256(values.astype(np.float64))
                else:
                    current["prediction_reference"] = str(root / "outside.npz")
                p.write_text(json.dumps({"entries": [current]}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_predictions(p, m, {"I0002": np.array([0, 1])}, (approved,),
                                     holdouts=("I0002",), protocols=("transductive",),
                                     views=("V1",), model_types=("lr",), seeds=(42,))
                if mutation in {"nonfinite", "range"}:
                    self._fixture(root)


class EvaluationTests(unittest.TestCase):
    def test_inventory_filesystem_metadata_does_not_change_identity_match(self):
        from reproduction.siren_a.g5.run_g5 import _load_identity_sites

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sites = ["I0002"] * 319 + ["I0006"] * 1142 + ["S0001"] * 5139
            private = [{"cache_name": f"cache_{i:06d}.npz",
                        "relative_path": f"{site}/record_{i}.npz", "site": site}
                       for i, site in enumerate(sites)]
            inventory = [dict(row, size=100 + i, mtime_ns=200 + i)
                         for i, row in enumerate(private)]
            (root / "inventory.json").write_text(
                json.dumps({"rows": inventory}), encoding="utf-8")
            (root / "private_index.json").write_text(json.dumps(private), encoding="utf-8")
            result = _load_identity_sites(root)
            self.assertEqual({key: len(value) for key, value in result.items()},
                             {"I0002": 319, "I0006": 1142, "S0001": 5139})

    def _metadata(self):
        records = tuple(
            MetadataRecord("I0002", f"sub-{i}", "01", label, age, "", i)
            for i, (label, age) in enumerate((("True", "60"), ("False", "62"), ("False", "63")))
        )
        return MetadataIndex(records, tuple(f"cache_{i:06d}.npz" for i in range(3)),
                             tuple(f"I0002/r{i}.npz" for i in range(3)),
                             np.array(["I0002"] * 3), "d", "j", {"matched": 3})

    def test_eval_targets_follow_indices_not_metadata_input_order(self):
        labels, ages = load_eval_targets(self._metadata(), np.array([2, 0, 1]))
        np.testing.assert_array_equal(labels, np.array([0, 1, 0], dtype=np.int8))
        np.testing.assert_array_equal(ages, np.array([63.0, 60.0, 62.0]))
        with self.assertRaises(ValueError):
            load_eval_targets(self._metadata(), np.array([0, 0]))

    def test_official_age_gap_ties_and_metrics(self):
        path = Path(__file__).resolve().parents[3] / "evaluate_model.py"
        spec = importlib.util.spec_from_file_location("official_eval_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        labels = np.array([1, 0, 0], dtype=np.int8)
        ages = np.array([60.0, 62.0, 63.0])
        scores = np.array([0.5, 0.5, 0.0])
        result = score_site(labels, scores, ages, module)
        self.assertEqual(result["ac_pair_count"], 1)
        self.assertEqual(result["ac_auroc"], 0.5)
        self.assertEqual(result["n_positive"], 1)
        self.assertEqual(result["n_negative"], 2)

    def test_macro_is_unweighted_and_incomplete_nan_is_not_ignored(self):
        rows = [{"holdout_site": "I0002", "ac_auroc": 0.2},
                {"holdout_site": "I0006", "ac_auroc": 0.4},
                {"holdout_site": "S0001", "ac_auroc": 0.9}]
        result = aggregate_protocol(rows)
        self.assertAlmostEqual(result["macro_ac"], 0.5)
        self.assertAlmostEqual(result["small_site_mean_ac"], 0.3)
        rows[2]["ac_auroc"] = float("nan")
        self.assertTrue(np.isnan(aggregate_protocol(rows)["macro_ac"]))


if __name__ == "__main__":
    unittest.main()
