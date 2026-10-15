import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reproduction.siren_a.g5.components import (
    _identity,
    _verify_complete,
    component_scores,
    evaluate_components,
    make_components,
    summarize_components,
    validate_protocol_pair,
)


def model_keys():
    return np.asarray([
        f"seed{seed}|{view}|{model}"
        for seed in range(42, 57)
        for view in ("V1", "V2", "V3", "V4", "V5")
        for model in ("lr", "hgb", "lambdarank")
    ])


class ComponentDefinitionTests(unittest.TestCase):
    def test_exact_24_components_and_column_counts(self):
        components = make_components(model_keys())
        self.assertEqual(len(components), 24)
        self.assertEqual(len({item.component_id for item in components}), 24)
        counts = [len(item.columns) for item in components]
        self.assertEqual(counts[:15], [15] * 15)
        self.assertEqual(counts[15:20], [45] * 5)
        self.assertEqual(counts[20:23], [75] * 3)
        self.assertEqual(counts[23:], [225])

    def test_components_are_key_driven_and_shuffle_invariant(self):
        keys = model_keys()
        ranks = np.arange(6 * 225, dtype=np.float64).reshape(6, 225) / 2000.0
        components = make_components(keys)
        first = component_scores(ranks, components)
        order = np.random.default_rng(7).permutation(225)
        shuffled_components = make_components(keys[order])
        second = component_scores(ranks[:, order], shuffled_components)
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-15)

    def test_rejects_missing_duplicate_and_illegal_keys(self):
        keys = model_keys()
        variants = (keys[:-1], np.concatenate((keys[:-1], keys[:1])),
                    np.concatenate((keys[:-1], np.array(["seed56|V5|zscore"]))))
        for bad in variants:
            with self.subTest(last=bad[-1]):
                with self.assertRaises(ValueError):
                    make_components(bad)

    def test_cell_contains_only_requested_view_and_model(self):
        keys = model_keys()
        components = {item.component_id: item for item in make_components(keys)}
        selected = keys[np.asarray(components["V3_hgb"].columns)]
        self.assertEqual(len(selected), 15)
        self.assertTrue(all("|V3|hgb" in key for key in selected))


class ComponentScoreTests(unittest.TestCase):
    def test_scores_average_existing_ranks_without_reranking(self):
        keys = model_keys()
        ranks = np.tile(np.linspace(0.001, 0.999, 225), (4, 1))
        ranks[0, :] *= 0.2
        components = make_components(keys)
        scores = component_scores(ranks, components)
        cell = next(i for i, item in enumerate(components) if item.component_id == "V1_lr")
        columns = np.asarray(components[cell].columns)
        np.testing.assert_array_equal(scores[:, cell], ranks[:, columns].mean(axis=1))
        self.assertLess(scores[0, cell], scores[1, cell])

    def test_all_component_closures_match_saved_full(self):
        keys = model_keys()
        ranks = np.random.default_rng(11).random((13, 225), dtype=np.float64)
        components = make_components(keys)
        full = ranks.mean(axis=1, dtype=np.float64)
        scores = component_scores(ranks, components, saved_full=full)
        by_id = {item.component_id: scores[:, i] for i, item in enumerate(components)}
        view_mean = np.column_stack([by_id[f"V{i}_all"] for i in range(1, 6)]).mean(axis=1)
        family_mean = np.column_stack([by_id[f"all_{m}"] for m in ("lr", "hgb", "lambdarank")]).mean(axis=1)
        cell_mean = np.column_stack([by_id[f"V{i}_{m}"] for i in range(1, 6)
                                     for m in ("lr", "hgb", "lambdarank")]).mean(axis=1)
        np.testing.assert_allclose(by_id["all_all"], full, rtol=0, atol=1e-15)
        np.testing.assert_allclose(view_mean, full, rtol=0, atol=1e-15)
        np.testing.assert_allclose(family_mean, full, rtol=0, atol=1e-15)
        np.testing.assert_allclose(cell_mean, full, rtol=0, atol=1e-15)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "evaluate_model.py"
        spec = importlib.util.spec_from_file_location("official_component_test", path)
        cls.official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.official)

    def test_official_gap_two_and_ties(self):
        labels = np.array([1, 0, 0], dtype=np.int8)
        ages = np.array([60.0, 62.0, 63.0])
        scores = np.column_stack((np.array([0.5, 0.5, 0.0]), np.array([0.9, 0.2, 0.1])))
        rows = evaluate_components(labels, ages, scores, ("a", "b"), self.official)
        self.assertEqual(rows[0]["ac_pair_count"], 1)
        self.assertEqual(rows[0]["ac_auroc"], 0.5)

    def test_macro_and_deltas_are_protocol_specific_and_unweighted(self):
        rows = []
        for protocol, component, values in (
            ("transductive", "all_all", (0.2, 0.4, 0.9)),
            ("transductive", "V1_all", (0.3, 0.3, 0.6)),
            ("inductive", "all_all", (0.5, 0.5, 0.5)),
            ("inductive", "V1_all", (0.6, 0.4, 0.5)),
        ):
            for site, value in zip(("I0002", "I0006", "S0001"), values):
                rows.append({"holdout_site": site, "protocol": protocol,
                             "component_id": component, "ac_auroc": value})
        summary = summarize_components(rows)
        self.assertAlmostEqual(summary[("transductive", "all_all")]["macro_ac"], 0.5)
        self.assertAlmostEqual(summary[("transductive", "V1_all")]["delta_macro_vs_full"], -0.1)
        self.assertAlmostEqual(summary[("inductive", "V1_all")]["delta_macro_vs_full"], 0.0)

    def test_protocol_pair_requires_identical_evaluation_rows(self):
        common = dict(test_global_indices=np.array([1, 2]), labels=np.array([0, 1]),
                      ages=np.array([60.0, 61.0]))
        validate_protocol_pair(common, common)
        for key, value in (("test_global_indices", np.array([2, 1])),
                           ("labels", np.array([1, 0])),
                           ("ages", np.array([60.0, 62.0]))):
            changed = dict(common)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol_pair(common, changed)

    def test_completed_result_is_reused_only_with_pass_and_same_identity(self):
        import reproduction.siren_a.g5.components as module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"schema": "test", "source": "unchanged"}
            summary = {"rows": [{}] * 144, "summaries": [{}] * 48}
            (root / "summaries.json").write_text(json.dumps(summary), encoding="utf-8")
            complete = {"schema": module.SCHEMA, "status": "PASS",
                        "identity_sha256": _identity(config), "files": {}}
            (root / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
            with patch.object(module, "PRIVATE_ROOT", root):
                self.assertEqual(len(_verify_complete(config)["rows"]), 144)
                complete["status"] = "INCOMPLETE"
                (root / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
                with self.assertRaises(ValueError):
                    _verify_complete(config)


if __name__ == "__main__":
    unittest.main()
