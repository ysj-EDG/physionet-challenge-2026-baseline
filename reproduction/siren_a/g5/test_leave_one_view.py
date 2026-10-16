import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reproduction.siren_a.g5.leave_one_view import (
    _identity,
    _verify_complete,
    evaluate,
    leave_one_view_scores,
    select_remaining_columns,
    summarize,
    validate_protocol_pair,
)


def model_keys():
    return np.asarray([
        f"seed{seed}|{view}|{model}"
        for seed in range(42, 57)
        for view in ("V1", "V2", "V3", "V4", "V5")
        for model in ("lr", "hgb", "lambdarank")
    ])


class SelectionTests(unittest.TestCase):
    def test_five_fixed_removals_have_exact_partition(self):
        keys = model_keys()
        selections = select_remaining_columns(keys)
        self.assertEqual(tuple(selections), tuple(f"drop_V{i}" for i in range(1, 6)))
        for name, item in selections.items():
            self.assertEqual(len(item.removed), 45)
            self.assertEqual(len(item.remaining), 180)
            self.assertFalse(set(item.removed) & set(item.remaining))
            self.assertEqual(set(item.removed) | set(item.remaining), set(range(225)))
            dropped = name[-2:]
            self.assertTrue(all(f"|{dropped}|" in keys[index] for index in item.removed))

    def test_shuffle_with_matrix_preserves_each_named_score(self):
        keys = model_keys()
        ranks = np.random.default_rng(17).random((9, 225), dtype=np.float64)
        full = ranks.mean(axis=1, dtype=np.float64)
        names, expected = leave_one_view_scores(ranks, keys, full)
        order = np.random.default_rng(9).permutation(225)
        shuffled_names, actual = leave_one_view_scores(ranks[:, order], keys[order], full)
        self.assertEqual(names, shuffled_names)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_missing_duplicate_and_illegal_keys_are_rejected(self):
        keys = model_keys()
        bad_sets = (keys[:-1], np.concatenate((keys[:-1], keys[:1])),
                    np.concatenate((keys[:-1], np.array(["seed56|V6|lr"]))))
        for bad in bad_sets:
            with self.subTest(last=bad[-1]), self.assertRaises(ValueError):
                select_remaining_columns(bad)


class ScoreTests(unittest.TestCase):
    def test_direct_mean_without_reranking_and_formula_closure(self):
        keys = model_keys()
        ranks = np.tile(np.linspace(0.001, 0.999, 225), (5, 1))
        ranks[0] *= 0.1
        full = ranks.mean(axis=1, dtype=np.float64)
        names, scores = leave_one_view_scores(ranks, keys, full)
        selections = select_remaining_columns(keys)
        for column, name in enumerate(names[1:], start=1):
            selection = selections[name]
            direct = ranks[:, np.asarray(selection.remaining)].mean(axis=1, dtype=np.float64)
            view = ranks[:, np.asarray(selection.removed)].mean(axis=1, dtype=np.float64)
            np.testing.assert_array_equal(scores[:, column], direct)
            np.testing.assert_allclose(direct, (5 * full - view) / 4, rtol=0, atol=1e-15)
        self.assertLess(scores[0, 1], scores[1, 1])
        with self.assertRaises(ValueError):
            leave_one_view_scores(ranks, keys, full + 1e-5)

    def test_protocol_pair_requires_identical_indices_labels_and_ages(self):
        common = {"test_global_indices": np.array([1, 2]), "labels": np.array([0, 1], dtype=np.int8),
                  "ages": np.array([60.0, 61.0])}
        validate_protocol_pair(common, common)
        for key, value in (("test_global_indices", np.array([2, 1])),
                           ("labels", np.array([1, 0], dtype=np.int8)),
                           ("ages", np.array([60.0, 62.0]))):
            changed = dict(common)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol_pair(common, changed)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "evaluate_model.py"
        spec = importlib.util.spec_from_file_location("official_g5d_test", path)
        cls.official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.official)

    def test_official_gap_two_tie_and_delta_sign(self):
        labels = np.array([1, 0, 0], dtype=np.int8)
        ages = np.array([60.0, 62.0, 63.0])
        scores = np.column_stack((np.array([0.5, 0.5, 0.0]), np.array([0.9, 0.2, 0.1])))
        baseline = {"ac_auroc": 0.5, "auroc": 0.5, "auprc": 0.5, "ac_pair_count": 1,
                    "n_records": 3, "n_positive": 1, "n_negative": 2, "status": "PASS"}
        rows = evaluate(labels, ages, scores, ("full", "drop_V1"), self.official, baseline)
        self.assertEqual(rows[0]["ac_pair_count"], 1)
        self.assertEqual(rows[0]["ac_auroc"], 0.5)
        self.assertEqual(rows[1]["ac_auroc"], 1.0)
        self.assertEqual(rows[1]["delta_ac_vs_full"], 0.5)

    def test_macro_is_equal_weight_and_deltas_use_protocol_full(self):
        rows = []
        for protocol, configuration, values in (
            ("transductive", "full", (0.2, 0.4, 0.9)),
            ("transductive", "drop_V1", (0.3, 0.3, 0.6)),
            ("inductive", "full", (0.5, 0.5, 0.5)),
            ("inductive", "drop_V1", (0.6, 0.4, 0.5)),
        ):
            for site, value in zip(("I0002", "I0006", "S0001"), values):
                rows.append({"holdout_site": site, "protocol": protocol,
                             "configuration": configuration, "ac_auroc": value})
        summary = summarize(rows)
        self.assertAlmostEqual(summary[("transductive", "full")]["macro_ac"], 0.5)
        self.assertAlmostEqual(summary[("transductive", "drop_V1")]["delta_macro_vs_full"], -0.1)
        self.assertAlmostEqual(summary[("inductive", "drop_V1")]["delta_macro_vs_full"], 0.0)

    def test_complete_reuse_requires_same_identity(self):
        import reproduction.siren_a.g5.leave_one_view as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"schema": "test", "source": "same"}
            payload = {"rows": [{}] * 36, "summaries": [{}] * 12}
            (root / "scores.json").write_text(json.dumps(payload), encoding="utf-8")
            complete = {"schema": module.SCHEMA, "status": "PASS", "identity_sha256": _identity(config)}
            (root / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
            with patch.object(module, "PRIVATE_ROOT", root):
                self.assertEqual(len(_verify_complete(config)["rows"]), 36)
                with self.assertRaises(ValueError):
                    _verify_complete({"schema": "test", "source": "changed"})

    def test_analysis_identity_directly_covers_all_six_result_files(self):
        import reproduction.siren_a.g5.leave_one_view as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "g5"
            repo = Path(tmp) / "repo"
            root.mkdir()
            repo.mkdir()
            for name in ("complete.json", "run_config.json", "scores.json"):
                (root / name).write_text(name, encoding="utf-8")
            (repo / "evaluate_model.py").write_text("official", encoding="utf-8")
            for holdout in module.HOLDOUTS:
                for protocol in module.PROTOCOLS:
                    path = root / "groups" / f"{holdout}_{protocol}" / "result.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{holdout}-{protocol}".encode())
            with patch.object(module, "G5_ROOT", root), patch.object(module, "REPO", repo):
                before = module._analysis_config()
                changed = root / "groups/I0002_transductive/result.npz"
                changed.write_bytes(b"changed")
                after = module._analysis_config()
            self.assertNotEqual(before["source_result_sha256"], after["source_result_sha256"])
            self.assertEqual(len(before["source_result_sha256"]), 6)

    def test_cached_build_republishes_verified_public_deliverables_without_scoring(self):
        import reproduction.siren_a.g5.leave_one_view as module
        payload = {"rows": [{}] * 36, "summaries": [{}] * 12}
        with patch.object(module, "_analysis_config", return_value={"schema": "x"}), \
                patch.object(module, "_verify_complete", return_value=payload), \
                patch.object(module, "_publish_from_payload") as publish, \
                patch.object(module, "load_inputs", side_effect=AssertionError("must not reload or score")):
            self.assertIs(module.build(), payload)
        publish.assert_called_once_with(payload)


if __name__ == "__main__":
    unittest.main()
