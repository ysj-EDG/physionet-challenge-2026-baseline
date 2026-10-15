import json
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from reproduction.siren_a.g4.verify_g4a import complete_unit


class G4BRunnerContractTests(unittest.TestCase):
    def test_runner_imports_and_schedules_exact_smoke_and_full_work(self):
        from reproduction.siren_a.g4b.run_g4b import (
            full_model_schedule,
            full_prediction_schedule,
            smoke_model_schedule,
            smoke_prediction_schedule,
        )

        smoke_models = smoke_model_schedule()
        smoke_predictions = smoke_prediction_schedule()
        self.assertEqual(len(smoke_models), 6)
        self.assertEqual(len(smoke_predictions), 18)
        self.assertEqual({item[1] for item in smoke_models}, {43})
        self.assertEqual({item[2] for item in smoke_models}, {"V1"})
        self.assertEqual({item[4] for item in smoke_predictions}, {43})
        self.assertEqual({item[2] for item in smoke_predictions}, {"V1"})
        self.assertEqual(len(full_model_schedule()), 420)
        self.assertEqual(len(full_prediction_schedule()), 1260)

    def test_paths_bind_seed_and_all_identity_fields(self):
        from reproduction.siren_a.g4b.run_g4b import (
            model_path,
            prediction_path,
            query_path,
            sample_path,
        )

        root = Path("/private")
        self.assertEqual(sample_path(root, 43, "S0001"), root / "seeds/seed43/samples/S0001")
        self.assertEqual(query_path(root, 43, "I0006"), root / "seeds/seed43/queries/I0006")
        self.assertEqual(
            model_path(root, 44, "S0001", "V3", "hgb"),
            root / "seeds/seed44/models/S0001/V3/hgb",
        )
        self.assertEqual(
            prediction_path(root, 56, "I0002", "inductive", "V5", "lambdarank"),
            root / "seeds/seed56/predictions/holdout_I0002/inductive/V5/lambdarank",
        )

    def test_unit_build_is_no_clobber_and_rejects_partial_or_changed_identity(self):
        from reproduction.siren_a.g4b.run_g4b import build_unit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "unit"
            identity = {"run": "a", "seed": 43}

            def writer(path):
                (path / "value.json").write_text('{"value":1}\n', encoding="utf-8")

            self.assertEqual(build_unit(target, "test_v1", identity, writer), "BUILT_NEW")
            original = (target / "value.json").read_bytes()
            self.assertEqual(build_unit(target, "test_v1", identity, writer), "SKIP_VERIFIED")
            self.assertEqual((target / "value.json").read_bytes(), original)
            with self.assertRaises(ValueError):
                build_unit(target, "test_v1", {"run": "b", "seed": 43}, writer)

            (target / "value.json").write_text('{"value":2}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                build_unit(target, "test_v1", identity, writer)

            partial = root / "partial"
            (root / "partial.building").mkdir()
            with self.assertRaises(ValueError):
                build_unit(partial, "test_v1", identity, writer)

            interrupted = root / "interrupted"

            def failing_writer(path):
                (path / "partial.bin").write_bytes(b"partial")
                raise RuntimeError("injected interruption")

            with self.assertRaises(RuntimeError):
                build_unit(interrupted, "test_v1", identity, failing_writer)
            self.assertFalse(interrupted.exists())
            self.assertTrue((root / "interrupted.building").is_dir())
            with self.assertRaises(ValueError):
                build_unit(interrupted, "test_v1", identity, writer)

    def test_upstream_code_identity_drift_is_rejected(self):
        from reproduction.siren_a.g4b.run_g4b import verify_g4a_source_code

        frozen = {"code_sha256": {"sampler.py": "aaa", "queries.py": "bbb"}}
        current = {"identity": {"code_sha256": dict(frozen["code_sha256"])}}
        self.assertTrue(verify_g4a_source_code(current, frozen))
        current["identity"]["code_sha256"]["sampler.py"] = "changed"
        with self.assertRaises(ValueError):
            verify_g4a_source_code(current, frozen)

    def test_runner_has_no_holdout_target_or_scoring_surface(self):
        from reproduction.siren_a.g4b import run_g4b

        source = inspect.getsource(run_g4b)
        for forbidden in (".y_test", ".age_test", "roc_auc", "average_precision", "f1_score"):
            self.assertNotIn(forbidden, source)

    def test_train_sharing_requires_exact_rows_masks_targets_and_feature_order(self):
        from reproduction.siren_a.g4b.run_g4b import check_shared_training

        class Fold:
            pass

        left = Fold()
        left.train_site = "S0001"
        left.X_train = np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float64)
        left.train_valid_mask = np.isfinite(left.X_train)
        left.y_train = np.array([0, 1], dtype=np.int8)
        left.age_train = np.array([55.0, 60.0], dtype=np.float64)
        left.train_age_groups = np.array([11, 12], dtype=np.int64)
        left.train_indices = np.array([3, 9], dtype=np.int64)
        left.feature_ids = ("a", "b")

        right = Fold()
        for name in ("train_site", "X_train", "train_valid_mask", "y_train", "age_train",
                     "train_age_groups", "train_indices", "feature_ids"):
            value = getattr(left, name)
            setattr(right, name, value.copy() if isinstance(value, np.ndarray) else value)
        self.assertTrue(check_shared_training(left, right))
        right.train_indices = np.array([9, 3], dtype=np.int64)
        with self.assertRaises(ValueError):
            check_shared_training(left, right)

    def test_cli_exposes_only_execution_modes_not_hyperparameters_or_scoring(self):
        from reproduction.siren_a.g4b.run_g4b import build_parser

        parser = build_parser()
        options = {option for action in parser._actions for option in action.option_strings}
        self.assertEqual(
            options,
            {"-h", "--help", "--preflight-only", "--smoke", "--build", "--resume", "--verify-only"},
        )
        for forbidden in ("--seed", "--learning-rate", "--evaluate", "--score"):
            self.assertNotIn(forbidden, options)

    def test_prediction_unit_replay_rejects_corruption(self):
        from reproduction.siren_a.g4b import run_g4b

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = SimpleNamespace(private_root=root)
            fold = SimpleNamespace(
                X_test=np.array([[0.0], [1.0]], dtype=np.float64),
                test_indices=np.array([11, 12], dtype=np.int64),
                feature_ids=("feature",),
            )
            scores = np.array([0.25, 0.75], dtype=np.float64)
            model_root = run_g4b.model_path(root, 43, "S0001", "V1", "lr")
            model_root.mkdir(parents=True)
            (model_root / "complete.json").write_text("{}\n", encoding="utf-8")
            model_meta = {"checkpoint_files": {"model.joblib": "checkpoint-sha"}}
            loaded = SimpleNamespace(model=object(), adapter=object())
            identity = run_g4b._prediction_identity(
                "run", 43, "I0002", "transductive", "V1", "lr",
                run_g4b.array_sha256(fold.test_indices),
                run_g4b.sha256_file(model_root / "complete.json"),
                run_g4b.array_sha256(scores),
            )
            target = run_g4b.prediction_path(
                root, 43, "I0002", "transductive", "V1", "lr")

            def writer(path):
                np.savez(path / "prediction.npz", scores=scores,
                         test_global_indices=fold.test_indices)
                (path / "metadata.json").write_text(
                    json.dumps({"checkpoint_sha256": "checkpoint-sha"}), encoding="utf-8")

            run_g4b.build_unit(target, "g4b_prediction_v1", identity, writer)
            patches = (
                patch.object(run_g4b, "_load", return_value=fold),
                patch.object(run_g4b, "_verify_model_unit",
                             return_value=(model_meta, loaded, scores)),
                patch.object(run_g4b, "predict_scores", return_value=scores),
            )
            with patches[0], patches[1], patches[2]:
                run_g4b._verify_prediction_unit(
                    config, "run", object(), 43, "I0002", "transductive", "V1",
                    "lr", object(), object())
            np.savez(target / "prediction.npz", scores=scores[::-1],
                     test_global_indices=fold.test_indices)
            with patches[0], patches[1], patches[2], self.assertRaises(ValueError):
                run_g4b._verify_prediction_unit(
                    config, "run", object(), 43, "I0002", "transductive", "V1",
                    "lr", object(), object())


if __name__ == "__main__":
    unittest.main()
