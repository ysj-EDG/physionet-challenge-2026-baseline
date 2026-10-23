import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from extensions.compare_e0_e3_v2 import (
    EXPERIMENT_SOURCES,
    SEEDS,
    assert_e0_fusion_regression,
    build_all_reference_entries,
    fixed_column_keys,
    fuse_prediction_columns,
    fuse_reference_group,
    group_reference_entries,
    load_prediction_row,
    normalize_score_type,
    validate_old_prediction_row,
    validate_v2_prediction_unit,
    verify_official_scoring_module,
)
from reproduction.siren_a.g5.core import (
    HOLDOUTS,
    HOLDOUT_TO_TRAIN,
    MODEL_TYPES,
    PROTOCOLS,
    VIEWS,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _array_sha(array):
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _v2_array_sha(array):
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(_canonical(list(value.shape)))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_v2_unit(root, *, seed, holdout="I0002", protocol="transductive", view="V1", model="lr"):
    root = Path(root)
    unit = root / "unit"
    unit.mkdir()
    scores = np.array([0.1, 0.8, 0.3], dtype=np.float64)
    indices = np.array([8, 2, 5], dtype=np.int64)
    with (unit / "prediction.npz").open("wb") as handle:
        np.savez_compressed(handle, scores=scores, test_global_indices=indices)
    schema = "featurelab_v2_seed42_prediction_v1" if seed == 42 else "featurelab_v2_multiseed_prediction_v1"
    run_identity = root / "run_identity.json"
    run_identity.write_bytes(_canonical({"schema": "synthetic_run"}))
    model_complete = root / "models" / HOLDOUT_TO_TRAIN[holdout] / f"seed{seed}" / view / model / "complete.json"
    model_complete.parent.mkdir(parents=True)
    model_complete.write_bytes(_canonical({"status": "COMPLETE"}))
    metadata = {
        "schema": schema,
        "seed": seed,
        "holdout_site": holdout,
        "train_site": HOLDOUT_TO_TRAIN[holdout],
        "protocol": protocol,
        "view": view,
        "model_type": model,
        "score_type": "lambdarank_raw" if model == "lambdarank" else "positive_probability",
        "test_count": len(indices),
        "finite": True,
        "nonconstant": True,
        "prediction_sha256": _v2_array_sha(scores),
        "test_indices_sha256": _v2_array_sha(indices),
        "model_complete_sha256": _file_sha(model_complete),
        "run_identity_sha256": _file_sha(run_identity),
    }
    if seed != 42:
        metadata["source_version"] = "v2"
    (unit / "metadata.json").write_bytes(_canonical(metadata))
    summaries = {p.name: _file_sha(p) for p in sorted(unit.iterdir()) if p.name != "complete.json"}
    complete = {"schema": schema, "status": "COMPLETE", "file_summaries": summaries}
    (unit / "complete.json").write_bytes(_canonical(complete))
    expected = {
        "seed": seed,
        "holdout_site": holdout,
        "train_site": HOLDOUT_TO_TRAIN[holdout],
        "protocol": protocol,
        "view": view,
        "model_type": model,
    }
    return unit, expected, scores, indices


def _write_old_unit(root, *, seed=42, holdout="I0002", protocol="transductive", view="V1", model="lr"):
    root = Path(root)
    unit = root / "old_unit"
    unit.mkdir()
    scores = np.array([0.2, 0.9, 0.4], dtype=np.float64)
    indices = np.array([8, 2, 5], dtype=np.int64)
    np.savez_compressed(unit / "prediction.npz", scores=scores, test_global_indices=indices)
    schema = "g4a_prediction_v1" if seed == 42 else "g4b_prediction_v1"
    expected = {
        "seed": seed,
        "holdout_site": holdout,
        "train_site": HOLDOUT_TO_TRAIN[holdout],
        "protocol": protocol,
        "view": view,
        "model_type": model,
    }
    model_complete = (root / "models" / HOLDOUT_TO_TRAIN[holdout] / f"seed{seed}" / view / model / "complete.json"
                      if seed == 42 else
                      root / "seeds" / f"seed{seed}" / "models" / HOLDOUT_TO_TRAIN[holdout] / view / model / "complete.json")
    model_complete.parent.mkdir(parents=True)
    model_complete.write_bytes(_canonical({"status": "COMPLETE"}))
    metadata = {
        "schema": schema,
        **expected,
        "score_type": "raw_ranking_score" if model == "lambdarank" else "positive_probability",
        "test_count": len(indices),
        "finite": True,
        "nonconstant": True,
        "prediction_sha256": _array_sha(scores),
        "test_indices_sha256": _array_sha(indices),
        "checkpoint_sha256": "c" * 64,
        "adapter_state_sha256": "d" * 64,
        "feature_ids_sha256": "e" * 64,
        "model_complete_sha256": _file_sha(model_complete),
        "run_identity_sha256": "1" * 64,
    }
    (unit / "metadata.json").write_bytes(_canonical(metadata))
    identity_keys = ("schema", "holdout_site", "train_site", "protocol", "view", "model_type", "seed",
                     "run_identity_sha256", "model_complete_sha256", "prediction_sha256", "test_indices_sha256")
    complete = {
        "schema": schema,
        "identity": {key: metadata[key] for key in identity_keys},
        "files": {name: _file_sha(unit / name) for name in ("metadata.json", "prediction.npz")},
    }
    (unit / "complete.json").write_bytes(_canonical(complete))
    row = {
        **expected,
        "prediction_reference": str(unit / "prediction.npz"),
        "prediction_length": len(indices),
        "prediction_sha256": _array_sha(scores),
        "test_indices_sha256": _array_sha(indices),
        "completion_sha256": _file_sha(unit / "complete.json"),
        "model_sha256": metadata["checkpoint_sha256"],
        "adapter_sha256": metadata["adapter_state_sha256"],
        "feature_order_sha256": metadata["feature_ids_sha256"],
        "score_type": metadata["score_type"],
    }
    return row, expected, scores, indices


def _synthetic_rows(source):
    return {
        (holdout, protocol, view, model, seed): {
            "source_version": source,
            "prediction_reference": f"/{source}/{holdout}/{protocol}/{view}/{model}/{seed}",
            "score_type": "lambdarank_raw" if model == "lambdarank" else "positive_probability",
            "bridge_prediction_sha256": f"p-{source}-{holdout}-{protocol}-{view}-{model}-{seed}",
            "bridge_test_indices_sha256": f"i-{holdout}",
            "source_identity_sha256": f"s-{source}-{holdout}-{protocol}-{view}-{model}-{seed}",
        }
        for holdout in HOLDOUTS
        for protocol in PROTOCOLS
        for view in VIEWS
        for model in MODEL_TYPES
        for seed in SEEDS
    }


class CompareE0E3Tests(unittest.TestCase):
    def test_fixed_experiment_sources(self):
        self.assertEqual(EXPERIMENT_SOURCES["E0"], {v: "old" for v in ("V1", "V2", "V3", "V4", "V5")})
        self.assertEqual(EXPERIMENT_SOURCES["E1"]["V5"], "old")
        self.assertEqual(EXPERIMENT_SOURCES["E2"]["V5"], "v2")
        self.assertEqual(EXPERIMENT_SOURCES["E3"]["V1"], "v2")
        self.assertEqual(len(fixed_column_keys()), 225)

    def test_score_type_bridge_is_explicit(self):
        self.assertEqual(normalize_score_type("raw_ranking_score"), "lambdarank_raw")
        self.assertEqual(normalize_score_type("lambdarank_raw"), "lambdarank_raw")
        with self.assertRaises(ValueError):
            normalize_score_type("unknown")

    def test_average_ties_then_mean(self):
        base = np.array([[0., 1., 0.], [1., 1., 2.], [2., 0., 2.]], dtype=np.float64)
        values = np.tile(base, 75)
        indices, fused, ranks = fuse_prediction_columns(values, np.array([4, 5, 6], dtype=np.int64))
        self.assertTrue(np.array_equal(indices, np.array([4, 5, 6])))
        expected = np.array([[1/6, 2/3, 1/6], [1/2, 2/3, 2/3], [5/6, 1/6, 2/3]])
        np.testing.assert_allclose(ranks[:, :3], expected)
        np.testing.assert_allclose(fused, expected.mean(axis=1))

    def test_seed42_legacy_schema_is_accepted_without_source_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit, expected, scores, indices = _write_v2_unit(tmp, seed=42)
            row = validate_v2_prediction_unit(unit, expected, seed42_legacy=True, approved_root=Path(tmp))
            self.assertEqual(row["source_version"], "v2")
            self.assertEqual(row["bridge_prediction_sha256"], _array_sha(scores))
            self.assertEqual(row["bridge_test_indices_sha256"], _array_sha(indices))

    def test_new_seed_requires_source_version_and_exact_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit, expected, _, _ = _write_v2_unit(tmp, seed=43)
            metadata_path = unit / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata.pop("source_version")
            metadata_path.write_bytes(_canonical(metadata))
            complete = json.loads((unit / "complete.json").read_text())
            complete["file_summaries"] = {p.name: _file_sha(p) for p in sorted(unit.iterdir()) if p.name != "complete.json"}
            (unit / "complete.json").write_bytes(_canonical(complete))
            with self.assertRaisesRegex(ValueError, "source_version"):
                validate_v2_prediction_unit(unit, expected, seed42_legacy=False, approved_root=Path(tmp))

            metadata["source_version"] = "v2"
            metadata["view"] = "V2"
            metadata_path.write_bytes(_canonical(metadata))
            complete["file_summaries"] = {p.name: _file_sha(p) for p in sorted(unit.iterdir()) if p.name != "complete.json"}
            (unit / "complete.json").write_bytes(_canonical(complete))
            with self.assertRaisesRegex(ValueError, "view"):
                validate_v2_prediction_unit(unit, expected, seed42_legacy=False, approved_root=Path(tmp))

    def test_v2_unit_rejects_payload_or_completion_digest_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit, expected, _, _ = _write_v2_unit(tmp, seed=42)
            metadata = json.loads((unit / "metadata.json").read_text())
            metadata["prediction_sha256"] = "0" * 64
            (unit / "metadata.json").write_bytes(_canonical(metadata))
            complete = json.loads((unit / "complete.json").read_text())
            complete["file_summaries"]["metadata.json"] = _file_sha(unit / "metadata.json")
            (unit / "complete.json").write_bytes(_canonical(complete))
            with self.assertRaisesRegex(ValueError, "prediction.*SHA256"):
                validate_v2_prediction_unit(unit, expected, seed42_legacy=True, approved_root=Path(tmp))

    def test_v2_unit_requires_actual_run_and_model_completion_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit, expected, _, _ = _write_v2_unit(tmp, seed=42)
            model_complete = (Path(tmp) / "models" / expected["train_site"] / "seed42" /
                              expected["view"] / expected["model_type"] / "complete.json")
            model_complete.unlink()
            with self.assertRaisesRegex(ValueError, "model completion"):
                validate_v2_prediction_unit(unit, expected, seed42_legacy=True, approved_root=Path(tmp))

    def test_prediction_row_rechecks_frozen_payload_after_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit, expected, _, indices = _write_v2_unit(tmp, seed=42)
            row = validate_v2_prediction_unit(unit, expected, seed42_legacy=True, approved_root=Path(tmp))
            np.savez_compressed(unit / "prediction.npz", scores=np.array([0.2, 0.8, 0.3]),
                                test_global_indices=indices)
            with self.assertRaisesRegex(ValueError, "file SHA256 drift"):
                load_prediction_row(row)

    def test_old_unit_validates_original_completion_and_trace_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            row, expected, scores, indices = _write_old_unit(tmp)
            validated = validate_old_prediction_row(row, expected, approved_roots=(Path(tmp),))
            self.assertEqual(validated["source_version"], "old")
            self.assertEqual(validated["bridge_prediction_sha256"], _array_sha(scores))
            self.assertEqual(validated["bridge_test_indices_sha256"], _array_sha(indices))
            row["adapter_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "adapter"):
                validate_old_prediction_row(row, expected, approved_roots=(Path(tmp),))

    def test_old_multiseed_model_trace_uses_frozen_seed_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            row, expected, _, _ = _write_old_unit(tmp, seed=43)
            validated = validate_old_prediction_row(row, expected, approved_roots=(Path(tmp),))
            self.assertEqual(validated["source_version"], "old")

    def test_old_unit_requires_actual_model_completion_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            row, expected, _, _ = _write_old_unit(tmp)
            model_complete = (Path(tmp) / "models" / expected["train_site"] / "seed42" /
                              expected["view"] / expected["model_type"] / "complete.json")
            model_complete.unlink()
            with self.assertRaisesRegex(ValueError, "model completion"):
                validate_old_prediction_row(row, expected, approved_roots=(Path(tmp),))

    def test_official_scoring_file_requires_frozen_path_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "official.py"
            path.write_text("VALUE = 1\n", encoding="utf-8")
            digest = _file_sha(path)
            self.assertEqual(verify_official_scoring_module(path, digest), path.resolve())
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_official_scoring_module(path, "0" * 64)

    def test_all_reference_entries_close_5400_and_24_groups(self):
        rows = build_all_reference_entries(_synthetic_rows("old"), _synthetic_rows("v2"))
        self.assertEqual(len(rows), 5400)
        groups = group_reference_entries(rows)
        self.assertEqual(len(groups), 24)
        self.assertEqual({len(group) for group in groups.values()}, {225})
        self.assertTrue(all(row["source_version"] == "old" for row in groups[("E0", "I0002", "transductive")]))
        self.assertEqual({row["source_version"] for row in groups[("E3", "S0001", "inductive")]}, {"v2"})

    def test_reference_builder_rejects_missing_identity(self):
        old = _synthetic_rows("old")
        old.pop(next(iter(old)))
        with self.assertRaisesRegex(ValueError, "coverage"):
            build_all_reference_entries(old, _synthetic_rows("v2"))

    def test_fusion_group_rejects_mixed_group_identity_before_io(self):
        rows = []
        for seed, view, model in fixed_column_keys():
            rows.append({
                "experiment": "E0",
                "holdout_site": "I0002",
                "protocol": "transductive",
                "seed": seed,
                "view": view,
                "model_type": model,
                "prediction_reference": "/does/not/matter",
                "score_type": "positive_probability",
            })
        rows[-1] = {**rows[-1], "protocol": "inductive"}
        with self.assertRaisesRegex(ValueError, "group identity"):
            fuse_reference_group(rows)

    def test_e0_regression_compares_all_four_arrays_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = {}
            for holdout in HOLDOUTS:
                for protocol in PROTOCOLS:
                    model_keys = np.array(["seed42|V1|lr"], dtype="U32")
                    indices = np.array([2, 7], dtype=np.int64)
                    ranks = np.array([[0.25], [0.75]], dtype=np.float64)
                    fused = ranks[:, 0].copy()
                    path = root / "groups" / f"{holdout}_{protocol}"
                    path.mkdir(parents=True)
                    np.savez_compressed(path / "fusion.npz", model_keys=model_keys,
                                        test_global_indices=indices, model_ranks=ranks,
                                        ensemble_scores=fused)
                    groups[(holdout, protocol)] = {
                        "model_keys": model_keys,
                        "test_global_indices": indices,
                        "model_ranks": ranks,
                        "ensemble_scores": fused,
                    }
            summary = assert_e0_fusion_regression(groups, root)
            self.assertEqual(summary["groups"], 6)
            groups[("I0002", "transductive")]["ensemble_scores"] = np.array([0.25, 0.70])
            with self.assertRaisesRegex(ValueError, "ensemble_scores"):
                assert_e0_fusion_regression(groups, root)


if __name__ == "__main__":
    unittest.main()
