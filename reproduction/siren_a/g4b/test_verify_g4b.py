import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproduction.siren_a.g4b.verify_g4b import (
    rebuild_query_expected,
    verify_adapter_arrays,
    verify_file_hash,
    verify_query_arrays,
)


class IndependentQueryVerificationTests(unittest.TestCase):
    def setUp(self):
        self.ages = np.array([61.0, 56.0, 62.0, 57.0, 61.5, 56.5], dtype=np.float64)
        self.labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int8)
        self.local = np.array([9, 3, 12, 5, 14, 7], dtype=np.int64)
        self.global_indices = np.array([109, 103, 112, 105, 114, 107], dtype=np.int64)
        self.expected = rebuild_query_expected(
            self.ages, self.labels, self.local, self.global_indices
        )

    def test_independent_query_has_literal_stable_order_and_groups(self):
        arrays = self.expected["arrays"]
        np.testing.assert_array_equal(arrays["permutation"], [1, 3, 5, 0, 2, 4])
        np.testing.assert_array_equal(arrays["inverse_permutation"], [3, 0, 4, 1, 5, 2])
        np.testing.assert_array_equal(arrays["sorted_local_positions"], [3, 5, 7, 9, 12, 14])
        np.testing.assert_array_equal(arrays["sorted_global_indices"], [103, 105, 107, 109, 112, 114])
        np.testing.assert_array_equal(arrays["sorted_group_ids"], [11, 11, 11, 12, 12, 12])
        np.testing.assert_array_equal(arrays["sorted_labels"], [0, 1, 1, 0, 1, 0])
        np.testing.assert_array_equal(arrays["group_ids"], [11, 12])
        np.testing.assert_array_equal(arrays["group_sizes"], [3, 3])
        np.testing.assert_array_equal(arrays["group_positive"], [2, 1])
        np.testing.assert_array_equal(arrays["group_negative"], [1, 2])
        self.assertEqual(self.expected["audit"]["truncation_level"], 3)

    def test_every_query_array_mutation_is_rejected(self):
        for key in self.expected["arrays"]:
            with self.subTest(key=key):
                arrays = {name: np.array(value, copy=True) for name, value in self.expected["arrays"].items()}
                arrays[key] = arrays[key].copy()
                arrays[key].flat[0] += 1
                with self.assertRaises(ValueError):
                    verify_query_arrays(
                        arrays,
                        self.expected["audit"],
                        self.ages,
                        self.labels,
                        self.local,
                        self.global_indices,
                        sample_sha256="sample",
                    )

    def test_wrong_dtype_fingerprint_coverage_and_single_class_are_rejected(self):
        arrays = {name: np.array(value, copy=True) for name, value in self.expected["arrays"].items()}
        arrays["permutation"] = arrays["permutation"].astype(np.int32)
        with self.assertRaises(ValueError):
            verify_query_arrays(arrays, self.expected["audit"], self.ages, self.labels,
                                self.local, self.global_indices, sample_sha256="sample")
        for field in ("query_fingerprint", "maximum_group_size", "truncation_level",
                      "all_valid_pairs", "within_group_valid_pairs", "pair_coverage",
                      "sample_sha256"):
            with self.subTest(field=field):
                audit = copy.deepcopy(self.expected["audit"])
                audit["sample_sha256"] = "sample"
                audit[field] = "bad" if field.endswith("sha256") or "fingerprint" in field else -1
                with self.assertRaises(ValueError):
                    verify_query_arrays(self.expected["arrays"], audit, self.ages, self.labels,
                                        self.local, self.global_indices, sample_sha256="sample")
        with self.assertRaises(ValueError):
            rebuild_query_expected(self.ages, np.zeros(6, dtype=np.int8), self.local, self.global_indices)


class IndependentAdapterVerificationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [[np.nan, 1.0, np.nan], [2.0, 3.0, np.nan], [4.0, np.nan, np.nan]],
            dtype=np.float64,
        )
        self.ids = ("a", "b", "c")
        self.indices = np.array([10, 20, 30], dtype=np.int64)
        feature_sha = hashlib.sha256("a\nb\nc".encode("utf-8")).hexdigest()
        index_sha = hashlib.sha256()
        index_sha.update(str(self.indices.shape).encode("ascii"))
        index_sha.update(self.indices.tobytes(order="C"))
        self.base = {
            "model_type": np.asarray("lr"),
            "feature_ids": np.asarray(self.ids, dtype="U"),
            "feature_ids_sha256": np.asarray(feature_sha),
            "fit_indices_sha256": np.asarray(index_sha.hexdigest()),
            "fit_scope": np.asarray("seed_subset"),
            "finite_counts": np.array([2, 2, 0], dtype=np.int64),
            "fill_values": np.array([3.0, 2.0, 128.0 / 255.0], dtype=np.float64),
            "all_missing": np.array([False, False, True], dtype=bool),
        }

    def test_lr_adapter_is_recomputed_from_selected_rows(self):
        result = verify_adapter_arrays(self.base, "lr", self.X, self.ids, self.indices)
        np.testing.assert_array_equal(result["finite_counts"], [2, 2, 0])
        np.testing.assert_allclose(result["fill_values"], [3.0, 2.0, 128.0 / 255.0], rtol=0, atol=0)
        self.assertFalse(np.isnan(result["transformed"]).any())
        np.testing.assert_array_equal(result["transformed"][np.isfinite(self.X)], self.X[np.isfinite(self.X)])

    def test_adapter_mutations_are_rejected(self):
        mutations = {
            "fit_scope": np.asarray("full_train_preview"),
            "feature_ids_sha256": np.asarray("bad"),
            "fit_indices_sha256": np.asarray("bad"),
            "finite_counts": np.array([3, 2, 0], dtype=np.int64),
            "fill_values": np.array([2.5, 2.0, 128.0 / 255.0], dtype=np.float64),
            "all_missing": np.array([False, False, False], dtype=bool),
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                saved = {name: np.array(item, copy=True) for name, item in self.base.items()}
                saved[key] = value
                with self.assertRaises(ValueError):
                    verify_adapter_arrays(saved, "lr", self.X, self.ids, self.indices)

    def test_native_nan_adapters_are_exact_passthrough(self):
        for model_type in ("hgb", "lambdarank"):
            saved = {name: np.array(item, copy=True) for name, item in self.base.items()}
            saved["model_type"] = np.asarray(model_type)
            saved["fill_values"] = np.full(3, np.nan, dtype=np.float64)
            result = verify_adapter_arrays(saved, model_type, self.X, self.ids, self.indices)
            self.assertTrue(np.array_equal(result["transformed"], self.X, equal_nan=True))
            bad = {name: np.array(item, copy=True) for name, item in saved.items()}
            bad["fill_values"][0] = 0.5
            with self.assertRaises(ValueError):
                verify_adapter_arrays(bad, model_type, self.X, self.ids, self.indices)


class ArtifactCorruptionTests(unittest.TestCase):
    def test_corrupted_or_symlinked_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.joblib"
            artifact.write_bytes(b"trusted-checkpoint")
            expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.assertTrue(verify_file_hash(artifact, expected))
            artifact.write_bytes(b"corrupted-checkpoint")
            with self.assertRaises(ValueError):
                verify_file_hash(artifact, expected)
            artifact.unlink()
            target = root / "target.bin"
            target.write_bytes(b"trusted-checkpoint")
            artifact.symlink_to(target)
            with self.assertRaises(ValueError):
                verify_file_hash(artifact, expected)


if __name__ == "__main__":
    unittest.main()
