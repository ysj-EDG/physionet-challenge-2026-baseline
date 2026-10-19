from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from extensions.common import (
    STAGE_CODES,
    adjacent_pairs,
    linear_quantile,
    pool_mean_iqr,
    relative_eeg_power,
    stage_mask,
)
from extensions.contracts import FamilyInputContract, FieldRequirement
from extensions.recipe import RecipeDependencyError, build_recipe_identity
from extensions.registry import FamilySpec, FeatureDefinition, list_families
from extensions.tests.fixtures import write_recipe_tree


def _contract() -> FamilyInputContract:
    return FamilyInputContract(
        family_id="dummy",
        contract_version="source_v1",
        requirements=(
            FieldRequirement("X_seq", ("T", 2), ("float32",), columns=(0, 1)),
            FieldRequirement("stage_valid", ("T",), ("bool",)),
        ),
        stage_time_rules=("strict_30_second_grid",),
        validity_rules=("stage_valid & finite",),
    )


def _feature(source_columns: tuple[int, ...] = (0,)) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id="dummy.feature_0",
        source_field="X_seq",
        source_columns=source_columns,
        formula_id="mean_iqr_v1",
        parameters={"minimum_support": 5},
        validity_expression="stage_valid & finite",
        minimum_support=5,
        unit="ratio",
    )


def _spec(**changes) -> FamilySpec:
    base = FamilySpec(
        family_id="dummy",
        family_version="1.0.0",
        extractor_symbol="pkg.family.extract_family",
        default_config={"alpha": 1, "nested": {"right": 2, "left": 1}},
        input_contract=_contract(),
        feature_definitions=(_feature(),),
        numerical_symbols=(
            "pkg.family.extract_family",
            "pkg.source.materialize",
            "pkg.common.stage_valid",
            "pkg.common.adjacency",
            "pkg.common.aggregate",
        ),
        numerical_constants=("pkg.common.SCALE",),
        numerical_dependencies={"numpy": np.__version__},
    )
    return replace(base, **changes)


class CommonHelperTests(unittest.TestCase):
    def test_fixed_stage_mapping_and_validity(self) -> None:
        self.assertEqual(STAGE_CODES["N3"], (1,))
        self.assertEqual(STAGE_CODES["NREM"], (1, 2, 3))
        self.assertEqual(STAGE_CODES["Sleep"], (1, 2, 3, 4))
        codes = np.asarray([1.0, 2.0, 4.0, 5.0, np.nan])
        valid = np.asarray([True, False, True, True, True])
        np.testing.assert_array_equal(
            stage_mask(codes, valid, "Sleep"),
            np.asarray([True, False, True, False, False]),
        )

    def test_adjacency_requires_two_valid_epochs_and_exactly_30_seconds(self) -> None:
        starts = np.asarray([0.0, 30.0, 61.0, 91.0, 121.0])
        valid = np.asarray([True, True, True, False, True])
        np.testing.assert_array_equal(
            adjacent_pairs(starts, valid),
            np.asarray([True, False, False, False]),
        )

    def test_linear_quantile_matches_numpy_and_zero_is_lawful(self) -> None:
        values = np.asarray([0.0, 1.0, 4.0, 9.0])
        self.assertEqual(
            linear_quantile(values, 0.88),
            float(np.quantile(values, 0.88, method="linear")),
        )
        np.testing.assert_array_equal(
            relative_eeg_power(
                np.asarray([0.0, 2.0]),
                np.asarray([0.0, 3.0]),
                np.asarray([0.0, 1.0]),
            ),
            np.asarray([0.0, 4.0]),
        )
        mean, iqr, support = pool_mean_iqr(
            np.asarray([0.0, 0.0, np.nan]),
            np.asarray([True, True, False]),
            minimum_support=2,
        )
        self.assertEqual((mean, iqr, support), (0.0, 0.0, 2))

    def test_pooling_returns_nan_without_minimum_support(self) -> None:
        mean, iqr, support = pool_mean_iqr(
            np.asarray([0.0, np.nan]),
            np.asarray([True, True]),
            minimum_support=2,
        )
        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(iqr))
        self.assertEqual(support, 1)


class RecipeTests(unittest.TestCase):
    def _identity(self, root: Path, spec: FamilySpec | None = None, config=None):
        return build_recipe_identity(spec or _spec(), config or {}, root)

    def test_registry_starts_empty_of_real_feature_families(self) -> None:
        self.assertEqual(list_families(), ())

    def test_parameter_order_is_canonical_and_feature_order_is_material(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_recipe_tree(Path(directory))
            left = self._identity(root, config={"beta": 2, "alpha": 3})
            right = self._identity(root, config={"alpha": 3, "beta": 2})
            self.assertEqual(left.recipe_id, right.recipe_id)

            second = replace(_feature(), feature_id="dummy.feature_1", source_columns=(1,))
            forward = self._identity(
                root, replace(_spec(), feature_definitions=(_feature(), second))
            )
            reverse = self._identity(
                root, replace(_spec(), feature_definitions=(second, _feature()))
            )
            self.assertNotEqual(forward.recipe_id, reverse.recipe_id)

    def test_each_declared_numerical_dependency_invalidates_recipe(self) -> None:
        mutations = {
            "family extractor": ("pkg/family.py", "return aggregate(values) * SCALE", "return aggregate(values) * SCALE + 1"),
            "source materializer": ("pkg/source.py", "return source", "return (source, )"),
            "stage validity": ("pkg/common.py", "return values > 0", "return values >= 0"),
            "adjacency": ("pkg/common.py", "== 30", "== 60"),
            "aggregation": ("pkg/common.py", "return sum(values) / len(values)", "return max(values)"),
            "constant": ("pkg/common.py", "SCALE = 2.0", "SCALE = 3.0"),
        }
        for label, (relative, old, new) in mutations.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                root = write_recipe_tree(Path(directory))
                before = self._identity(root).recipe_id
                path = root / relative
                path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
                self.assertNotEqual(before, self._identity(root).recipe_id)

    def test_contract_columns_numpy_version_and_config_invalidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_recipe_tree(Path(directory))
            baseline = self._identity(root)
            changed_validity = replace(
                _spec(),
                input_contract=replace(
                    _contract(), validity_rules=("stage_valid & finite & quality",)
                ),
            )
            changed_column = replace(
                _spec(), feature_definitions=(_feature(source_columns=(1,)),)
            )
            self.assertNotEqual(baseline.recipe_id, self._identity(root, changed_validity).recipe_id)
            self.assertNotEqual(baseline.recipe_id, self._identity(root, changed_column).recipe_id)
            self.assertNotEqual(
                baseline.recipe_id,
                self._identity(root, config={"alpha": 99}).recipe_id,
            )
            with mock.patch("extensions.recipe.np.__version__", "999.0"):
                self.assertNotEqual(baseline.recipe_id, self._identity(root).recipe_id)

    def test_unrelated_changes_and_git_head_do_not_invalidate(self) -> None:
        unrelated = (
            "cache.py",
            "record_key.py",
            "README.md",
            "test_family.py",
            "pkg/other_family.py",
        )
        for relative in unrelated:
            with self.subTest(relative), tempfile.TemporaryDirectory() as directory:
                root = write_recipe_tree(Path(directory))
                before = self._identity(root).recipe_id
                path = root / relative
                path.write_text(path.read_text(encoding="utf-8") + "# unrelated\n", encoding="utf-8")
                self.assertEqual(before, self._identity(root).recipe_id)

        with tempfile.TemporaryDirectory() as directory:
            root = write_recipe_tree(Path(directory))
            common = root / "pkg/common.py"
            before = self._identity(root).recipe_id
            common.write_text(
                common.read_text(encoding="utf-8").replace(
                    "return values\n", "return values + 100\n"
                ),
                encoding="utf-8",
            )
            self.assertEqual(before, self._identity(root).recipe_id)

            with mock.patch("extensions.recipe._git_head", side_effect=("aaa", "bbb")):
                first = self._identity(root)
                second = self._identity(root)
            self.assertEqual(first.recipe_id, second.recipe_id)
            self.assertNotEqual(first.provenance["git_commit"], second.provenance["git_commit"])

    def test_comments_and_docstrings_do_not_invalidate_ast_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_recipe_tree(Path(directory))
            before = self._identity(root).recipe_id
            family = root / "pkg/family.py"
            family.write_text(
                family.read_text(encoding="utf-8").replace(
                    "def extract_family(values):\n",
                    'def extract_family(values):\n    """changed documentation"""\n    # comment\n',
                ),
                encoding="utf-8",
            )
            self.assertEqual(before, self._identity(root).recipe_id)

    def test_unresolved_symbols_constants_and_undeclared_helpers_fail(self) -> None:
        cases = (
            replace(_spec(), numerical_symbols=("pkg.missing.nope",)),
            replace(_spec(), numerical_constants=("pkg.common.MISSING",)),
            replace(
                _spec(),
                numerical_symbols=tuple(
                    symbol
                    for symbol in _spec().numerical_symbols
                    if symbol != "pkg.common.aggregate"
                ),
            ),
            replace(_spec(), numerical_constants=()),
        )
        for spec in cases:
            with self.subTest(spec=spec), tempfile.TemporaryDirectory() as directory:
                root = write_recipe_tree(Path(directory))
                with self.assertRaises(RecipeDependencyError):
                    self._identity(root, spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
