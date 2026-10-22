"""Seed-aware v2 training adapters for the authorized seeds 43--56."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from reproduction.siren_a.g4b.models import fit_seed_model, verify_runtime_parameters
from reproduction.siren_a.g4b.config import parameters_for_seed
from reproduction.siren_a.g4.queries import QueryResult, build_queries
from reproduction.siren_a.g4.sampler import SampleResult, sample_training_rows
from reproduction.siren_a.g4.models import TrainedModel

from .train_inputs_v2 import V2FoldInputs

SEEDS = tuple(range(43, 57))
MODEL_TYPES = ("lr", "hgb", "lambdarank")


@dataclass(frozen=True)
class MultiSeedTraining:
    fold: V2FoldInputs
    seed: int
    sample: SampleResult
    query: QueryResult

    @property
    def X_sample(self) -> np.ndarray:
        return self.fold.X_train[self.sample.local_positions]

    @property
    def y_sample(self) -> np.ndarray:
        return self.fold.y_train[self.sample.local_positions]

    @property
    def age_sample(self) -> np.ndarray:
        return self.fold.age_train[self.sample.local_positions]


def prepare_training_v2(fold: V2FoldInputs, seed: int) -> MultiSeedTraining:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) not in SEEDS:
        raise ValueError("v2 multi-seed training accepts only seeds 43..56")
    seed = int(seed)
    sample = sample_training_rows(fold.train_site, seed, fold.train_age_groups,
                                  fold.y_train, fold.train_indices)
    query = build_queries(fold.age_train[sample.local_positions],
                          fold.y_train[sample.local_positions],
                          sample.local_positions, sample.global_indices)
    return MultiSeedTraining(fold=fold, seed=seed, sample=sample, query=query)


def fit_multiseed_model_v2(model_type: str, prepared: MultiSeedTraining) -> TrainedModel:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model type: {model_type}")
    trained = fit_seed_model(model_type, prepared.seed, prepared.X_sample,
                             prepared.y_sample, prepared.fold.feature_ids,
                             prepared.sample.global_indices, prepared.query)
    expected = parameters_for_seed(model_type, prepared.seed, prepared.query.truncation_level)
    verify_runtime_parameters(model_type, trained.model, expected)
    if dict(trained.parameters) != dict(expected):
        raise ValueError("returned estimator parameters differ from frozen seed parameters")
    return trained


def compare_training_roles_v2(first: MultiSeedTraining, second: MultiSeedTraining) -> bool:
    """Check that two protocols or holdout aliases share the exact training role."""
    if first.seed != second.seed or first.fold.train_site != second.fold.train_site:
        raise ValueError("training role seed/site mismatch")
    for name in ("X_train", "train_valid_mask", "y_train", "age_train", "train_age_groups", "train_indices"):
        if not np.array_equal(getattr(first.fold, name), getattr(second.fold, name), equal_nan=True):
            raise ValueError(f"training role mismatch: {name}")
    if first.fold.feature_ids != second.fold.feature_ids:
        raise ValueError("training feature identity mismatch")
    if not np.array_equal(first.sample.local_positions, second.sample.local_positions):
        raise ValueError("sample identity mismatch")
    if first.query.fingerprint != second.query.fingerprint:
        raise ValueError("query identity mismatch")
    return True


__all__ = ["MODEL_TYPES", "MultiSeedTraining", "SEEDS", "compare_training_roles_v2",
           "fit_multiseed_model_v2", "prepare_training_v2"]
