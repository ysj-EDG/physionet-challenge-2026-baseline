"""Seed-aware thin model wrapper over the frozen G4-A model contract."""

from typing import Tuple

import lightgbm
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from reproduction.siren_a.g4.models import TrainedModel, predict_scores
from reproduction.siren_a.g4.queries import QueryResult
from reproduction.siren_a.pipeline.adapters import fit_adapter, transform_adapter

from .config import parameters_for_seed


def fit_seed_model(
    model_type: str,
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    feature_ids: Tuple[str, ...],
    global_indices: np.ndarray,
    query: QueryResult,
) -> TrainedModel:
    values = np.asarray(X)
    labels = np.asarray(y)
    indices = np.asarray(global_indices)
    if model_type not in {"lr", "hgb", "lambdarank"}:
        raise ValueError("unknown model type")
    if values.ndim != 2 or values.dtype != np.float64:
        raise ValueError("training matrix must be float64 and two-dimensional")
    if labels.shape != (len(values),) or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("training labels invalid")
    if indices.shape != (len(values),) or indices.dtype.kind not in "iu":
        raise ValueError("training indices invalid")
    if len(np.unique(labels)) != 2:
        raise ValueError("training labels need both classes")
    if len(feature_ids) != values.shape[1] or len(set(feature_ids)) != len(feature_ids):
        raise ValueError("feature order invalid")
    if len(query.permutation) != len(values):
        raise ValueError("query length mismatch")

    parameters = parameters_for_seed(model_type, seed, int(query.truncation_level))
    adapter = fit_adapter(
        values,
        model_type,
        feature_ids,
        fit_indices=indices.astype(np.int64, copy=False),
        fit_scope="seed_subset",
    )
    transformed = transform_adapter(values, adapter)
    if model_type == "lr":
        model = LogisticRegression(**parameters)
        model.fit(transformed, labels)
    elif model_type == "hgb":
        model = HistGradientBoostingClassifier(**parameters)
        model.fit(transformed, labels)
    else:
        model = lightgbm.LGBMRanker(**parameters)
        model.fit(
            transformed[query.permutation],
            labels[query.permutation],
            group=query.group_sizes.tolist(),
        )
    score = predict_scores(model_type, model, adapter, values)
    if not np.isfinite(score).all() or np.ptp(score) == 0:
        raise ValueError("training prediction invalid or constant")
    return TrainedModel(model_type, model, adapter, parameters, transformed, score)


def verify_runtime_parameters(model_type: str, model: object, expected: dict) -> bool:
    """Check parameters exposed by the fitted estimator, allowing Booster aliases later."""

    if model_type in {"lr", "hgb"}:
        observed = model.get_params(deep=False)
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ValueError(f"runtime parameter mismatch: {key}")
    elif model_type == "lambdarank":
        observed = model.get_params(deep=False)
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ValueError(f"runtime parameter mismatch: {key}")
    else:
        raise ValueError("unknown model type")
    return True
