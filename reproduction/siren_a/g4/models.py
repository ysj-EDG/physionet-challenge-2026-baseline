"""Frozen estimator factories, adapters, prediction semantics and trusted checkpoints."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Tuple

import joblib
import lightgbm
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from reproduction.siren_a.pipeline.adapters import AdapterState, fit_adapter, transform_adapter
from .config import HGB_PARAMS, LGBM_PARAMS, LR_PARAMS
from .queries import QueryResult


@dataclass(frozen=True)
class TrainedModel:
    model_type: str
    model: object
    adapter: AdapterState
    parameters: Mapping[str, object]
    train_matrix: np.ndarray
    train_score: np.ndarray


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_to_npz(path: Path, state: AdapterState) -> None:
    np.savez(path, model_type=np.asarray(state.model_type),
             feature_ids=np.asarray(state.feature_ids, dtype="U"),
             feature_ids_sha256=np.asarray(state.feature_ids_sha256),
             fit_indices_sha256=np.asarray(state.fit_indices_sha256),
             fit_scope=np.asarray(state.fit_scope), finite_counts=state.finite_counts,
             fill_values=state.fill_values, all_missing=state.all_missing)


def _adapter_from_npz(path: Path) -> AdapterState:
    with np.load(path, allow_pickle=False) as archive:
        scalar = lambda key: str(np.asarray(archive[key]).item())
        return AdapterState(scalar("model_type"), tuple(np.asarray(archive["feature_ids"]).tolist()),
                            scalar("feature_ids_sha256"), scalar("fit_indices_sha256"),
                            scalar("fit_scope"), np.asarray(archive["finite_counts"]),
                            np.asarray(archive["fill_values"]), np.asarray(archive["all_missing"]))


def fit_frozen_model(model_type: str, X: np.ndarray, y: np.ndarray, feature_ids: Tuple[str, ...],
                     global_indices: np.ndarray, query: QueryResult) -> TrainedModel:
    X = np.asarray(X)
    y = np.asarray(y)
    if model_type not in {"lr", "hgb", "lambdarank"}:
        raise ValueError("unknown model type")
    if X.ndim != 2 or X.dtype != np.float64 or y.shape != (len(X),) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("training schema invalid")
    if len(np.unique(y)) != 2:
        raise ValueError("training labels need both classes")
    adapter = fit_adapter(X, model_type, feature_ids,
                          fit_indices=np.asarray(global_indices, dtype=np.int64),
                          fit_scope="seed_subset")
    transformed = transform_adapter(X, adapter)
    if model_type == "lr":
        params = dict(LR_PARAMS)
        model = LogisticRegression(**params)
        model.fit(transformed, y)
    elif model_type == "hgb":
        params = dict(HGB_PARAMS)
        model = HistGradientBoostingClassifier(**params)
        model.fit(transformed, y)
    else:
        if len(query.permutation) != len(X):
            raise ValueError("query length mismatch")
        params = dict(LGBM_PARAMS)
        params["lambdarank_truncation_level"] = query.truncation_level
        model = lightgbm.LGBMRanker(**params)
        model.fit(transformed[query.permutation], y[query.permutation],
                  group=query.group_sizes.tolist())
    score = predict_scores(model_type, model, adapter, X)
    if not np.isfinite(score).all() or np.ptp(score) == 0:
        raise ValueError("training prediction invalid or constant")
    return TrainedModel(model_type, model, adapter, params, transformed, score)


def predict_scores(model_type: str, model: object, adapter: AdapterState, X: np.ndarray) -> np.ndarray:
    values = transform_adapter(np.asarray(X), adapter)
    if model_type in {"lr", "hgb"}:
        classes = np.asarray(model.classes_)
        matches = np.flatnonzero(classes == 1)
        if len(matches) != 1:
            raise ValueError("positive class unavailable")
        score = np.asarray(model.predict_proba(values)[:, int(matches[0])], dtype=np.float64)
        if np.any(score < 0) or np.any(score > 1):
            raise ValueError("probability outside range")
    elif model_type == "lambdarank":
        score = np.asarray(model.predict(values), dtype=np.float64)
    else:
        raise ValueError("unknown model type")
    if score.shape != (len(values),) or not np.isfinite(score).all():
        raise ValueError("prediction invalid")
    return score


def save_checkpoint(path: Path, trained: TrainedModel) -> Mapping[str, str]:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model_name = "model.txt" if trained.model_type == "lambdarank" else "model.joblib"
    model_path = path / model_name
    if model_path.exists() or (path / "adapter_state.npz").exists():
        raise FileExistsError(path)
    if trained.model_type == "lambdarank":
        trained.model.booster_.save_model(str(model_path))
    else:
        joblib.dump(trained.model, model_path)
    _adapter_to_npz(path / "adapter_state.npz", trained.adapter)
    return {model_name: _sha(model_path), "adapter_state.npz": _sha(path / "adapter_state.npz")}


def load_checkpoint(path: Path, model_type: str, files: Mapping[str, str]) -> TrainedModel:
    path = Path(path)
    model_name = "model.txt" if model_type == "lambdarank" else "model.joblib"
    for name, digest in files.items():
        if not (path / name).is_file() or _sha(path / name) != digest:
            raise ValueError("checkpoint hash mismatch")
    model = (lightgbm.Booster(model_file=str(path / model_name))
             if model_type == "lambdarank" else joblib.load(path / model_name))
    adapter = _adapter_from_npz(path / "adapter_state.npz")
    params = dict(LGBM_PARAMS if model_type == "lambdarank"
                  else LR_PARAMS if model_type == "lr" else HGB_PARAMS)
    return TrainedModel(model_type, model, adapter, params,
                        np.empty((0, len(adapter.feature_ids))), np.empty(0))
