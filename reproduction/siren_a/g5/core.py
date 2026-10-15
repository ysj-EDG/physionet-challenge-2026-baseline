"""Minimal, label-safe fixed rank ensemble for Branch A G5."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from reproduction.siren_a.pipeline.label_join import MetadataIndex, parse_age, parse_label


HOLDOUT_TO_TRAIN = {"I0002": "S0001", "I0006": "S0001", "S0001": "I0006"}
HOLDOUTS = ("I0002", "I0006", "S0001")
PROTOCOLS = ("transductive", "inductive")
VIEWS = ("V1", "V2", "V3", "V4", "V5")
MODEL_TYPES = ("lr", "hgb", "lambdarank")
SEEDS = tuple(range(42, 57))


@dataclass(frozen=True)
class PredictionGroup:
    holdout_site: str
    protocol: str
    model_keys: tuple[str, ...]
    test_indices: np.ndarray
    scores: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def mean_ranks(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores)
    if values.ndim != 2 or values.dtype.kind not in "f" or not np.isfinite(values).all():
        raise ValueError("scores must be a finite two-dimensional floating array")
    n_rows, n_models = values.shape
    if n_rows == 0 or n_models == 0:
        raise ValueError("scores may not be empty")
    ranks = np.empty(values.shape, dtype=np.float64)
    for column in range(n_models):
        _, inverse, counts = np.unique(values[:, column], return_inverse=True, return_counts=True)
        less = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts[:-1])))
        ranks[:, column] = (less[inverse] + 0.5 * counts[inverse]) / n_rows
    return ranks, ranks.mean(axis=1, dtype=np.float64)


def _inside_approved(path: Path, approved_roots: Sequence[Path]) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"prediction path is missing or symlinked: {path}")
    resolved = path.resolve()
    for root_value in approved_roots:
        root = Path(root_value).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"symlinked prediction component: {cursor}")
        return resolved
    raise ValueError(f"prediction path outside approved roots: {path}")


def _json_entries(path: Path) -> list[Mapping[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"manifest entries missing: {path}")
    return entries


def _prediction_key(row: Mapping[str, object]) -> tuple[str, str, str, str, int]:
    return (str(row["holdout_site"]), str(row["protocol"]), str(row["view"]),
            str(row["model_type"]), int(row["seed"]))


def _model_key(row: Mapping[str, object]) -> tuple[str, int, str, str]:
    return str(row["train_site"]), int(row["seed"]), str(row["view"]), str(row["model_type"])


def load_predictions(
    predictions_manifest: Path,
    models_manifest: Path,
    site_indices: Mapping[str, np.ndarray],
    approved_roots: Sequence[Path],
    *,
    holdouts: Sequence[str] = HOLDOUTS,
    protocols: Sequence[str] = PROTOCOLS,
    views: Sequence[str] = VIEWS,
    model_types: Sequence[str] = MODEL_TYPES,
    seeds: Sequence[int] = SEEDS,
) -> tuple[dict[tuple[str, str], PredictionGroup], Mapping[str, object]]:
    prediction_rows = _json_entries(predictions_manifest)
    model_rows = _json_entries(models_manifest)
    expected_prediction_keys = {
        (holdout, protocol, view, model_type, int(seed))
        for holdout in holdouts for protocol in protocols for seed in seeds
        for view in views for model_type in model_types
    }
    keys = [_prediction_key(row) for row in prediction_rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_prediction_keys:
        raise ValueError("prediction manifest has missing, duplicate, or extra combinations")
    models = {_model_key(row): row for row in model_rows}
    if len(models) != len(model_rows):
        raise ValueError("model manifest contains duplicate identities")
    expected_model_keys = {
        (HOLDOUT_TO_TRAIN[holdout], int(seed), view, model_type)
        for holdout in holdouts for seed in seeds for view in views for model_type in model_types
    }
    if set(models) != expected_model_keys:
        raise ValueError("model manifest has missing or extra identities")
    rows_by_key = {_prediction_key(row): row for row in prediction_rows}

    grouped: dict[tuple[str, str], PredictionGroup] = {}
    arrays = []
    holdout_order: dict[str, np.ndarray] = {}
    for holdout in holdouts:
        expected_site_indices = np.asarray(site_indices[holdout])
        if expected_site_indices.ndim != 1 or expected_site_indices.dtype.kind not in "iu":
            raise ValueError("site identity indices must be one-dimensional integers")
        for protocol in protocols:
            columns = []
            model_key_text = []
            canonical = None
            for seed in seeds:
                for view in views:
                    for model_type in model_types:
                        row = rows_by_key[(holdout, protocol, view, model_type, int(seed))]
                        train_site = HOLDOUT_TO_TRAIN[holdout]
                        if str(row.get("train_site")) != train_site:
                            raise ValueError("prediction train-site mismatch")
                        model = models.get((train_site, int(seed), view, model_type))
                        if model is None:
                            raise ValueError("prediction model identity missing")
                        for field in ("model_sha256", "adapter_sha256", "feature_order_sha256"):
                            if row.get(field) != model.get(field):
                                raise ValueError(f"prediction model trace mismatch: {field}")
                        path = _inside_approved(Path(str(row["prediction_reference"])), approved_roots)
                        with np.load(path, allow_pickle=False) as archive:
                            if set(archive.files) != {"scores", "test_global_indices"}:
                                raise ValueError("prediction NPZ keys mismatch")
                            score = np.asarray(archive["scores"])
                            indices = np.asarray(archive["test_global_indices"])
                        if score.ndim != 1 or score.dtype != np.float64 or not np.isfinite(score).all():
                            raise ValueError("prediction scores must be finite one-dimensional float64")
                        if indices.ndim != 1 or indices.dtype.kind not in "iu":
                            raise ValueError("test indices must be one-dimensional integers")
                        if len(score) != len(indices) or len(np.unique(indices)) != len(indices):
                            raise ValueError("prediction/index length or uniqueness mismatch")
                        if not np.array_equal(indices, expected_site_indices):
                            raise ValueError("prediction indices do not match frozen site identity order")
                        if array_sha256(score) != row.get("prediction_sha256"):
                            raise ValueError("prediction array SHA256 mismatch")
                        if array_sha256(indices) != row.get("test_indices_sha256"):
                            raise ValueError("test-index array SHA256 mismatch")
                        expected_score_type = "raw_ranking_score" if model_type == "lambdarank" else "positive_probability"
                        if row.get("score_type") != expected_score_type:
                            raise ValueError("prediction score-type mismatch")
                        if model_type in {"lr", "hgb"} and (np.any(score < 0) or np.any(score > 1)):
                            raise ValueError("probability prediction outside [0,1]")
                        if canonical is not None and not np.array_equal(indices, canonical):
                            raise ValueError("test index order differs inside holdout/protocol")
                        canonical = indices.copy()
                        columns.append(score)
                        text_key = f"seed{int(seed)}|{view}|{model_type}"
                        model_key_text.append(text_key)
                        arrays.append({"key": f"{holdout}|{protocol}|{text_key}",
                                       "prediction_sha256": array_sha256(score),
                                       "test_indices_sha256": array_sha256(indices)})
            if holdout in holdout_order and not np.array_equal(canonical, holdout_order[holdout]):
                raise ValueError("protocols do not share the same holdout order")
            holdout_order[holdout] = canonical.copy()
            grouped[(holdout, protocol)] = PredictionGroup(
                holdout, protocol, tuple(model_key_text), canonical,
                np.column_stack(columns).astype(np.float64, copy=False))
    return grouped, {
        "predictions_manifest_sha256": sha256_file(predictions_manifest),
        "models_manifest_sha256": sha256_file(models_manifest),
        "prediction_arrays": arrays,
    }


def load_eval_targets(metadata: MetadataIndex, test_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(test_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("evaluation indices must be one-dimensional integers")
    if np.any(indices < 0) or np.any(indices >= len(metadata.records)) or len(np.unique(indices)) != len(indices):
        raise ValueError("evaluation indices are duplicated or out of range")
    labels = np.empty(len(indices), dtype=np.int8)
    ages = np.empty(len(indices), dtype=np.float64)
    for output_index, source_index in enumerate(indices):
        record = metadata.records[int(source_index)]
        labels[output_index] = parse_label(record.label_text)
        ages[output_index] = parse_age(record.age_text)
    return labels, ages


def _ac_pair_count(labels: np.ndarray, ages: np.ndarray, gap: float = 2.0) -> int:
    positive_ages = ages[labels == 1]
    negative_ages = ages[labels == 0]
    return int(sum(np.count_nonzero(np.abs(negative_ages - age) <= gap) for age in positive_ages))


def score_site(labels: np.ndarray, scores: np.ndarray, ages: np.ndarray, official) -> dict[str, object]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    ages = np.asarray(ages)
    if labels.ndim != 1 or scores.ndim != 1 or ages.ndim != 1 or not (labels.shape == scores.shape == ages.shape):
        raise ValueError("labels, scores, and ages must be aligned vectors")
    if not np.all(np.isin(labels, [0, 1])) or not np.isfinite(scores).all() or not np.isfinite(ages).all():
        raise ValueError("evaluation inputs are invalid")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    pairs = _ac_pair_count(labels, ages, gap=2.0)
    complete = positives > 0 and negatives > 0
    ac = float(official.compute_auroc_age(labels, scores, ages, gap=2)) if complete and pairs else float("nan")
    auroc = float(official.compute_auroc(labels, scores)) if complete else float("nan")
    auprc = float(official.compute_auprc(labels, scores)) if complete else float("nan")
    return {"n_records": len(labels), "n_positive": positives, "n_negative": negatives,
            "ac_pair_count": pairs, "n_score_unique": int(len(np.unique(scores))),
            "ac_auroc": ac, "auroc": auroc, "auprc": auprc,
            "status": "PASS" if complete and pairs else "INCOMPLETE"}


def aggregate_protocol(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    by_site = {str(row["holdout_site"]): float(row["ac_auroc"]) for row in rows}
    if set(by_site) != set(HOLDOUTS):
        raise ValueError("protocol aggregate requires exactly three holdout sites")
    values = np.asarray([by_site[site] for site in HOLDOUTS], dtype=np.float64)
    if not np.isfinite(values).all():
        return {"macro_ac": float("nan"), "worst_site_ac": float("nan"),
                "small_site_mean_ac": float("nan")}
    return {"macro_ac": float(values.mean()), "worst_site_ac": float(values.min()),
            "small_site_mean_ac": float((by_site["I0002"] + by_site["I0006"]) / 2.0)}


__all__ = ["PredictionGroup", "aggregate_protocol", "array_sha256", "load_eval_targets",
           "load_predictions", "mean_ranks", "score_site", "sha256_file"]
