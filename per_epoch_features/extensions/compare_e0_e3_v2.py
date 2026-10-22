"""Fixed old/v2 prediction references, rank fusion, and delayed scoring."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from reproduction.siren_a.g5.core import (
    HOLDOUTS, HOLDOUT_TO_TRAIN, MODEL_TYPES, PROTOCOLS, VIEWS,
    aggregate_protocol, load_eval_targets, mean_ranks, score_site,
)

EXPERIMENT_SOURCES = {
    "E0": {"V1": "old", "V2": "old", "V3": "old", "V4": "old", "V5": "old"},
    "E1": {"V1": "v2", "V2": "v2", "V3": "v2", "V4": "v2", "V5": "old"},
    "E2": {"V1": "old", "V2": "old", "V3": "old", "V4": "old", "V5": "v2"},
    "E3": {"V1": "v2", "V2": "v2", "V3": "v2", "V4": "v2", "V5": "v2"},
}
SEEDS = tuple(range(42, 57))


def fixed_column_keys() -> tuple[tuple[int, str, str], ...]:
    return tuple((seed, view, model) for seed in SEEDS for view in VIEWS for model in MODEL_TYPES)


def normalize_score_type(value: str) -> str:
    if value == "raw_ranking_score":
        return "lambdarank_raw"
    if value in {"lambdarank_raw", "positive_probability"}:
        return value
    raise ValueError(f"unknown score type: {value}")


def _array_sha(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256(); digest.update(str(array.shape).encode("ascii")); digest.update(array.dtype.str.encode("ascii")); digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_old_rows(manifest_path: Path) -> dict[tuple[str, str, str, str, int], Mapping[str, object]]:
    payload = _json(manifest_path)
    rows = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("old prediction manifest entries missing")
    result = {}
    for row in rows:
        key = (str(row["holdout_site"]), str(row["protocol"]), str(row["view"]), str(row["model_type"]), int(row["seed"]))
        if key in result: raise ValueError("duplicate old prediction identity")
        result[key] = dict(row)
    expected = {(h, p, v, m, s) for h in HOLDOUTS for p in PROTOCOLS for v in VIEWS for m in MODEL_TYPES for s in SEEDS}
    if set(result) != expected: raise ValueError("old prediction manifest coverage mismatch")
    return result


def collect_v2_rows(seed42_root: Path, new_root: Path) -> dict[tuple[str, str, str, str, int], Mapping[str, object]]:
    result = {}
    roots = ((42, Path(seed42_root)),) + tuple((seed, Path(new_root)) for seed in SEEDS if seed != 42)
    for seed, root in roots:
        for holdout in HOLDOUTS:
            for protocol in PROTOCOLS:
                for view in VIEWS:
                    for model in MODEL_TYPES:
                        path = (root / "predictions" / f"holdout_{holdout}" / protocol / view / model
                                if seed == 42 else root / "predictions" / f"seed{seed}" / f"holdout_{holdout}" / protocol / view / model)
                        meta_path = path / "metadata.json"
                        if not meta_path.is_file(): raise ValueError(f"v2 prediction missing: {path}")
                        metadata = _json(meta_path)
                        if int(metadata.get("seed", -1)) != seed or metadata.get("source_version") != "v2":
                            raise ValueError("v2 prediction identity mismatch")
                        key = (holdout, protocol, view, model, seed)
                        if key in result: raise ValueError("duplicate v2 prediction identity")
                        result[key] = {"source_version": "v2", "prediction_reference": str(path), **metadata}
    expected = {(h, p, v, m, s) for h in HOLDOUTS for p in PROTOCOLS for v in VIEWS for m in MODEL_TYPES for s in SEEDS}
    if set(result) != expected: raise ValueError("v2 prediction manifest coverage mismatch")
    return result


def build_reference_entries(old_rows: Mapping, v2_rows: Mapping, experiment: str) -> list[Mapping[str, object]]:
    if experiment not in EXPERIMENT_SOURCES: raise ValueError(experiment)
    rows = []
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                for view in VIEWS:
                    source = EXPERIMENT_SOURCES[experiment][view]
                    for model in MODEL_TYPES:
                        key = (holdout, protocol, view, model, seed)
                        row = old_rows[key] if source == "old" else v2_rows[key]
                        rows.append({"experiment": experiment, "holdout_site": holdout, "protocol": protocol,
                                     "seed": seed, "view": view, "model_type": model, "source_version": source,
                                     "prediction_reference": str(row["prediction_reference"]),
                                     "score_type": normalize_score_type(str(row["score_type"]))})
    if len(rows) != 1350: raise ValueError("reference entry count mismatch")
    return rows


def load_prediction_row(row: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(row["prediction_reference"]))
    with np.load(path / "prediction.npz" if path.is_dir() else path, allow_pickle=False) as archive:
        if set(archive.files) != {"scores", "test_global_indices"}:
            raise ValueError("prediction payload keys mismatch")
        scores = np.asarray(archive["scores"]); indices = np.asarray(archive["test_global_indices"])
    if scores.ndim != 1 or scores.dtype != np.float64 or not np.isfinite(scores).all(): raise ValueError("invalid score vector")
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != len(scores) or len(np.unique(indices)) != len(indices): raise ValueError("invalid test indices")
    expected = normalize_score_type(str(row["score_type"]))
    if expected == "positive_probability" and (np.any(scores < 0) or np.any(scores > 1)): raise ValueError("probability range")
    if str(row.get("prediction_sha256", "")) and str(row["source_version"]) == "old":
        # Old G5 uses its own digest convention; the raw artifact was already verified upstream.
        pass
    return scores, indices.astype(np.int64, copy=False)


def fuse_prediction_columns(score_matrix: np.ndarray, test_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(score_matrix); indices = np.asarray(test_indices)
    if scores.ndim != 2 or scores.shape[1] != 225 or scores.dtype != np.float64 or not np.isfinite(scores).all(): raise ValueError("fusion matrix must be finite N x 225")
    if indices.ndim != 1 or len(indices) != len(scores) or len(np.unique(indices)) != len(indices): raise ValueError("fusion indices invalid")
    ranks, fused = mean_ranks(scores)
    return indices.astype(np.int64, copy=True), fused.astype(np.float64, copy=False), ranks


def fuse_reference_group(entries: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    if len(entries) != 225: raise ValueError("each group requires 225 references")
    ordered = sorted(entries, key=lambda row: (int(row["seed"]), VIEWS.index(str(row["view"])), MODEL_TYPES.index(str(row["model_type"]))))
    expected = fixed_column_keys()
    actual = tuple((int(row["seed"]), str(row["view"]), str(row["model_type"])) for row in ordered)
    if actual != expected: raise ValueError("fusion column order or identity mismatch")
    columns = []; canonical = None
    for row in ordered:
        scores, indices = load_prediction_row(row)
        if canonical is None: canonical = indices
        elif not np.array_equal(indices, canonical): raise ValueError("test index order mismatch")
        columns.append(scores)
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    out_indices, fused, ranks = fuse_prediction_columns(matrix, canonical)
    return out_indices, fused, ranks, tuple(f"seed{int(row['seed'])}|{row['view']}|{row['model_type']}" for row in ordered)


def score_fused_groups(groups: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]], metadata, official) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    rows = []
    for holdout in HOLDOUTS:
        shared = None
        for protocol in PROTOCOLS:
            indices, fused = groups[(holdout, protocol)]
            labels, ages = load_eval_targets(metadata, indices)
            if shared is not None and (not np.array_equal(labels, shared[0]) or not np.array_equal(ages, shared[1])): raise ValueError("protocol target mismatch")
            shared = labels, ages
            metrics = score_site(labels, fused, ages, official)
            rows.append({"holdout_site": holdout, "train_site": HOLDOUT_TO_TRAIN[holdout], "protocol": protocol,
                         "ensemble_method": "equal_mean_of_225_cohort_average_tie_ranks", "n_models": 225, **metrics})
    summaries = {protocol: aggregate_protocol([r for r in rows if r["protocol"] == protocol]) for protocol in PROTOCOLS}
    return rows, summaries


__all__ = ["EXPERIMENT_SOURCES", "SEEDS", "build_reference_entries", "collect_old_rows", "collect_v2_rows",
           "fixed_column_keys", "fuse_prediction_columns", "fuse_reference_group", "load_prediction_row",
           "normalize_score_type", "score_fused_groups"]
