"""Post-hoc stability and replacement diagnostics for frozen E0--E3 predictions.

This module never trains, predicts, extracts features, or changes the formal
E0--E3 fusion. It validates the frozen prediction references before loading
evaluation targets, then constructs deterministic rank fusions in memory.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from reproduction.siren_a.g5.core import (
    HOLDOUTS,
    MODEL_TYPES,
    PROTOCOLS,
    VIEWS,
    load_eval_targets,
    mean_ranks,
    score_site,
)

from .compare_e0_e3_v2 import (
    EXPERIMENT_SOURCES,
    group_reference_entries,
    load_prediction_row,
)

SEEDS = tuple(range(42, 57))
COMPARISONS = ("E1-E0", "E2-E0", "E3-E0", "E3-E2")
BOOTSTRAP_SEED = 20261023


def _fixed_keys() -> tuple[tuple[int, str, str], ...]:
    return tuple((seed, view, model) for seed in SEEDS for view in VIEWS for model in MODEL_TYPES)


def select_seed_columns(keys: Sequence[tuple[int, str, str]], scores: np.ndarray, seed: int):
    """Select one exact 15-column seed cohort in view→model order."""
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != len(keys):
        raise ValueError("score/key shape mismatch")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate prediction key")
    expected = tuple((int(seed), view, model) for view in VIEWS for model in MODEL_TYPES)
    selected = tuple(sorted((key for key in keys if int(key[0]) == int(seed)),
                            key=lambda key: (VIEWS.index(key[1]), MODEL_TYPES.index(key[2]))))
    if selected != expected:
        raise ValueError("seed does not have exactly one complete 15-column cohort")
    positions = [keys.index(key) for key in expected]
    return expected, values[:, positions].astype(np.float64, copy=False)


def fuse_score_columns(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return anonymous row positions and the frozen mean-tie-rank fusion."""
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != 15 or values.dtype != np.float64:
        raise ValueError("a seed fusion requires an N x 15 float64 matrix")
    if not np.isfinite(values).all():
        raise ValueError("fusion scores must be finite")
    _, fused = mean_ranks(values)
    return np.arange(len(values), dtype=np.int64), fused.astype(np.float64, copy=False)


def pair_contributions(labels: np.ndarray, ages: np.ndarray, scores: np.ndarray, *, gap: float = 2.0):
    """Precompute official AC eligible pairs and 0/0.5/1 contributions."""
    y = np.asarray(labels)
    a = np.asarray(ages, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or a.shape != y.shape or s.shape != y.shape:
        raise ValueError("pair inputs must be aligned vectors")
    if not np.all(np.isin(y, [0, 1])) or not np.isfinite(a).all() or not np.isfinite(s).all():
        raise ValueError("pair inputs invalid")
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    pos_parts = []
    neg_parts = []
    contributions = []
    for i in positives:
        eligible = negatives[np.abs(a[negatives] - a[i]) <= gap]
        if len(eligible) == 0:
            continue
        pos_parts.append(np.full(len(eligible), int(i), dtype=np.int64))
        neg_parts.append(eligible.astype(np.int64, copy=False))
        contributions.append(np.where(s[i] > s[eligible], 1.0,
                                      np.where(s[i] == s[eligible], 0.5, 0.0)))
    if not pos_parts:
        return {"positive": np.empty(0, dtype=np.int64),
                "negative": np.empty(0, dtype=np.int64),
                "contribution": np.empty(0, dtype=np.float64)}
    return {"positive": np.concatenate(pos_parts), "negative": np.concatenate(neg_parts),
            "contribution": np.concatenate(contributions).astype(np.float64, copy=False)}


def weighted_ac_from_pairs(pairs: Mapping[str, np.ndarray], multiplicity: np.ndarray) -> float:
    """Evaluate a weighted pair AC; zero-denominator replicates are NaN."""
    weights = np.asarray(multiplicity, dtype=np.int64)
    if weights.ndim != 1 or np.any(weights < 0):
        raise ValueError("multiplicity must be a nonnegative vector")
    p = np.asarray(pairs["positive"], dtype=np.int64)
    n = np.asarray(pairs["negative"], dtype=np.int64)
    c = np.asarray(pairs["contribution"], dtype=np.float64)
    if p.shape != n.shape or p.shape != c.shape:
        raise ValueError("pair payload shape mismatch")
    if len(p) == 0:
        return float("nan")
    pair_weight = weights[p].astype(np.float64) * weights[n].astype(np.float64)
    denominator = float(pair_weight.sum())
    if denominator <= 0:
        return float("nan")
    return float(np.dot(pair_weight, c) / denominator)


def weighted_ac_batch(pairs: Mapping[str, np.ndarray], multiplicities: np.ndarray,
                      *, chunk_size: int = 512) -> np.ndarray:
    """Vectorized weighted AC for many fixed multiplicity vectors."""
    matrix = np.asarray(multiplicities, dtype=np.int64)
    if matrix.ndim != 2 or np.any(matrix < 0):
        raise ValueError("multiplicities must be a nonnegative B x N matrix")
    p = np.asarray(pairs["positive"], dtype=np.int64)
    n = np.asarray(pairs["negative"], dtype=np.int64)
    c = np.asarray(pairs["contribution"], dtype=np.float64)
    if len(p) == 0:
        return np.full(matrix.shape[0], np.nan, dtype=np.float64)
    positive = np.unique(p)
    negative = np.unique(n)
    positive_positions = np.searchsorted(positive, p)
    negative_positions = np.searchsorted(negative, n)
    contribution_matrix = np.zeros((len(positive), len(negative)), dtype=np.float64)
    eligibility_matrix = np.zeros((len(positive), len(negative)), dtype=np.float64)
    contribution_matrix[positive_positions, negative_positions] = c
    eligibility_matrix[positive_positions, negative_positions] = 1.0
    result = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    for start in range(0, matrix.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), matrix.shape[0])
        positive_weights = matrix[start:stop, positive].astype(np.float64)
        negative_weights = matrix[start:stop, negative].astype(np.float64)
        numerator = ((positive_weights @ contribution_matrix) * negative_weights).sum(axis=1)
        denominator = ((positive_weights @ eligibility_matrix) * negative_weights).sum(axis=1)
        valid = denominator > 0
        block = result[start:stop]
        block[valid] = numerator[valid] / denominator[valid]
        result[start:stop] = block
    return result


def bootstrap_multiplicity(n_records: int, rng: np.random.Generator,
                           groups: Sequence[Sequence[int]] | None = None) -> np.ndarray:
    """Draw record multiplicities; cluster multiplicity is shared by all records."""
    if n_records <= 0:
        raise ValueError("n_records must be positive")
    if groups is None:
        groups = tuple((i,) for i in range(n_records))
    normalized = tuple(tuple(int(i) for i in group) for group in groups)
    flat = [i for group in normalized for i in group]
    if sorted(flat) != list(range(n_records)):
        raise ValueError("groups must partition all record positions")
    chosen = rng.integers(0, len(normalized), size=len(normalized))
    out = np.zeros(n_records, dtype=np.int64)
    for unit in chosen:
        for index in normalized[int(unit)]:
            out[index] += 1
    return out


def build_replacement_keys(base: Mapping[tuple[int, str, str], object],
                           v2: Mapping[tuple[int, str, str], object], *,
                           view: str, model: str | None = None):
    """Replace one view or one view×model in an E0 identity mapping."""
    expected = set(_fixed_keys())
    if set(base) != expected or set(v2) != expected or view not in VIEWS:
        raise ValueError("replacement identity coverage mismatch")
    out = dict(base)
    for key in expected:
        if key[1] == view and (model is None or key[2] == model):
            out[key] = v2[key]
    return out


def expanded_ac(labels: np.ndarray, ages: np.ndarray, scores: np.ndarray,
                multiplicity: np.ndarray, official) -> float:
    """Reference implementation using explicit repeated records."""
    w = np.asarray(multiplicity, dtype=np.int64)
    expanded = np.repeat(np.arange(len(w), dtype=np.int64), w)
    if len(expanded) == 0:
        return float("nan")
    return float(official.compute_auroc_age(labels[expanded], scores[expanded],
                                            ages[expanded], gap=2))


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_reference_sources(reference_manifest: Path):
    payload = _read_json(reference_manifest)
    if payload.get("schema") != "featurelab_v2_e0_e3_reference_v1":
        raise ValueError("reference manifest schema mismatch")
    rows = payload.get("entries")
    if not isinstance(rows, list) or len(rows) != 5400:
        raise ValueError("reference manifest coverage mismatch")
    group_reference_entries(rows)
    row_by_source = {}
    arrays = {}
    for row in rows:
        key = (str(row["source_version"]), str(row["holdout_site"]),
               str(row["protocol"]), str(row["view"]), str(row["model_type"]),
               int(row["seed"]))
        previous = row_by_source.get(key)
        if previous is not None and previous["prediction_reference"] != row["prediction_reference"]:
            raise ValueError("source identity differs across repeated experiment references")
        row_by_source[key] = row
    for key, row in sorted(row_by_source.items()):
        scores, indices = load_prediction_row(row)
        arrays[key] = (scores.copy(), indices.copy())
    if len(arrays) != 2700:
        raise ValueError("expected old/v2 source coverage")
    return row_by_source, arrays


def _fuse_source(arrays, holdout: str, protocol: str,
                 source_by_key: Mapping[tuple[int, str, str], str], seed: int | None = None):
    seeds = (int(seed),) if seed is not None else SEEDS
    columns = []
    keys = []
    canonical = None
    for current_seed in seeds:
        for view in VIEWS:
            for model in MODEL_TYPES:
                source = source_by_key[(current_seed, view, model)]
                scores, indices = arrays[(source, holdout, protocol, view, model, current_seed)]
                if canonical is None:
                    canonical = indices
                elif not np.array_equal(canonical, indices):
                    raise ValueError("prediction test-index order mismatch")
                columns.append(scores)
                keys.append((current_seed, view, model))
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    if seed is not None:
        return keys, matrix, *fuse_score_columns(matrix)
    _, fused = mean_ranks(matrix)
    return keys, matrix, canonical.copy(), fused.astype(np.float64, copy=False)


def _source_map(experiment: str) -> dict[str, str]:
    if experiment not in EXPERIMENT_SOURCES:
        raise ValueError(experiment)
    return dict(EXPERIMENT_SOURCES[experiment])


def build_prediction_source_keys(*, experiment: str | None = None,
                                 view: str | None = None,
                                 model: str | None = None) -> dict[tuple[int, str, str], str]:
    """Resolve the actual old/v2 source for every frozen prediction key.

    Replacement selection is deliberately expressed at ``(seed, view, model)``
    granularity so a View×Model diagnostic cannot silently become a full-View
    replacement.
    """
    if experiment is not None and (view is not None or model is not None):
        raise ValueError("choose experiment or replacement")
    if model is not None and view is None:
        raise ValueError("model replacement requires a view")
    if experiment is not None:
        view_sources = _source_map(experiment)
    else:
        if view is not None and view not in VIEWS:
            raise ValueError("invalid replacement view")
        if model is not None and model not in MODEL_TYPES:
            raise ValueError("invalid replacement model")
        view_sources = {current_view: "old" for current_view in VIEWS}
    result = {}
    for seed in SEEDS:
        for current_view in VIEWS:
            for current_model in MODEL_TYPES:
                source = view_sources[current_view]
                if experiment is None and view is not None:
                    if current_view == view and (model is None or current_model == model):
                        source = "v2"
                result[(seed, current_view, current_model)] = source
    if len(result) != len(_fixed_keys()) or set(result) != set(_fixed_keys()):
        raise RuntimeError("prediction source key coverage mismatch")
    return result


def _replacement_map(view: str | None = None, model: str | None = None) -> dict[tuple[int, str, str], str]:
    return build_prediction_source_keys(view=view, model=model)


def build_fused_groups(arrays, *, experiment: str | None = None,
                       view: str | None = None, model: str | None = None):
    source_map = build_prediction_source_keys(experiment=experiment, view=view, model=model)
    groups = {}
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            _, _, indices, fused = _fuse_source(arrays, holdout, protocol, source_map)
            groups[(holdout, protocol)] = (indices, fused)
    return groups


def build_single_seed_groups(arrays, *, experiment: str, seed: int):
    source_map = build_prediction_source_keys(experiment=experiment)
    groups = {}
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            _, matrix, indices, fused = _fuse_source(arrays, holdout, protocol,
                                                      source_map, seed=seed)
            groups[(holdout, protocol)] = (indices, fused, matrix)
    return groups


SCOPES = (*HOLDOUTS, "Macro", "Worst-site", "Small-site mean")


def _metric_scope(site_metrics: Mapping[str, float], scope: str) -> float:
    if scope in HOLDOUTS:
        return float(site_metrics[scope])
    values = np.asarray([site_metrics[s] for s in HOLDOUTS], dtype=np.float64)
    if not np.isfinite(values).all():
        return float("nan")
    if scope == "Macro":
        return float(values.mean())
    if scope == "Worst-site":
        return float(values.min())
    if scope == "Small-site mean":
        return float((site_metrics["I0002"] + site_metrics["I0006"]) / 2.0)
    raise ValueError(scope)


def _ac_fast(labels: np.ndarray, ages: np.ndarray, scores: np.ndarray) -> float:
    """Official-equivalent AC after the pair engine has been validated."""
    pairs = pair_contributions(labels, ages, scores, gap=2.0)
    return weighted_ac_from_pairs(pairs, np.ones(len(labels), dtype=np.int64))


def seed_stability_rows(arrays, metadata, official):
    """Return aggregate and private per-seed fixed-fusion diagnostics."""
    metrics = {}
    for experiment in ("E0", "E1", "E2", "E3"):
        for seed in SEEDS:
            groups = build_single_seed_groups(arrays, experiment=experiment, seed=seed)
            per_site = {}
            for holdout in HOLDOUTS:
                for protocol in PROTOCOLS:
                    idx, fused, _ = groups[(holdout, protocol)]
                    labels, ages = load_eval_targets(metadata, idx)
                    per_site[(protocol, holdout)] = {"ac_auroc": _ac_fast(labels, ages, fused)}
            metrics[(experiment, seed)] = per_site
    rows = []
    for protocol in PROTOCOLS:
        for comparison in COMPARISONS:
            left, right = comparison.split("-")
            for scope in SCOPES:
                values = []
                for seed in SEEDS:
                    left_sites = {site: metrics[(left, seed)][(protocol, site)]["ac_auroc"]
                                  for site in HOLDOUTS}
                    right_sites = {site: metrics[(right, seed)][(protocol, site)]["ac_auroc"]
                                   for site in HOLDOUTS}
                    values.append(_metric_scope(left_sites, scope) -
                                  _metric_scope(right_sites, scope))
                values = np.asarray(values, dtype=np.float64)
                rows.append({
                    "comparison": comparison, "protocol": protocol, "scope": scope,
                    "n_seeds": 15, "n_positive": int(np.sum(values > 0)),
                    "n_zero": int(np.sum(values == 0)),
                    "n_negative": int(np.sum(values < 0)),
                    "positive_fraction": float(np.mean(values > 0)),
                    "mean_delta": float(values.mean()),
                    "median_delta": float(np.median(values)),
                    "q25": float(np.quantile(values, .25)),
                    "q75": float(np.quantile(values, .75)),
                    "min": float(values.min()), "max": float(values.max()),
                })
    return rows, metrics


def paired_bootstrap(fused_by_experiment, metadata, *, official,
                     n_bootstrap: int = 2000, seed: int = BOOTSTRAP_SEED,
                     return_raw: bool = False):
    """Compute paired E2/E3 deltas with identical site multiplicities."""
    required = {"E0", "E2", "E3"}
    if not required.issubset(fused_by_experiment):
        raise ValueError("bootstrap requires E0, E2 and E3")
    rng = np.random.default_rng(seed)
    comparisons = ("E2-E0", "E3-E0", "E3-E2")
    raw = {(protocol, comparison, scope): [] for protocol in PROTOCOLS
           for comparison in comparisons for scope in SCOPES}
    invalid = {(protocol, comparison, scope): 0 for protocol in PROTOCOLS
               for comparison in comparisons for scope in SCOPES}
    observed = {}
    pair_cache = {}
    for protocol in PROTOCOLS:
        for holdout in HOLDOUTS:
            idx = fused_by_experiment["E0"][(holdout, protocol)][0]
            labels, ages = load_eval_targets(metadata, idx)
            for experiment in ("E0", "E2", "E3"):
                scores = fused_by_experiment[experiment][(holdout, protocol)][1]
                pair_cache[(experiment, holdout, protocol)] = (
                    labels, ages, pair_contributions(labels, ages, scores))
        for comparison in comparisons:
            left, right = comparison.split("-")
            left_sites = {}
            right_sites = {}
            for holdout in HOLDOUTS:
                for experiment, target in ((left, left_sites), (right, right_sites)):
                    idx, scores = fused_by_experiment[experiment][(holdout, protocol)]
                    labels, ages = load_eval_targets(metadata, idx)
                    target[holdout] = _ac_fast(labels, ages, scores)
            observed[(protocol, comparison)] = (left_sites, right_sites)
    multiplicities = {}
    for holdout in HOLDOUTS:
        n_records = len(pair_cache[("E0", holdout, PROTOCOLS[0])][0])
        draws = np.empty((n_bootstrap, n_records), dtype=np.int64)
        for replicate in range(n_bootstrap):
            draws[replicate] = bootstrap_multiplicity(n_records, rng)
        multiplicities[holdout] = draws
    for protocol in PROTOCOLS:
        site_values = {experiment: {} for experiment in ("E0", "E2", "E3")}
        for holdout in HOLDOUTS:
            draws = multiplicities[holdout]
            for experiment in ("E0", "E2", "E3"):
                pair = pair_cache[(experiment, holdout, protocol)][2]
                site_values[experiment][holdout] = weighted_ac_batch(pair, draws)
        for replicate in range(n_bootstrap):
            for comparison in comparisons:
                left, right = comparison.split("-")
                left_sites = {site: site_values[left][site][replicate] for site in HOLDOUTS}
                right_sites = {site: site_values[right][site][replicate] for site in HOLDOUTS}
                for scope in SCOPES:
                    key = (protocol, comparison, scope)
                    left_value = _metric_scope(left_sites, scope)
                    right_value = _metric_scope(right_sites, scope)
                    value = left_value - right_value
                    if np.isfinite(value):
                        raw[key].append(float(value))
                    else:
                        invalid[key] += 1
    summaries = []
    for protocol in PROTOCOLS:
        for comparison in comparisons:
            left, right = comparison.split("-")
            for scope in SCOPES:
                values = np.asarray(raw[(protocol, comparison, scope)], dtype=np.float64)
                left_obs = _metric_scope(observed[(protocol, comparison)][0], scope)
                right_obs = _metric_scope(observed[(protocol, comparison)][1], scope)
                summaries.append({
                    "comparison": comparison, "protocol": protocol, "scope": scope,
                    "observed_delta": float(left_obs - right_obs),
                    "bootstrap_mean_delta": float(values.mean()) if len(values) else float("nan"),
                    "bootstrap_median_delta": float(np.median(values)) if len(values) else float("nan"),
                    "bootstrap_sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                    "percentile_2_5": float(np.quantile(values, .025)) if len(values) else float("nan"),
                    "percentile_97_5": float(np.quantile(values, .975)) if len(values) else float("nan"),
                    "fraction_gt_0": float(np.mean(values > 0)) if len(values) else float("nan"),
                    "valid_replicates": int(len(values)),
                    "invalid_replicates": int(invalid[(protocol, comparison, scope)]),
                })
    return (summaries, raw) if return_raw else summaries


def score_replacements(arrays, metadata, official):
    """Score five single-view and fifteen view×model post-hoc replacements."""
    e0 = build_fused_groups(arrays, experiment="E0")
    e2 = build_fused_groups(arrays, experiment="E2")
    scores = {}
    def score_groups(groups):
        rows = {}
        for holdout in HOLDOUTS:
            for protocol in PROTOCOLS:
                idx, fused = groups[(holdout, protocol)]
                labels, ages = load_eval_targets(metadata, idx)
                rows[(holdout, protocol)] = {
                    "ac_auroc": _ac_fast(labels, ages, fused),
                    "auroc": float(official.compute_auroc(labels, fused)),
                    "auprc": float(official.compute_auprc(labels, fused)),
                }
        return rows
    scores["E0"] = score_groups(e0)
    scores["E2"] = score_groups(e2)
    replacements = {"R_V5": e2}
    for view in VIEWS:
        name = f"R_{view}"
        groups = e2 if view == "V5" else build_fused_groups(arrays, view=view)
        scores[name] = score_groups(groups)
        if view != "V5":
            del groups
    for view in VIEWS:
        for model in MODEL_TYPES:
            name = f"R_{view}_{model}"
            groups = build_fused_groups(arrays, view=view, model=model)
            scores[name] = score_groups(groups)
            del groups
    def metric_delta(name, protocol, scope, metric):
        base = {site: scores["E0"][(site, protocol)][metric] for site in HOLDOUTS}
        current = {site: scores[name][(site, protocol)][metric] for site in HOLDOUTS}
        return _metric_scope(current, scope) - _metric_scope(base, scope)

    view_rows = []
    for name in [f"R_{view}" for view in VIEWS]:
        for protocol in PROTOCOLS:
            for scope in SCOPES:
                view_rows.append({
                    "replacement": name, "protocol": protocol, "scope": scope,
                    "delta_ac": metric_delta(name, protocol, scope, "ac_auroc"),
                    "delta_auroc": metric_delta(name, protocol, scope, "auroc"),
                    "delta_auprc": metric_delta(name, protocol, scope, "auprc"),
                })
    model_rows = []
    for view in VIEWS:
        for model in MODEL_TYPES:
            name = f"R_{view}_{model}"
            for protocol in PROTOCOLS:
                for scope in SCOPES:
                    model_rows.append({
                        "replacement": name, "view": view, "model": model,
                        "protocol": protocol, "scope": scope,
                        "delta_ac": metric_delta(name, protocol, scope, "ac_auroc"),
                        "delta_auroc": metric_delta(name, protocol, scope, "auroc"),
                        "delta_auprc": metric_delta(name, protocol, scope, "auprc"),
                    })
    return scores, view_rows, model_rows, replacements


def score_view_model_replacements(arrays, metadata, official):
    """Score only the fifteen View×Model replacements from the fixed E0 base."""
    e0 = build_fused_groups(arrays, experiment="E0")

    def score_groups(groups):
        rows = {}
        for holdout in HOLDOUTS:
            for protocol in PROTOCOLS:
                idx, fused = groups[(holdout, protocol)]
                labels, ages = load_eval_targets(metadata, idx)
                rows[(holdout, protocol)] = {
                    "ac_auroc": _ac_fast(labels, ages, fused),
                    "auroc": float(official.compute_auroc(labels, fused)),
                    "auprc": float(official.compute_auprc(labels, fused)),
                }
        return rows

    base_scores = score_groups(e0)
    model_rows = []
    source_counts = {}
    for view in VIEWS:
        for model in MODEL_TYPES:
            name = f"R_{view}_{model}"
            source_keys = build_prediction_source_keys(view=view, model=model)
            source_counts[name] = {
                "n_v2_columns": sum(source == "v2" for source in source_keys.values()),
                "n_old_columns": sum(source == "old" for source in source_keys.values()),
                "n_total_columns": len(source_keys),
            }
            groups = build_fused_groups(arrays, view=view, model=model)
            current_scores = score_groups(groups)
            for protocol in PROTOCOLS:
                for scope in SCOPES:
                    base = {site: base_scores[(site, protocol)] for site in HOLDOUTS}
                    current = {site: current_scores[(site, protocol)] for site in HOLDOUTS}
                    model_rows.append({
                        "replacement": name, "view": view, "model": model,
                        "protocol": protocol, "scope": scope,
                        "delta_ac": _metric_scope({site: current[site]["ac_auroc"] for site in HOLDOUTS}, scope) - _metric_scope({site: base[site]["ac_auroc"] for site in HOLDOUTS}, scope),
                        "delta_auroc": _metric_scope({site: current[site]["auroc"] for site in HOLDOUTS}, scope) - _metric_scope({site: base[site]["auroc"] for site in HOLDOUTS}, scope),
                        "delta_auprc": _metric_scope({site: current[site]["auprc"] for site in HOLDOUTS}, scope) - _metric_scope({site: base[site]["auprc"] for site in HOLDOUTS}, scope),
                    })
    return model_rows, source_counts


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty public table")
    fields = list(rows[0].keys())
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists() or path.is_symlink():
        temporary.unlink(missing_ok=True)
        raise ValueError(f"refusing to overwrite public output: {path}")
    os.rename(temporary, path)


def assert_public_privacy(rows: Sequence[Mapping[str, object]]) -> None:
    forbidden = ("record", "patient", "age", "label", "prediction",
                 "path", "multiplicity", "row_index")
    for row in rows:
        for key, value in row.items():
            low = str(key).lower()
            if any(token in low for token in forbidden):
                raise ValueError(f"forbidden public field: {key}")
            if isinstance(value, str) and any(token in value.lower() for token in forbidden):
                raise ValueError("forbidden public value")


__all__ = [
    "BOOTSTRAP_SEED", "COMPARISONS", "SCOPES", "SEEDS",
    "assert_public_privacy", "bootstrap_multiplicity", "build_fused_groups",
    "build_prediction_source_keys",
    "build_replacement_keys", "expanded_ac", "fuse_score_columns",
    "load_reference_sources", "pair_contributions", "paired_bootstrap",
    "score_replacements", "score_view_model_replacements", "seed_stability_rows", "select_seed_columns",
    "weighted_ac_batch", "weighted_ac_from_pairs",
]
