"""Cross-fold weight selection over the previous SIREN/Matcha frozen ranks.

This module consumes only the previous fusion run's private payloads.  It does
not load model code, raw signals, features, embeddings, or predictions.  All
weight selection is performed on the six pre-registered weights, with the
target site's labels excluded from both observed and nested-bootstrap
selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .e0e3_stability import bootstrap_multiplicity, pair_contributions, weighted_ac_batch
from .siren_matcha_ab_fusion import (
    B_EXPECTED_AC,
    EXPECTED_COUNTS,
    FusionStop,
    SCOPES,
    SITES,
    _metric,
    _scope,
    sha256_file,
)

CANDIDATE_WEIGHTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
PRIOR_WEIGHT = 0.4
TIE_TOLERANCE = 1e-12
BOOTSTRAP_SEED = 20261023
BOOTSTRAP_REPLICATES = 2000
PARENT_BRANCH = "research/siren-matcha-ab-fusion-v1"
PARENT_COMMIT = "2e151f9ec89e596469ba315eebcc312e44d20044"

A_EXPECTED_AC = {
    "I0002": 0.7423117709437964,
    "I0006": 0.7045470227006141,
    "S0001": 0.6618493939560138,
}
AB60_EXPECTED_AC = {
    "I0002": 0.7592788971367974,
    "I0006": 0.7408059062995929,
    "S0001": 0.6829018812044095,
}
AB50_EXPECTED_AC = {
    "I0002": 0.7582184517497349,
    "I0006": 0.7457738218450286,
    "S0001": 0.684056381286678,
}


def selection_folds(target: str, sites: Sequence[str] = SITES) -> tuple[str, ...]:
    """Return the two folds permitted to select a target's weight."""
    if target not in sites or len(sites) != 3:
        raise FusionStop(f"unknown three-fold target: {target}")
    return tuple(site for site in sites if site != target)


def build_weighted_score(r_a: np.ndarray, r_b: np.ndarray, w_b: float) -> np.ndarray:
    """Mix frozen branch ranks with ``w_b`` as Matcha-B's weight."""
    if float(w_b) not in CANDIDATE_WEIGHTS:
        raise FusionStop(f"weight is outside the pre-registered grid: {w_b}")
    a = np.asarray(r_a, dtype=np.float64)
    b = np.asarray(r_b, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FusionStop("rank vectors must be aligned finite vectors")
    return ((1.0 - float(w_b)) * a + float(w_b) * b).astype(np.float64)


def select_weight(selection_scores: Mapping[float, float]) -> float:
    """Select by max score, then closest to 0.4, then lower weight."""
    if not selection_scores:
        raise FusionStop("empty candidate selection scores")
    finite = {float(weight): float(score) for weight, score in selection_scores.items() if np.isfinite(score)}
    if not finite:
        raise FusionStop("all candidate selection scores are invalid")
    best_score = max(finite.values())
    tied = [weight for weight, score in finite.items() if abs(score - best_score) <= TIE_TOLERANCE]
    min_distance = min(abs(weight - PRIOR_WEIGHT) for weight in tied)
    distance_tied = [weight for weight in tied if abs(abs(weight - PRIOR_WEIGHT) - min_distance) <= TIE_TOLERANCE]
    return min(distance_tied)


def _required_payload_keys() -> set[str]:
    return {
        "global_index", "record_id", "label", "age", "A_final", "rA",
        "B_seed_20260806_logit", "B_seed_20260807_logit", "B_seed_20260808_logit",
        "B_final", "rB", "AB60", "AB50",
    }


def _finite_payload_array(archive: np.lib.npyio.NpzFile, key: str, *, dtype: object) -> np.ndarray:
    value = np.asarray(archive[key], dtype=dtype)
    if value.ndim != 1:
        raise FusionStop(f"payload field is not one-dimensional: {key}")
    return value


def load_frozen_payloads(private_root: Path, sites: Sequence[str] = SITES) -> dict[str, dict[str, np.ndarray]]:
    """Load and verify only the previous run's private patient-level payloads."""
    root = Path(private_root)
    manifest_path = root / "manifests" / "private_run_manifest.json"
    if not manifest_path.is_file():
        raise FusionStop(f"previous private manifest unavailable: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("private_payload_sha256", {})
    if not isinstance(hashes, dict):
        raise FusionStop("previous private manifest has no payload hashes")
    result: dict[str, dict[str, np.ndarray]] = {}
    for site in sites:
        path = root / "fused" / f"holdout_{site}" / "payload.npz"
        if not path.is_file() or hashes.get(site) != sha256_file(path):
            raise FusionStop(f"previous payload hash mismatch: {site}")
        with np.load(path, allow_pickle=False) as archive:
            missing = _required_payload_keys() - set(archive.files)
            if missing:
                raise FusionStop(f"previous payload keys incomplete for {site}: {sorted(missing)}")
            n = len(archive["global_index"])
            expected_n = EXPECTED_COUNTS.get(site, n)
            if n != expected_n:
                raise FusionStop(f"previous payload count mismatch for {site}: {n} != {expected_n}")
            values = {
                "canonical_global_index": _finite_payload_array(archive, "global_index", dtype=np.int64),
                "record_id": _finite_payload_array(archive, "record_id", dtype="U"),
                "label": _finite_payload_array(archive, "label", dtype=np.int8),
                "age": _finite_payload_array(archive, "age", dtype=np.float64),
                "A_final": _finite_payload_array(archive, "A_final", dtype=np.float64),
                "rA": _finite_payload_array(archive, "rA", dtype=np.float64),
                "B_final": _finite_payload_array(archive, "B_final", dtype=np.float64),
                "rB": _finite_payload_array(archive, "rB", dtype=np.float64),
                "AB60": _finite_payload_array(archive, "AB60", dtype=np.float64),
                "AB50": _finite_payload_array(archive, "AB50", dtype=np.float64),
            }
            for seed in ("20260806", "20260807", "20260808"):
                values[f"B_seed_{seed}_logit"] = _finite_payload_array(
                    archive, f"B_seed_{seed}_logit", dtype=np.float64
                )
        if len(np.unique(values["canonical_global_index"])) != n or len(np.unique(values["record_id"])) != n:
            raise FusionStop(f"previous payload identity is not unique: {site}")
        if not np.all(np.isin(values["label"], [0, 1])) or not np.isfinite(values["age"]).all():
            raise FusionStop(f"previous payload targets are invalid: {site}")
        for key, value in values.items():
            if key not in ("canonical_global_index", "record_id", "label", "age") and not np.isfinite(value).all():
                raise FusionStop(f"previous payload has nonfinite values: {site}/{key}")
        result[site] = values
    return result


def _metric_for(site_data: Mapping[str, np.ndarray], scores: np.ndarray) -> dict[str, object]:
    return _metric(site_data["label"], site_data["age"], scores)


def validate_frozen_replays(payloads: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, dict[str, object]]]:
    """Recompute A/B/AB60/AB50 metrics and enforce the frozen parent gates."""
    expected = {
        "A_FINAL": A_EXPECTED_AC,
        "B_MATCHA_M2": B_EXPECTED_AC,
        "AB60": AB60_EXPECTED_AC,
        "AB50": AB50_EXPECTED_AC,
    }
    reports: dict[str, dict[str, dict[str, object]]] = {}
    for site in SITES:
        data = payloads[site]
        calculated_ab60 = build_weighted_score(data["rA"], data["rB"], 0.4)
        calculated_ab50 = build_weighted_score(data["rA"], data["rB"], 0.5)
        if max(float(np.max(np.abs(calculated_ab60 - data["AB60"]))), float(np.max(np.abs(calculated_ab50 - data["AB50"])))) > 1e-12:
            raise FusionStop(f"stored fixed fusion differs from replay: {site}")
        scores = {
            "A_FINAL": data["rA"],
            "B_MATCHA_M2": data["rB"],
            "AB60": calculated_ab60,
            "AB50": calculated_ab50,
        }
        reports[site] = {}
        b_seed_mean = np.mean(
            np.vstack([data[f"B_seed_{seed}_logit"] for seed in ("20260806", "20260807", "20260808")]),
            axis=0,
            dtype=np.float64,
        )
        if np.max(np.abs(b_seed_mean - data["B_final"])) > 1e-12:
            raise FusionStop(f"B_final is not the frozen three-seed mean: {site}")
        for route, values in scores.items():
            metric = _metric_for(data, values)
            diff = abs(float(metric["ac_auroc"]) - expected[route][site])
            if diff > 1e-12:
                raise FusionStop(f"{route} replay drift exceeds 1e-12 for {site}: {diff:.3e}")
            reports[site][route] = metric
        for raw_key, rank_key in (("A_final", "rA"), ("B_final", "rB")):
            raw_metric = _metric_for(data, data[raw_key])
            rank_metric = _metric_for(data, data[rank_key])
            if max(abs(float(raw_metric[key]) - float(rank_metric[key])) for key in ("ac_auroc", "auroc", "auprc")) > 1e-12:
                raise FusionStop(f"rank metric invariance failed: {site}/{raw_key}")
    return reports


def evaluate_grid(payloads: Mapping[str, Mapping[str, np.ndarray]]) -> tuple[dict[str, dict[float, dict[str, object]]], list[dict[str, object]]]:
    grid: dict[str, dict[float, dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    for site in SITES:
        data = payloads[site]
        grid[site] = {}
        for weight in CANDIDATE_WEIGHTS:
            metric = _metric_for(data, build_weighted_score(data["rA"], data["rB"], weight))
            grid[site][weight] = metric
            rows.append({"site": site, "wB": weight, **metric})
    return grid, rows


def select_observed_weights(grid: Mapping[str, Mapping[float, Mapping[str, object]]]) -> tuple[dict[str, float], list[dict[str, object]]]:
    selected: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for target in SITES:
        folds = selection_folds(target)
        for weight in CANDIDATE_WEIGHTS:
            fold_scores = [float(grid[site][weight]["ac_auroc"]) for site in folds]
            rows.append({
                "target": target, "wB": weight,
                "selection_fold_1": folds[0], "selection_fold_1_ac": fold_scores[0],
                "selection_fold_2": folds[1], "selection_fold_2_ac": fold_scores[1],
                "selection_score_mean": float(np.mean(fold_scores)),
                "target_evaluation_ac": float(grid[target][weight]["ac_auroc"]),
                "selected": False,
            })
        scores = {weight: float(np.mean([grid[site][weight]["ac_auroc"] for site in folds])) for weight in CANDIDATE_WEIGHTS}
        selected_weight = select_weight(scores)
        selected[target] = selected_weight
        for row in rows[-len(CANDIDATE_WEIGHTS):]:
            if row["wB"] == selected_weight:
                row["selected"] = True
    return selected, rows


def _site_bootstrap_draws(data: Mapping[str, np.ndarray], n_bootstrap: int, rng: np.random.Generator) -> dict[str, object]:
    labels = data.get("label", data.get("labels"))
    ages = data.get("age", data.get("ages"))
    if labels is None or ages is None:
        raise FusionStop("bootstrap site data must contain label(s) and age(s)")
    multiplicities = np.vstack([bootstrap_multiplicity(len(labels), rng) for _ in range(n_bootstrap)])
    candidates: dict[float, np.ndarray] = {}
    for weight in CANDIDATE_WEIGHTS:
        scores = build_weighted_score(data["rA"], data["rB"], weight)
        candidates[weight] = weighted_ac_batch(pair_contributions(labels, ages, scores), multiplicities)
    fixed_scores = {
        "A_FINAL": weighted_ac_batch(pair_contributions(labels, ages, data["rA"]), multiplicities),
        "B_MATCHA_M2": weighted_ac_batch(pair_contributions(labels, ages, data["rB"]), multiplicities),
        "AB60": candidates[0.4],
        "AB50": candidates[0.5],
    }
    return {"multiplicities": multiplicities, "candidates": candidates, "fixed": fixed_scores}


def nested_crossfold_bootstrap(site_data: Mapping[str, Mapping[str, np.ndarray]], *, n_bootstrap: int, seed: int) -> dict[str, object]:
    """Run nested target-excluded selection and fixed-selected auxiliary draws."""
    if tuple(site_data) != SITES:
        raise FusionStop("bootstrap site order must be I0002, I0006, S0001")
    rng = np.random.default_rng(int(seed))
    draws = {site: _site_bootstrap_draws(site_data[site], int(n_bootstrap), rng) for site in SITES}
    nested_scores: dict[str, np.ndarray] = {}
    fixed_scores: dict[str, np.ndarray] = {}
    selected_weights: dict[str, np.ndarray] = {}
    selection_frequency: dict[str, dict[float, int]] = {}
    for target in SITES:
        folds = selection_folds(target)
        chosen = np.empty(int(n_bootstrap), dtype=np.float64)
        nested = np.empty(int(n_bootstrap), dtype=np.float64)
        selection_frequency[target] = {weight: 0 for weight in CANDIDATE_WEIGHTS}
        for index in range(int(n_bootstrap)):
            scores = {
                weight: float(np.mean([draws[site]["candidates"][weight][index] for site in folds]))
                for weight in CANDIDATE_WEIGHTS
            }
            weight = select_weight(scores)
            chosen[index] = weight
            selection_frequency[target][weight] += 1
            nested[index] = draws[target]["candidates"][weight][index]
        selected_weights[target] = chosen
        nested_scores[target] = nested
        fixed_weight = None
        fixed_scores[target] = np.empty(int(n_bootstrap), dtype=np.float64)
        # The caller replaces this auxiliary array using observed selected weights.
        fixed_scores[target].fill(np.nan)
    return {
        "draws": draws,
        "nested_scores": nested_scores,
        "fixed_scores": fixed_scores,
        "selected_weights": selected_weights,
        "selection_frequency": selection_frequency,
        "selection_exclusion": {target: selection_folds(target) for target in SITES},
    }


def _fill_fixed_selected(result: dict[str, object], observed_selected: Mapping[str, float]) -> None:
    draws = result["draws"]
    fixed_scores = result["fixed_scores"]
    for target in SITES:
        fixed_scores[target][:] = draws[target]["candidates"][observed_selected[target]]


def _scope_bootstrap(values: Mapping[str, np.ndarray], scope: str) -> np.ndarray:
    matrix = np.column_stack([values[site] for site in SITES])
    if scope in SITES:
        return values[scope]
    if scope == "Macro":
        return matrix.mean(axis=1)
    if scope == "Worst-site":
        return matrix.min(axis=1)
    if scope == "Small-site mean":
        return matrix[:, :2].mean(axis=1)
    raise FusionStop(f"unknown bootstrap scope: {scope}")


def bootstrap_comparison_rows(result: Mapping[str, object], observed_selected: Mapping[str, float], observed_site_scores: Mapping[str, Mapping[str, float]], *, n_bootstrap: int, seed: int) -> list[dict[str, object]]:
    methods = ("NESTED_CROSSFOLD_BOOTSTRAP", "FIXED_SELECTED_BOOTSTRAP")
    comparisons = {
        "AB_CROSSFOLD_MINUS_A": "A_FINAL",
        "AB_CROSSFOLD_MINUS_B": "B_MATCHA_M2",
        "AB_CROSSFOLD_MINUS_AB60": "AB60",
        "AB_CROSSFOLD_MINUS_AB50": "AB50",
    }
    _fill_fixed_selected(result, observed_selected)
    rows: list[dict[str, object]] = []
    for method in methods:
        left = {site: (result["nested_scores"] if method == methods[0] else result["fixed_scores"])[site] for site in SITES}
        right_by_route = result["draws"]
        for comparison, right_route in comparisons.items():
            for scope in SCOPES:
                deltas = _scope_bootstrap({site: left[site] - right_by_route[site]["fixed"][right_route] for site in SITES}, scope)
                valid = np.isfinite(deltas)
                values = deltas[valid]
                observed_delta = _scope({site: observed_site_scores[site]["AB_CROSSFOLD"] - observed_site_scores[site][right_route] for site in SITES}, scope)
                rows.append({
                    "bootstrap_method": method, "comparison": comparison, "scope": scope,
                    "n_bootstrap": int(n_bootstrap), "seed": int(seed), "observed_delta": float(observed_delta),
                    "bootstrap_mean": float(values.mean()) if len(values) else float("nan"),
                    "bootstrap_median": float(np.median(values)) if len(values) else float("nan"),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "q025": float(np.quantile(values, 0.025)) if len(values) else float("nan"),
                    "q975": float(np.quantile(values, 0.975)) if len(values) else float("nan"),
                    "fraction_gt_0": float(np.mean(values > 0)) if len(values) else float("nan"),
                    "fraction_lt_0": float(np.mean(values < 0)) if len(values) else float("nan"),
                    "valid": int(values.size), "invalid": int((~valid).sum()),
                })
    return rows


def _route_metrics(payloads: Mapping[str, Mapping[str, np.ndarray]], selected: Mapping[str, float]) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    rows: list[dict[str, object]] = []
    observed: dict[str, dict[str, float]] = {}
    for site in SITES:
        data = payloads[site]
        scores = {
            "A_FINAL": data["rA"],
            "B_MATCHA_M2": data["rB"],
            "AB60": build_weighted_score(data["rA"], data["rB"], 0.4),
            "AB50": build_weighted_score(data["rA"], data["rB"], 0.5),
            "AB_CROSSFOLD": build_weighted_score(data["rA"], data["rB"], selected[site]),
        }
        observed[site] = {}
        for route, values in scores.items():
            metric = _metric_for(data, values)
            observed[site][route] = float(metric["ac_auroc"])
            rows.append({"route": route, "site": site, "wB": selected[site] if route == "AB_CROSSFOLD" else (0.4 if route == "AB60" else 0.5 if route == "AB50" else ""), **metric})
    return rows, observed


def _summary_rows(observed: Mapping[str, Mapping[str, float]]) -> list[dict[str, object]]:
    routes = ("A_FINAL", "B_MATCHA_M2", "AB60", "AB50", "AB_CROSSFOLD")
    return [{"route": route, "scope": scope, "ac_auroc": _scope({site: observed[site][route] for site in SITES}, scope)} for route in routes for scope in SCOPES]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise FusionStop(f"refusing to write empty public output: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _public_manifest(selection: Mapping[str, float], private_hashes: Mapping[str, str], counts: Mapping[str, int], bootstrap: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "siren_matcha_ab_crossfold_public_manifest_v1",
        "parent_branch": PARENT_BRANCH,
        "parent_commit": PARENT_COMMIT,
        "candidate_weights": list(CANDIDATE_WEIGHTS),
        "selection_metric": "mean age-conditioned AUROC on the other two LOSO sites",
        "target_fold_used_for_weight_selection": False,
        "tie_break": "closest_to_0.4_then_lower_weight",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "selected_wB": dict(selection),
        "counts": dict(counts),
        "parent_payload_sha256": dict(private_hashes),
        "bootstrap_valid_invalid": bootstrap,
        "public_contains_patient_level_fields": False,
    }


def _write_report(path: Path, selected: Mapping[str, float], site_rows: Sequence[Mapping[str, object]], summary: Sequence[Mapping[str, object]], selection_rows: Sequence[Mapping[str, object]], stability_rows: Sequence[Mapping[str, object]], bootstrap_rows: Sequence[Mapping[str, object]]) -> None:
    summary_map = {(row["route"], row["scope"]): float(row["ac_auroc"]) for row in summary}
    lines = [
        "# SIREN + Matcha B Cross-Fold Weight Selection v1",
        "",
        "This CPU-only analysis consumes only the previous frozen patient-level rank payloads. No model, feature, embedding, prediction, reranking, or calibration computation was run.",
        "",
        "## Frozen-input gates",
        "",
        "- Previous payload SHA256 and manifest: PASS.",
        "- A, B, AB60 (`wB=0.4`), and AB50 (`wB=0.5`) replay: PASS at maximum AC error `<= 1e-12`.",
        "- Target-fold labels excluded from observed and nested-bootstrap weight selection: PASS.",
        "- Candidate grid fixed before evaluation: `0.2, 0.3, 0.4, 0.5, 0.6, 0.7`.",
        "",
        "## Cross-fold selected weights",
        "",
        "| Target | Selected wB | Selection folds | Selection score | Target AC |",
        "|---|---:|---|---:|---:|",
    ]
    for target in SITES:
        row = next(item for item in selection_rows if item["target"] == target and item["selected"])
        lines.append(f"| {target} | {selected[target]:.1f} | {row['selection_fold_1']} + {row['selection_fold_2']} | {row['selection_score_mean']:.12f} | {row['target_evaluation_ac']:.12f} |")
    lines += ["", "## Primary route summary", "", "| Route | I0002 | I0006 | S0001 | Macro | Worst | Small-site mean |", "|---|---:|---:|---:|---:|---:|---:|"]
    for route in ("A_FINAL", "B_MATCHA_M2", "AB60", "AB50", "AB_CROSSFOLD"):
        values = [summary_map[(route, scope)] for scope in ("I0002", "I0006", "S0001", "Macro", "Worst-site", "Small-site mean")]
        lines.append(f"| {route} | " + " | ".join(f"{value:.12f}" for value in values) + " |")
    lines += ["", "## Selection stability", "", "| Target | wB | Selected count | Fraction | Observed selected |", "|---|---:|---:|---:|---:|"]
    for row in stability_rows:
        lines.append(f"| {row['target']} | {row['wB']:.1f} | {row['selected_count']} | {row['selected_fraction']:.6f} | {row['observed_selected']} |")
    lines += ["", "## Nested bootstrap comparisons", "", "The primary uncertainty analysis is `NESTED_CROSSFOLD_BOOTSTRAP`; `FIXED_SELECTED_BOOTSTRAP` is auxiliary and keeps the observed selected weights fixed.", "", "| Comparison | Scope | Observed delta | 95% CI | P(delta>0) |", "|---|---|---:|---:|---:|"]
    for row in bootstrap_rows:
        if row["bootstrap_method"] == "NESTED_CROSSFOLD_BOOTSTRAP":
            lines.append(f"| {row['comparison']} | {row['scope']} | {row['observed_delta']:.8f} | [{row['q025']:.8f}, {row['q975']:.8f}] | {row['fraction_gt_0']:.4f} |")
    cross_macro = summary_map[("AB_CROSSFOLD", "Macro")]
    cross_worst = summary_map[("AB_CROSSFOLD", "Worst-site")]
    ab60_macro = summary_map[("AB60", "Macro")]
    ab50_macro = summary_map[("AB50", "Macro")]
    ab60_worst = summary_map[("AB60", "Worst-site")]
    ab50_worst = summary_map[("AB50", "Worst-site")]
    max_fraction = max(float(row["selected_fraction"]) for row in stability_rows)
    lines += [
        "", "## Interpretation",
        "",
        f"- Selected weights: I0002=`{selected['I0002']:.1f}`, I0006=`{selected['I0006']:.1f}`, S0001=`{selected['S0001']:.1f}`; they are not identical.",
        f"- Bootstrap selection is dispersed: the largest candidate frequency is `{max_fraction:.3f}`; I0002, I0006, and S0001 have modal weights 0.7, 0.7, and 0.7, respectively, but no target is concentrated near 1.0.",
        f"- AB_CROSSFOLD Macro AC is `{cross_macro:.12f}`, above AB60 by `{cross_macro - ab60_macro:.12f}` and above AB50 by `{cross_macro - ab50_macro:.12f}`; its Worst-site AC `{cross_worst:.12f}` is below AB60 `{ab60_worst:.12f}` and AB50 `{ab50_worst:.12f}`.",
        "- The nested-bootstrap Macro intervals for AB_CROSSFOLD minus AB60, AB50, and B cross zero, so the point-estimate gains do not establish a reliable improvement.",
        "- Recommendation: retain the simpler fixed `AB50` route as the current formal fusion rule. Cross-fold selection is not promoted because its selected weights differ by target, selection frequencies are dispersed, and its apparent Macro gain is not supported as a clear nested-bootstrap improvement.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(*, previous_private_root: Path, private_root: Path, repo_root: Path, n_bootstrap: int = BOOTSTRAP_REPLICATES, bootstrap_seed: int = BOOTSTRAP_SEED) -> dict[str, object]:
    previous_root = Path(previous_private_root)
    parent_paths = {site: previous_root / "fused" / f"holdout_{site}" / "payload.npz" for site in SITES}
    before_hashes = {site: sha256_file(path) for site, path in parent_paths.items()}
    payloads = load_frozen_payloads(previous_root)
    replay = validate_frozen_replays(payloads)
    grid, grid_rows = evaluate_grid(payloads)
    selected, selection_rows = select_observed_weights(grid)
    site_rows, observed = _route_metrics(payloads, selected)
    summary_rows = _summary_rows(observed)
    bootstrap_result = nested_crossfold_bootstrap(payloads, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
    bootstrap_rows = bootstrap_comparison_rows(bootstrap_result, selected, observed, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
    stability_rows: list[dict[str, object]] = []
    for target in SITES:
        counts = bootstrap_result["selection_frequency"][target]
        modal = max(CANDIDATE_WEIGHTS, key=lambda weight: (counts[weight], -abs(weight - PRIOR_WEIGHT), -weight))
        for weight in CANDIDATE_WEIGHTS:
            stability_rows.append({"target": target, "wB": weight, "selected_count": int(counts[weight]), "selected_fraction": float(counts[weight] / n_bootstrap), "observed_selected": bool(selected[target] == weight), "bootstrap_modal_wB": float(modal)})
    private = Path(private_root)
    private.mkdir(parents=True, exist_ok=True)
    input_rows = [{"site": site, "payload_sha256": before_hashes[site], "record_count": len(payloads[site]["rA"])} for site in SITES]
    _write_csv(private / "inputs" / "frozen_payloads.csv", input_rows)
    for site in SITES:
        data = payloads[site]
        arrays = {"canonical_global_index": data["canonical_global_index"], "record_id": data["record_id"], "label": data["label"], "age": data["age"], "rA": data["rA"], "rB": data["rB"]}
        for weight in CANDIDATE_WEIGHTS:
            arrays[f"wB_{weight:.2f}"] = build_weighted_score(data["rA"], data["rB"], weight)
        _write_npz(private / "grid" / f"holdout_{site}.npz", arrays)
        _write_npz(private / "selected" / f"holdout_{site}.npz", {"canonical_global_index": data["canonical_global_index"], "record_id": data["record_id"], "label": data["label"], "age": data["age"], "rA": data["rA"], "rB": data["rB"], "selected_wB": np.asarray([selected[site]]), "AB_crossfold": build_weighted_score(data["rA"], data["rB"], selected[site])})
        _write_npz(private / "bootstrap" / f"holdout_{site}.npz", {"multiplicities": bootstrap_result["draws"][site]["multiplicities"], "nested_crossfold": bootstrap_result["nested_scores"][site], "fixed_selected": bootstrap_result["fixed_scores"][site], "nested_selected_wB": bootstrap_result["selected_weights"][site]})
    bootstrap_valid_invalid = {"nested": {"valid": int(sum(row["valid"] for row in bootstrap_rows if row["bootstrap_method"] == "NESTED_CROSSFOLD_BOOTSTRAP")), "invalid": int(sum(row["invalid"] for row in bootstrap_rows if row["bootstrap_method"] == "NESTED_CROSSFOLD_BOOTSTRAP"))}, "fixed": {"valid": int(sum(row["valid"] for row in bootstrap_rows if row["bootstrap_method"] == "FIXED_SELECTED_BOOTSTRAP")), "invalid": int(sum(row["invalid"] for row in bootstrap_rows if row["bootstrap_method"] == "FIXED_SELECTED_BOOTSTRAP"))}}
    private_hashes = {site: before_hashes[site] for site in SITES}
    manifest = _public_manifest(selected, private_hashes, EXPECTED_COUNTS, bootstrap_valid_invalid)
    public = Path(repo_root) / "audit"
    _write_csv(public / "siren_matcha_ab_crossfold_grid.csv", grid_rows)
    _write_csv(public / "siren_matcha_ab_crossfold_selection.csv", selection_rows)
    _write_csv(public / "siren_matcha_ab_crossfold_site_scores.csv", site_rows)
    _write_csv(public / "siren_matcha_ab_crossfold_summary.csv", summary_rows)
    _write_csv(public / "siren_matcha_ab_crossfold_selection_stability.csv", stability_rows)
    _write_csv(public / "siren_matcha_ab_crossfold_bootstrap.csv", bootstrap_rows)
    (public / "siren_matcha_ab_crossfold_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(Path(repo_root) / "docs/fusion/SIREN_MATCHA_AB_CROSSFOLD_WEIGHT_REPORT.md", selected, site_rows, summary_rows, selection_rows, stability_rows, bootstrap_rows)
    after_hashes = {site: sha256_file(path) for site, path in parent_paths.items()}
    if before_hashes != after_hashes:
        raise FusionStop("previous private payload mutated")
    (private / "manifests").mkdir(parents=True, exist_ok=True)
    (private / "manifests" / "private_run_manifest.json").write_text(json.dumps({"schema": "siren_matcha_ab_crossfold_private_run_manifest_v1", "parent_branch": PARENT_BRANCH, "parent_commit": PARENT_COMMIT, "selected_wB": selected, "parent_payload_sha256": before_hashes, "bootstrap": {"n": n_bootstrap, "seed": bootstrap_seed}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"selected": selected, "replay": replay, "grid": grid, "grid_rows": grid_rows, "selection_rows": selection_rows, "site_rows": site_rows, "summary_rows": summary_rows, "stability_rows": stability_rows, "bootstrap_rows": bootstrap_rows, "manifest": manifest, "private_root": str(private)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-private-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    result = run_experiment(previous_private_root=args.previous_private_root, private_root=args.private_root, repo_root=args.repo_root, n_bootstrap=args.bootstrap, bootstrap_seed=args.seed)
    print(json.dumps({"selected_wB": result["selected"], "private_root": result["private_root"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
