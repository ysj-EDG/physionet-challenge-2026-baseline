"""Post-hoc analysis for the CORE_NO_CAISR20 structural ablation.

Only the new V5 prediction units are loaded from the private route root.  The
frozen V1--V4 units, CORE, CAISR20 and CURRENT_FUSION parents are read-only.
No score is exposed in the public files written by this module.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from reproduction.siren_a.g5.core import HOLDOUTS, MODEL_TYPES, VIEWS, load_eval_targets, mean_ranks, score_site
from .e0e3_stability import bootstrap_multiplicity, pair_contributions, weighted_ac_batch
from .route_fusion_50_50 import fuse_scores, route_scope_delta, route_scope_value
from .v5_family_loo_analysis import OFFICIAL, SCOPES, SEEDS, _old_prediction, _rows_by_key

ROUTE_ID = "CORE_NO_CAISR20"
CURRENT_FUSION = "CURRENT_FUSION"
DEDUP_FUSION = "DEDUP_FUSION"
COMPARISONS = ("CORE_NO_CAISR20_MINUS_CORE", "DEDUP_FUSION_MINUS_CURRENT_FUSION")
ALL_ROUTES = ("CAISR20", "CORE", ROUTE_ID, CURRENT_FUSION, DEDUP_FUSION)


class CoreNoCAISRAnalysisStop(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_unit(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(Path(path) / "payload.npz", allow_pickle=False) as archive:
            indices = np.asarray(archive["test_global_indices"], dtype=np.int64)
            scores = np.asarray(archive["scores"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise CoreNoCAISRAnalysisStop(f"invalid prediction unit: {path}") from exc
    if indices.ndim != 1 or scores.ndim != 1 or indices.shape != scores.shape or len(np.unique(indices)) != len(indices) or not np.isfinite(scores).all():
        raise CoreNoCAISRAnalysisStop(f"prediction unit shape/identity mismatch: {path}")
    return indices, scores


def _new_v5_unit(root: Path, seed: int, holdout: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    return _load_unit(Path(root) / "predictions" / f"seed{seed}" / f"holdout_{holdout}" / "inductive" / model)


def _old_v1_v4(old_rows: Mapping, holdout: str, seed: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str, str]]]:
    columns = []
    keys = []
    canonical = None
    for view in VIEWS[:4]:
        for model in MODEL_TYPES:
            scores, indices = _old_prediction(old_rows[(holdout, "inductive", view, model, int(seed))])
            if canonical is None:
                canonical = np.asarray(indices, dtype=np.int64).copy()
            elif not np.array_equal(canonical, indices):
                raise CoreNoCAISRAnalysisStop("frozen V1-V4 row identity mismatch")
            columns.append(np.asarray(scores, dtype=np.float64)); keys.append((int(seed), view, model))
    return canonical, np.column_stack(columns).astype(np.float64), keys


def _new_route_matrix(old_rows: Mapping, new_root: Path, holdout: str, seed: int | None = None):
    seeds = (int(seed),) if seed is not None else SEEDS
    columns = []
    keys = []
    canonical = None
    for current_seed in seeds:
        indices, frozen, frozen_keys = _old_v1_v4(old_rows, holdout, current_seed)
        if canonical is None:
            canonical = indices
        elif not np.array_equal(canonical, indices):
            raise CoreNoCAISRAnalysisStop("frozen V1-V4 row identity differs across seeds")
        columns.extend(frozen.T)
        keys.extend(frozen_keys)
        for model in MODEL_TYPES:
            new_indices, scores = _new_v5_unit(new_root, current_seed, holdout, model)
            if not np.array_equal(canonical, new_indices):
                raise CoreNoCAISRAnalysisStop("new V5 row identity differs from frozen V1-V4")
            columns.append(scores); keys.append((int(current_seed), "V5", model))
    matrix = np.column_stack(columns).astype(np.float64)
    expected = 15 if seed is not None else 225
    if matrix.shape[1] != expected:
        raise CoreNoCAISRAnalysisStop(f"CORE_NO_CAISR20 column count mismatch: {matrix.shape}")
    return canonical, matrix, tuple(keys)


def _core_matrix(old_rows: Mapping, core_root: Path, holdout: str, seed: int | None = None):
    columns = []; keys = []; canonical = None
    seeds = (int(seed),) if seed is not None else SEEDS
    for current_seed in seeds:
        for view in VIEWS:
            for model in MODEL_TYPES:
                if view == "V5":
                    indices, scores = _load_unit(Path(core_root) / "predictions" / "CORE_COH_EMG" / f"seed{current_seed}" / f"holdout_{holdout}" / "inductive" / model)
                else:
                    scores, indices = _old_prediction(old_rows[(holdout, "inductive", view, model, int(current_seed))])
                if canonical is None: canonical = np.asarray(indices, dtype=np.int64).copy()
                elif not np.array_equal(canonical, indices): raise CoreNoCAISRAnalysisStop("CORE row identity mismatch")
                columns.append(np.asarray(scores, dtype=np.float64)); keys.append((int(current_seed), view, model))
    matrix = np.column_stack(columns).astype(np.float64)
    if matrix.shape[1] != (15 if seed is not None else 225): raise CoreNoCAISRAnalysisStop("CORE width mismatch")
    return canonical, matrix, tuple(keys)


def _caisr_matrix(root: Path, holdout: str, seed: int | None = None):
    seeds = (int(seed),) if seed is not None else SEEDS
    columns = []; canonical = None
    for current_seed in seeds:
        indices, scores = _load_unit(Path(root) / "predictions" / f"seed{current_seed}" / f"holdout_{holdout}")
        if canonical is None: canonical = indices
        elif not np.array_equal(canonical, indices): raise CoreNoCAISRAnalysisStop("CAISR row identity mismatch")
        columns.append(scores)
    return canonical, np.column_stack(columns).astype(np.float64)


def _rank_fuse(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all(): raise CoreNoCAISRAnalysisStop("cannot rank nonfinite matrix")
    return np.asarray(mean_ranks(values)[1], dtype=np.float64)


def _score(metadata, indices, scores):
    labels, ages = load_eval_targets(metadata, indices)
    return score_site(labels, scores, ages, OFFICIAL)


def _summary_values(site_values: Mapping[str, float]) -> dict[str, float]:
    return {scope: route_scope_value(site_values, scope) for scope in SCOPES}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: raise CoreNoCAISRAnalysisStop(f"empty public output: {path}")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def run_core_no_caisr20_analysis(*, metadata, reference_manifest: Path, core_root: Path, caisr_root: Path, current_fusion_root: Path, new_root: Path, public_root: Path, private_root: Path, n_bootstrap: int = 2000, bootstrap_seed: int = 20261023) -> dict[str, object]:
    """Build route-level public summaries from frozen and new prediction units."""
    old_rows = _rows_by_key(Path(reference_manifest), "E0")
    public_root = Path(public_root); private_root = Path(private_root)
    public_root.mkdir(parents=True, exist_ok=True); private_root.mkdir(parents=True, exist_ok=True)
    site_rows = []; route_data = {}; seed_data = {}
    for holdout in HOLDOUTS:
        idx_new, new_matrix, new_keys = _new_route_matrix(old_rows, new_root, holdout)
        idx_core, core_matrix, core_keys = _core_matrix(old_rows, core_root, holdout)
        if not np.array_equal(idx_new, idx_core): raise CoreNoCAISRAnalysisStop("new and CORE test identity mismatch")
        if tuple(new_keys[i] for i in range(180)) != tuple(core_keys[i] for i in range(180)):
            raise CoreNoCAISRAnalysisStop("frozen V1-V4 key order mismatch")
        no_caisr = _rank_fuse(new_matrix)
        core = _rank_fuse(core_matrix)
        idx_c, caisr_matrix = _caisr_matrix(caisr_root, holdout)
        if not np.array_equal(idx_new, idx_c): raise CoreNoCAISRAnalysisStop("CAISR test identity mismatch")
        caisr = _rank_fuse(caisr_matrix)
        idx_f, current = _load_unit(Path(current_fusion_root) / "final" / f"holdout_{holdout}")
        if not np.array_equal(idx_new, idx_f): raise CoreNoCAISRAnalysisStop("CURRENT_FUSION test identity mismatch")
        expected_current = fuse_scores(core, caisr)
        if not np.array_equal(current, expected_current): raise CoreNoCAISRAnalysisStop("frozen CURRENT_FUSION replay mismatch")
        dedup = fuse_scores(no_caisr, caisr)
        route_vectors = {ROUTE_ID: no_caisr, "CORE": core, "CAISR20": caisr, CURRENT_FUSION: current, DEDUP_FUSION: dedup}
        route_data[holdout] = {"indices": idx_new, "vectors": route_vectors}
        for route, vector in route_vectors.items():
            metrics = _score(metadata, idx_new, vector)
            site_rows.append({"route": route, "holdout_site": holdout, "n_records": int(len(idx_new)), "ac_auroc": float(metrics["ac_auroc"]), "auroc": float(metrics["auroc"]), "auprc": float(metrics["auprc"]), "status": "PASS"})
        for seed in SEEDS:
            idx_s, m_s, k_s = _new_route_matrix(old_rows, new_root, holdout, seed)
            idx_cs, cm_s, _ = _core_matrix(old_rows, core_root, holdout, seed)
            idx_as, am_s = _caisr_matrix(caisr_root, holdout, seed)
            if not (np.array_equal(idx_s, idx_cs) and np.array_equal(idx_s, idx_as)): raise CoreNoCAISRAnalysisStop("seed test identity mismatch")
            no_seed = _rank_fuse(m_s); core_seed = _rank_fuse(cm_s); caisr_seed = _rank_fuse(am_s)
            seed_data[(holdout, seed)] = {"CORE_NO_CAISR20": no_seed, "CORE": core_seed, "CAISR20": caisr_seed, "CURRENT_FUSION": fuse_scores(core_seed, caisr_seed), "DEDUP_FUSION": fuse_scores(no_seed, caisr_seed)}
    # Aggregate route summaries and explicitly registered comparisons.
    site_by_route = {route: {h: float(next(r["ac_auroc"] for r in site_rows if r["route"] == route and r["holdout_site"] == h)) for h in HOLDOUTS} for route in ALL_ROUTES}
    summary = []
    for route in ALL_ROUTES:
        for scope in SCOPES:
            summary.append({"row_type":"route_summary", "route":route, "scope":scope, "ac_auroc":route_scope_value(site_by_route[route], scope)})
    for name, left, right in ((COMPARISONS[0], ROUTE_ID, "CORE"), (COMPARISONS[1], DEDUP_FUSION, CURRENT_FUSION), ("DEDUP_FUSION_MINUS_CORE", DEDUP_FUSION, "CORE"), ("DEDUP_FUSION_MINUS_CAISR20", DEDUP_FUSION, "CAISR20")):
        for scope in SCOPES:
            summary.append({"row_type":"comparison", "comparison":name, "left_route":left, "right_route":right, "scope":scope, "delta":route_scope_delta(site_by_route[left], site_by_route[right], scope)})
    stability = []
    for comparison, left, right in ((COMPARISONS[0], ROUTE_ID, "CORE"), (COMPARISONS[1], DEDUP_FUSION, CURRENT_FUSION)):
        for scope in SCOPES:
            vals=[]
            for seed in SEEDS:
                left_sites={h:float(_score(metadata, route_data[h]["indices"], seed_data[(h,seed)][left])["ac_auroc"]) for h in HOLDOUTS}
                right_sites={h:float(_score(metadata, route_data[h]["indices"], seed_data[(h,seed)][right])["ac_auroc"]) for h in HOLDOUTS}
                vals.append(route_scope_delta(left_sites, right_sites, scope))
            arr=np.asarray(vals, dtype=np.float64)
            stability.append({"comparison":comparison,"scope":scope,"n_seeds":15,"n_positive":int((arr>0).sum()),"n_zero":int((arr==0).sum()),"n_negative":int((arr<0).sum()),"positive_fraction":float((arr>0).mean()),"mean_delta":float(arr.mean()),"median_delta":float(np.median(arr)),"q25":float(np.quantile(arr,.25)),"q75":float(np.quantile(arr,.75)),"min":float(arr.min()),"max":float(arr.max())})
    bootstrap = []
    rng = np.random.default_rng(int(bootstrap_seed)); draws = {}
    for holdout in HOLDOUTS:
        idx = route_data[holdout]["indices"]; labels, ages = load_eval_targets(metadata, idx)
        mult = np.vstack([bootstrap_multiplicity(len(idx), rng) for _ in range(int(n_bootstrap))])
        site_draws = {}
        for route, scores in route_data[holdout]["vectors"].items():
            site_draws[route] = weighted_ac_batch(pair_contributions(labels, ages, scores), mult)
        draws[holdout] = site_draws
    for comparison, left, right in ((COMPARISONS[0], ROUTE_ID, "CORE"), (COMPARISONS[1], DEDUP_FUSION, CURRENT_FUSION)):
        for scope in SCOPES:
            columns = np.column_stack([draws[h][left] - draws[h][right] for h in HOLDOUTS])
            delta = columns[:, HOLDOUTS.index(scope)] if scope in HOLDOUTS else columns.mean(1) if scope == "Macro" else columns.min(1) if scope == "Worst-site" else columns[:, :2].mean(1)
            finite=np.isfinite(delta); valid=delta[finite]
            if len(valid) != n_bootstrap: raise CoreNoCAISRAnalysisStop("invalid bootstrap replicate")
            bootstrap.append({"comparison":comparison,"scope":scope,"n_bootstrap":int(n_bootstrap),"bootstrap_seed":int(bootstrap_seed),"valid":int(valid.size),"invalid":int((~finite).sum()),"mean_delta":float(valid.mean()),"median_delta":float(np.median(valid)),"q025":float(np.quantile(valid,.025)),"q975":float(np.quantile(valid,.975)),"fraction_gt_0":float(np.mean(valid>0)),"fraction_lt_0":float(np.mean(valid<0))})
    _write_csv(public_root / "audit/featurelab_core_no_caisr20_site_scores.csv", site_rows)
    _write_csv(public_root / "audit/featurelab_core_no_caisr20_summary.csv", summary)
    _write_csv(public_root / "audit/featurelab_core_no_caisr20_seed_stability.csv", stability)
    _write_csv(public_root / "audit/featurelab_core_no_caisr20_bootstrap.csv", bootstrap)
    _write_csv(public_root / "audit/featurelab_dedup_fusion_50_50_summary.csv", [r for r in summary if r.get("route") in (CURRENT_FUSION, DEDUP_FUSION) or r.get("comparison") in (COMPARISONS[1], "DEDUP_FUSION_MINUS_CORE", "DEDUP_FUSION_MINUS_CAISR20")])
    _write_csv(public_root / "audit/featurelab_dedup_fusion_50_50_seed_stability.csv", [r for r in stability if r["comparison"] == COMPARISONS[1]])
    _write_csv(public_root / "audit/featurelab_dedup_fusion_50_50_bootstrap.csv", [r for r in bootstrap if r["comparison"] == COMPARISONS[1]])
    manifest = {"schema":"featurelab_core_no_caisr20_analysis_v1","route":ROUTE_ID,"new_v5_models":90,"new_v5_predictions":135,"final_width":225,"removed_caisr20":20,"current_fusion_replayed":True,"dedup_formula":"0.5*CORE_NO_CAISR20_FINAL+0.5*CAISR20_FINAL","post_fusion_rerank":False,"bootstrap_seed":int(bootstrap_seed),"bootstrap_replicates":int(n_bootstrap),"raw_source_independence":"NOT_CLAIMED"}
    (private_root / "analysis_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"site_rows":len(site_rows),"summary_rows":len(summary),"seed_rows":len(stability),"bootstrap_rows":len(bootstrap),"bootstrap_invalid":0,"final_width":225}


__all__ = ["COMPARISONS", "CoreNoCAISRAnalysisStop", "run_core_no_caisr20_analysis"]
