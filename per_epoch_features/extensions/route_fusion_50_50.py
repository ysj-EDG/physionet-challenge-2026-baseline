"""Fixed 50/50 prediction-level fusion for frozen CORE and CAISR20 routes.

This module contains no fitting, prediction, feature extraction, or weight search.
Parent scores are validated for exact identity before arithmetic fusion; the
post-fusion score is never ranked or otherwise transformed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

SITES = ("I0002", "I0006", "S0001")
SEEDS = tuple(range(42, 57))
CORE_WEIGHT = 0.5
CAISR_WEIGHT = 0.5


class FusionStop(RuntimeError):
    """A frozen parent identity or score contract failed."""


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise FusionStop(f"{name} must be a one-dimensional vector")
    if array.dtype != np.dtype(np.float64):
        array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise FusionStop(f"{name} contains nonfinite scores")
    return array


def validate_parent_pair(core_indices: np.ndarray, caisr_indices: np.ndarray,
                         core_scores: np.ndarray, caisr_scores: np.ndarray,
                         *, holdout: str, seed: int | None = None,
                         caisr_seed: int | None = None) -> None:
    """Require exact row identity and finite score vectors for one paired unit."""
    if holdout not in SITES:
        raise FusionStop(f"unknown holdout site: {holdout}")
    left_idx = np.asarray(core_indices)
    right_idx = np.asarray(caisr_indices)
    if left_idx.ndim != 1 or right_idx.ndim != 1 or left_idx.dtype.kind not in "iu" or right_idx.dtype.kind not in "iu":
        raise FusionStop("parent indices must be one-dimensional integer vectors")
    if not np.array_equal(left_idx, right_idx):
        raise FusionStop(f"parent row identity mismatch for {holdout}")
    if len(np.unique(left_idx)) != len(left_idx):
        raise FusionStop(f"duplicate parent test indices for {holdout}")
    left = _vector(core_scores, "CORE scores")
    right = _vector(caisr_scores, "CAISR scores")
    if left.shape != right.shape or left.shape != left_idx.shape:
        raise FusionStop(f"parent score/index length mismatch for {holdout}")
    if seed is not None and seed not in SEEDS:
        raise FusionStop(f"unsupported seed: {seed}")
    if caisr_seed is not None and (caisr_seed not in SEEDS or seed != caisr_seed):
        raise FusionStop("parent seed identity mismatch")


def fuse_scores(core_scores: np.ndarray, caisr_scores: np.ndarray) -> np.ndarray:
    """Return exactly 0.5*CORE + 0.5*CAISR, with no post-fusion rerank."""
    core = _vector(core_scores, "CORE scores")
    caisr = _vector(caisr_scores, "CAISR scores")
    if core.shape != caisr.shape:
        raise FusionStop("CORE and CAISR score lengths differ")
    fused = (CORE_WEIGHT * core + CAISR_WEIGHT * caisr).astype(np.float64, copy=False)
    if not np.isfinite(fused).all():
        raise FusionStop("fusion produced nonfinite scores")
    return fused


def build_seed_fusion(core_scores: np.ndarray, caisr_scores: np.ndarray) -> np.ndarray:
    """Fuse paired single-seed route scores without ranking the result."""
    return fuse_scores(core_scores, caisr_scores)


def _payload_digest(indices: np.ndarray, scores: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (np.asarray(indices), np.asarray(scores)):
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def publish_score_unit(target: Path, indices: np.ndarray, scores: np.ndarray,
                       metadata: Mapping[str, object]) -> str:
    """Atomically publish one private score unit without clobbering a winner."""
    target = Path(target)
    idx = np.asarray(indices)
    values = _vector(scores, "scores")
    if idx.ndim != 1 or idx.dtype.kind not in "iu" or idx.shape != values.shape:
        raise FusionStop("score unit indices and values are not aligned")
    identity = dict(metadata)
    identity["payload_sha256"] = _payload_digest(idx, values)
    if target.exists():
        required = (target / "payload.npz", target / "metadata.json", target / "complete.json")
        if not target.is_dir() or not all(path.is_file() for path in required):
            raise FusionStop(f"existing private score unit is incomplete: {target}")
        saved = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        if saved != identity:
            raise FusionStop(f"existing private score unit identity mismatch: {target}")
        with np.load(target / "payload.npz", allow_pickle=False) as archive:
            saved_idx = np.asarray(archive["test_global_indices"])
            saved_values = np.asarray(archive["scores"])
        if not np.array_equal(saved_idx, idx) or not np.array_equal(saved_values, values):
            raise FusionStop(f"existing private score unit payload mismatch: {target}")
        complete = json.loads((target / "complete.json").read_text(encoding="utf-8"))
        if complete.get("status") != "COMPLETE" or complete.get("payload_sha256") != identity["payload_sha256"]:
            raise FusionStop(f"existing private score unit completion mismatch: {target}")
        return "SKIP_VERIFIED"
    target.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent))
    try:
        np.savez(building / "payload.npz", test_global_indices=idx, scores=values)
        (building / "metadata.json").write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
        (building / "complete.json").write_text(json.dumps({"status": "COMPLETE", **identity}, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.rename(building, target)
        except FileExistsError as exc:
            raise FusionStop(f"private score unit lost publish race: {target}") from exc
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise
    return "BUILD_NEW"


def bootstrap_route_series(pair_payloads: Mapping[str, Mapping[str, np.ndarray]],
                           multiplicities: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate all routes against one shared record-multiplicity matrix."""
    matrix = np.asarray(multiplicities, dtype=np.int64)
    if matrix.ndim != 2 or np.any(matrix < 0):
        raise FusionStop("bootstrap multiplicities must be a nonnegative matrix")
    required = {"CORE", "CAISR20", "FUSION_50_50"}
    if set(pair_payloads) != required:
        raise FusionStop("bootstrap route set must be exactly CORE, CAISR20, FUSION_50_50")
    from .e0e3_stability import weighted_ac_batch
    return {route: weighted_ac_batch(pair_payloads[route], matrix) for route in ("CORE", "CAISR20", "FUSION_50_50")}


def route_scope_value(site_values: Mapping[str, float], scope: str) -> float:
    if set(site_values) != set(SITES):
        raise FusionStop("site scope values must cover exactly the three holdouts")
    ordered = np.asarray([float(site_values[site]) for site in SITES], dtype=np.float64)
    if not np.isfinite(ordered).all():
        raise FusionStop("site scope values must be finite")
    if scope in SITES:
        return float(site_values[scope])
    if scope == "Macro":
        return float(ordered.mean())
    if scope == "Worst-site":
        return float(ordered.min())
    if scope == "Small-site mean":
        return float(ordered[:2].mean())
    raise FusionStop(f"unknown scope: {scope}")


def build_scope_rows(route_values: Mapping[str, Mapping[str, float]]) -> list[dict[str, object]]:
    """Build aggregate route and only the two pre-registered fusion comparisons."""
    expected = {"CAISR20", "CORE", "FUSION_50_50"}
    if set(route_values) != expected:
        raise FusionStop("scope rows require exactly the two parents and fixed fusion")
    scopes = (*SITES, "Macro", "Worst-site", "Small-site mean")
    rows: list[dict[str, object]] = []
    for route in ("CAISR20", "CORE", "FUSION_50_50"):
        for scope in scopes:
            rows.append({"row_type": "route_summary", "route": route, "scope": scope, "ac": route_scope_value(route_values[route], scope)})
    for comparison, parent in (("FUSION_MINUS_CORE", "CORE"), ("FUSION_MINUS_CAISR", "CAISR20")):
        for scope in scopes:
            rows.append({"row_type": "comparison", "comparison": comparison, "left_route": "FUSION_50_50", "right_route": parent, "scope": scope, "delta": route_scope_delta(route_values["FUSION_50_50"], route_values[parent], scope)})
    return rows


def route_scope_delta(left: Mapping[str, float], right: Mapping[str, float], scope: str) -> float:
    """Subtract route-level scope aggregates; never aggregate site deltas."""
    return float(route_scope_value(left, scope) - route_scope_value(right, scope))


def route_scope_delta_batch(left: np.ndarray, right: np.ndarray, scope: str) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.shape != a.shape or a.shape[1] != len(SITES) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FusionStop(f"route batch values must be finite [n, {len(SITES)}] arrays")
    if scope in SITES:
        return a[:, SITES.index(scope)] - b[:, SITES.index(scope)]
    if scope == "Macro":
        return a.mean(axis=1) - b.mean(axis=1)
    if scope == "Worst-site":
        return a.min(axis=1) - b.min(axis=1)
    if scope == "Small-site mean":
        return a[:, :2].mean(axis=1) - b[:, :2].mean(axis=1)
    raise FusionStop(f"unknown scope: {scope}")


SCOPES = (*SITES, "Macro", "Worst-site", "Small-site mean")
COMPARISONS = ("FUSION_MINUS_CORE", "FUSION_MINUS_CAISR")
BOOTSTRAP_SEED = 20261023


def build_seed_rows(seed_deltas: Mapping[str, Mapping[str, np.ndarray]], *, seeds: tuple[int, ...] = SEEDS) -> list[dict[str, object]]:
    """Summarize fixed seed deltas without exposing record-level values."""
    rows: list[dict[str, object]] = []
    for comparison in COMPARISONS:
        if comparison not in seed_deltas:
            continue
        for scope in SCOPES:
            values = np.asarray(seed_deltas[comparison][scope], dtype=np.float64)
            if values.ndim != 1 or len(values) != len(seeds) or not np.isfinite(values).all():
                raise FusionStop("seed delta coverage or finiteness mismatch")
            rows.append({
                "comparison": comparison, "scope": scope, "n_seeds": int(len(values)),
                "n_positive": int(np.sum(values > 0)), "n_zero": int(np.sum(values == 0)),
                "n_negative": int(np.sum(values < 0)), "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)), "q25": float(np.quantile(values, .25)),
                "q75": float(np.quantile(values, .75)), "min": float(values.min()),
                "max": float(values.max()),
            })
    return rows


def build_bootstrap_rows(series: Mapping[str, Mapping[str, np.ndarray]], observed: Mapping[str, Mapping[str, float]], *, bootstrap_seed: int = BOOTSTRAP_SEED) -> list[dict[str, object]]:
    """Summarize shared-multiplicity bootstrap deltas as aggregate public rows."""
    rows: list[dict[str, object]] = []
    for comparison in COMPARISONS:
        if comparison not in series or comparison not in observed:
            continue
        for scope in SCOPES:
            values = np.asarray(series[comparison][scope], dtype=np.float64)
            finite = np.isfinite(values)
            valid = values[finite]
            if valid.size == 0:
                raise FusionStop("bootstrap delta has no valid replicates")
            if not np.isfinite(float(observed[comparison][scope])):
                raise FusionStop("bootstrap observed delta is nonfinite")
            sd = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
            rows.append({
                "comparison": comparison, "scope": scope, "observed": float(observed[comparison][scope]),
                "mean": float(valid.mean()), "median": float(np.median(valid)), "sd": sd,
                "q025": float(np.quantile(valid, .025)), "q975": float(np.quantile(valid, .975)),
                "fraction_gt_0": float(np.mean(valid > 0)), "fraction_lt_0": float(np.mean(valid < 0)),
                "valid": int(valid.size), "invalid": int((~finite).sum()),
                "n_bootstrap": int(values.size), "bootstrap_seed": int(bootstrap_seed),
            })
    return rows


def _load_caisr_site(route_root: Path, holdout: str) -> dict[str, object]:
    from .caisr20_core_protocol import mean_rank_fusion
    columns = []
    canonical = None
    for seed in SEEDS:
        path = Path(route_root) / "predictions" / f"seed{seed}" / f"holdout_{holdout}" / "payload.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                scores = np.asarray(archive["scores"], dtype=np.float64)
                indices = np.asarray(archive["test_global_indices"], dtype=np.int64)
        except (OSError, KeyError, ValueError) as exc:
            raise FusionStop(f"invalid CAISR payload: {path}") from exc
        if scores.ndim != 1 or indices.ndim != 1 or scores.shape != indices.shape or not np.isfinite(scores).all():
            raise FusionStop(f"invalid CAISR payload shape or score: {path}")
        if canonical is None:
            canonical = indices.copy()
        elif not np.array_equal(canonical, indices):
            raise FusionStop(f"CAISR row identity differs across seeds: {holdout}")
        columns.append(scores)
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    return {"indices": canonical, "final": mean_rank_fusion(matrix).astype(np.float64), "matrix": matrix}


def _load_core_site(old_rows: Mapping[tuple[str, str, str, str, int], Mapping[str, object]], core_root: Path, holdout: str) -> dict[str, object]:
    from .v5_family_loo_analysis import _load_columns
    indices, final, matrix, keys = _load_columns(old_rows, Path(core_root), "CORE_COH_EMG", holdout, "inductive")
    if len(keys) != 225:
        raise FusionStop(f"CORE final column count mismatch: {len(keys)}")
    return {"indices": np.asarray(indices, dtype=np.int64), "final": np.asarray(final, dtype=np.float64), "matrix": np.asarray(matrix, dtype=np.float64), "keys": keys}


def _load_metadata_index(manifest_path: Path):
    import pandas as pd
    from reproduction.siren_a.pipeline.label_join import MetadataIndex, MetadataRecord
    manifest = pd.read_csv(Path(manifest_path), dtype={"record_id": str, "patient_id": str, "site": str})
    required = {"record_id", "patient_id", "site", "label", "age"}
    if not required.issubset(manifest.columns):
        raise FusionStop("metadata manifest lacks scoring fields")
    records = tuple(MetadataRecord(str(row.site), str(row.record_id), str(row.record_id), str(int(row.label)), str(float(row.age)), str(row.record_id), int(i)) for i, row in manifest.iterrows())
    metadata = MetadataIndex(records=records, cache_names=tuple(f"cache_{i:06d}.npz" for i in range(len(records))), relative_paths=tuple(f"{row.site}/{row.record_id}.npz" for _, row in manifest.iterrows()), sites=manifest.site.to_numpy(str), demographics_sha256="private", join_fingerprint="private", audit={})
    return metadata


def _read_parent_site_values(core_site_csv: Path, caisr_site_csv: Path) -> dict[str, dict[str, float]]:
    import csv
    expected = {"CORE": {}, "CAISR20": {}}
    with Path(core_site_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("config_id")) == "CORE_COH_EMG" and str(row.get("protocol")) == "inductive":
                expected["CORE"][str(row["holdout_site"])] = float(row["ac_auroc"])
    with Path(caisr_site_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("route")) == "CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL":
                expected["CAISR20"][str(row["holdout_site"])] = float(row["ac_auroc"])
    for route in expected:
        if set(expected[route]) != set(SITES):
            raise FusionStop(f"parent public site coverage mismatch: {route}")
    return expected


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv
    if not rows:
        raise FusionStop(f"refusing to write empty public artifact: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def run_fusion_analysis(*, core_root: Path, core_reference_manifest: Path, caisr_root: Path,
                        metadata_manifest: Path, public_root: Path, private_root: Path,
                        core_site_csv: Path, caisr_site_csv: Path,
                        n_bootstrap: int = 2000, bootstrap_seed: int = BOOTSTRAP_SEED) -> dict[str, object]:
    """Replay frozen parents, compute fixed 50/50 fusion, and emit aggregates."""
    from reproduction.siren_a.g5.core import load_eval_targets, score_site
    from .v5_family_loo_analysis import OFFICIAL, _rows_by_key
    from .e0e3_stability import bootstrap_multiplicity, pair_contributions
    from .caisr20_core_protocol_runner import _rank_vector
    if int(n_bootstrap) <= 0 or int(bootstrap_seed) != BOOTSTRAP_SEED:
        raise FusionStop("bootstrap contract mismatch")
    old_rows = _rows_by_key(Path(core_reference_manifest), "E0")
    metadata = _load_metadata_index(metadata_manifest)
    parent_expected = _read_parent_site_values(core_site_csv, caisr_site_csv)
    core_units = {}; caisr_units = {}; fusion_units = {}
    site_rows: list[dict[str, object]] = []
    replay_diffs = []
    for holdout in SITES:
        core = _load_core_site(old_rows, Path(core_root), holdout)
        caisr = _load_caisr_site(Path(caisr_root), holdout)
        validate_parent_pair(core["indices"], caisr["indices"], core["final"], caisr["final"], holdout=holdout)
        labels, ages = load_eval_targets(metadata, core["indices"])
        core_metrics = score_site(labels, core["final"], ages, OFFICIAL)
        caisr_metrics = score_site(labels, caisr["final"], ages, OFFICIAL)
        fusion = fuse_scores(core["final"], caisr["final"])
        fusion_metrics = score_site(labels, fusion, ages, OFFICIAL)
        for route, metrics in (("CORE", core_metrics), ("CAISR20", caisr_metrics), ("FUSION_50_50", fusion_metrics)):
            site_rows.append({"route": route, "holdout_site": holdout, **metrics})
        replay_diffs.extend([abs(float(core_metrics["ac_auroc"]) - parent_expected["CORE"][holdout]), abs(float(caisr_metrics["ac_auroc"]) - parent_expected["CAISR20"][holdout])])
        core_units[holdout] = core; caisr_units[holdout] = caisr; fusion_units[holdout] = {"indices": core["indices"], "final": fusion}
    if max(replay_diffs, default=float("inf")) > 1e-12:
        raise FusionStop(f"parent replay mismatch: {max(replay_diffs)}")
    private_root = Path(private_root); private_root.mkdir(parents=True, exist_ok=True)
    publish_status = []
    for holdout in SITES:
        payload = fusion_units[holdout]
        target = private_root / "final" / f"holdout_{holdout}"
        publish_status.append(publish_score_unit(target, payload["indices"], payload["final"], {"schema": "featurelab_core_caisr_fusion_50_50_score_v1", "route": "FUSION_50_50", "holdout": holdout, "core_weight": CORE_WEIGHT, "caisr_weight": CAISR_WEIGHT, "parent_core_payload_sha256": _payload_digest(core_units[holdout]["indices"], core_units[holdout]["final"]), "parent_caisr_payload_sha256": _payload_digest(caisr_units[holdout]["indices"], caisr_units[holdout]["final"])}))
    route_values = {route: {site: float(next(row["ac_auroc"] for row in site_rows if row["route"] == route and row["holdout_site"] == site)) for site in SITES} for route in ("CAISR20", "CORE", "FUSION_50_50")}
    summary_rows = build_scope_rows(route_values)
    seed_deltas = {comparison: {scope: [] for scope in SCOPES} for comparison in COMPARISONS}
    for seed in SEEDS:
        seed_values = {route: {} for route in ("CORE", "CAISR20", "FUSION_50_50")}
        for holdout in SITES:
            core_seed_idx, core_seed, _, core_keys = _load_core_site_seed(old_rows, Path(core_root), holdout, seed)
            caisr_seed = _rank_vector(caisr_units[holdout]["matrix"][:, SEEDS.index(seed)])
            validate_parent_pair(core_seed_idx, caisr_units[holdout]["indices"], core_seed, caisr_seed, holdout=holdout, seed=seed, caisr_seed=seed)
            labels, ages = load_eval_targets(metadata, core_seed_idx)
            seed_values["CORE"][holdout] = float(score_site(labels, core_seed, ages, OFFICIAL)["ac_auroc"])
            seed_values["CAISR20"][holdout] = float(score_site(labels, caisr_seed, ages, OFFICIAL)["ac_auroc"])
            seed_values["FUSION_50_50"][holdout] = float(score_site(labels, build_seed_fusion(core_seed, caisr_seed), ages, OFFICIAL)["ac_auroc"])
        for comparison, parent in (("FUSION_MINUS_CORE", "CORE"), ("FUSION_MINUS_CAISR", "CAISR20")):
            for scope in SCOPES:
                seed_deltas[comparison][scope].append(route_scope_delta(seed_values["FUSION_50_50"], seed_values[parent], scope))
    seed_rows = build_seed_rows({k: {s: np.asarray(v) for s, v in scopes.items()} for k, scopes in seed_deltas.items()})
    bootstrap_series = {comparison: {scope: {} for scope in SCOPES} for comparison in COMPARISONS}
    bootstrap_observed = {comparison: {} for comparison in COMPARISONS}
    rng = np.random.default_rng(int(bootstrap_seed)); private_bootstrap = {}
    for holdout in SITES:
        idx = core_units[holdout]["indices"]; labels, ages = load_eval_targets(metadata, idx)
        mult = np.vstack([bootstrap_multiplicity(len(labels), rng) for _ in range(int(n_bootstrap))])
        pairs = {route: pair_contributions(labels, ages, units[holdout]["final"]) for route, units in (("CORE", core_units), ("CAISR20", caisr_units), ("FUSION_50_50", fusion_units))}
        series = bootstrap_route_series(pairs, mult); matrix = {route: series[route] for route in series}
        private_bootstrap[holdout] = {"multiplicities": mult, **matrix}
        for comparison, parent in (("FUSION_MINUS_CORE", "CORE"), ("FUSION_MINUS_CAISR", "CAISR20")):
            for scope in SITES:
                if scope == holdout:
                    bootstrap_series[comparison][scope] = matrix["FUSION_50_50"] - matrix[parent]
                    bootstrap_observed[comparison][scope] = route_scope_delta(route_values["FUSION_50_50"], route_values[parent], scope)
    for comparison, parent in (("FUSION_MINUS_CORE", "CORE"), ("FUSION_MINUS_CAISR", "CAISR20")):
        route_matrix_left = np.column_stack([private_bootstrap[site]["FUSION_50_50"] for site in SITES])
        route_matrix_right = np.column_stack([private_bootstrap[site][parent] for site in SITES])
        for scope in ("Macro", "Worst-site", "Small-site mean"):
            bootstrap_series[comparison][scope] = route_scope_delta_batch(route_matrix_left, route_matrix_right, scope)
            bootstrap_observed[comparison][scope] = route_scope_delta(route_values["FUSION_50_50"], route_values[parent], scope)
    bootstrap_rows = build_bootstrap_rows(bootstrap_series, bootstrap_observed, bootstrap_seed=bootstrap_seed)
    for holdout, payload in private_bootstrap.items():
        np.savez_compressed(private_root / f"bootstrap_{holdout}.npz", **payload)
    public_root = Path(public_root); public_root.mkdir(parents=True, exist_ok=True)
    _write_csv(public_root / "core_caisr_fusion_50_50_site_scores.csv", site_rows)
    _write_csv(public_root / "core_caisr_fusion_50_50_summary.csv", summary_rows)
    _write_csv(public_root / "core_caisr_fusion_50_50_seed_stability.csv", seed_rows)
    _write_csv(public_root / "core_caisr_fusion_50_50_bootstrap.csv", bootstrap_rows)
    manifest = {"schema": "featurelab_core_caisr_fusion_50_50_v1", "route": "CORE_CAISR_FUSION_50_50", "core_weight": CORE_WEIGHT, "caisr_weight": CAISR_WEIGHT, "post_fusion_rerank": False, "holdout_sites": list(SITES), "seeds": list(SEEDS), "parent_final_identity_count": 3, "parent_seed_identity_count": 45, "parent_replay_max_abs_diff": float(max(replay_diffs, default=0.0)), "site_score_rows": len(site_rows), "summary_rows": len(summary_rows), "seed_rows": len(seed_rows), "bootstrap_rows": len(bootstrap_rows), "bootstrap_replicates": int(n_bootstrap), "bootstrap_seed": int(bootstrap_seed), "private_publish_status": publish_status, "model_refit_count": 0, "prediction_recompute_count": 0, "feature_extraction_count": 0, "route_search": "CLOSED_FIXED_50_50", "no_candidate_selection": True}
    (public_root / "core_caisr_fusion_50_50_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"site_rows": site_rows, "summary_rows": summary_rows, "seed_rows": seed_rows, "bootstrap_rows": bootstrap_rows, "manifest": manifest}


def _load_core_site_seed(old_rows, core_root: Path, holdout: str, seed: int):
    from .v5_family_loo_analysis import _load_columns
    indices, fused, matrix, keys = _load_columns(old_rows, Path(core_root), "CORE_COH_EMG", holdout, "inductive", seed=seed)
    if len(keys) != 15:
        raise FusionStop("CORE seed column count mismatch")
    return np.asarray(indices, dtype=np.int64), np.asarray(fused, dtype=np.float64), matrix, keys


__all__ = [
    "CAISR_WEIGHT", "CORE_WEIGHT", "FusionStop", "SEEDS", "SITES",
    "bootstrap_route_series", "build_bootstrap_rows", "build_scope_rows", "build_seed_fusion", "build_seed_rows", "fuse_scores",
    "publish_score_unit", "route_scope_delta", "route_scope_delta_batch",
    "route_scope_value", "validate_parent_pair", "run_fusion_analysis",
]
