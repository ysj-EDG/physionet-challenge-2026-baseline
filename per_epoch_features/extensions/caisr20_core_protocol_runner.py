"""Bounded CAISR20-vs-CORE route runner.

The runner is intentionally independent from the frozen CORE implementation.
It reads the existing DATA2 summary cache and frozen sample artifacts, writes
only private route units, and emits aggregate public artifacts after all model
and prediction units have been validated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .caisr20_core_protocol import (
    CAISR20_FEATURE_IDS, CAISR20_X_STATIC_INDICES, EXPECTED_SITE_COUNTS,
    EXPECTED_SITE_POSITIVES, HOLDOUT_TO_TRAIN, ROUTE_ID, SEEDS, SITES,
    apply_historical_typed_caisr20, apply_selected_preprocessor,
    fit_historical_typed_caisr20, fit_selected_preprocessor, manifest_payload,
    mean_rank_fusion, model_config,
)


class RouteStop(RuntimeError):
    """A safety gate failed; no downstream route work may continue."""


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _tree_digest(path: Path) -> dict[str, dict[str, object]]:
    return {
        str(item.relative_to(path)): {"size": item.stat().st_size, "sha256": sha256_file(item)}
        for item in sorted(Path(path).rglob("*")) if item.is_file() and item.name != "complete.json"
    }


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b))


def _validate_unit(path: Path, payload: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> None:
    path = Path(path)
    required = (path / "payload.npz", path / "metadata.json", path / "complete.json")
    if not (path.is_dir() and all(item.is_file() for item in required)):
        raise RouteStop(f"incomplete route unit: {path}")
    saved = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if saved != json.loads(_canonical(metadata).decode("utf-8")):
        raise RouteStop(f"route unit identity mismatch: {path}")
    complete = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if complete.get("status") != "COMPLETE" or complete.get("payload_digest") != _tree_digest(path):
        raise RouteStop(f"route unit completion digest mismatch: {path}")
    with np.load(path / "payload.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(payload):
            raise RouteStop(f"route unit payload key mismatch: {path}")
        for key, value in payload.items():
            if not _same_array(np.asarray(archive[key]), np.asarray(value)):
                raise RouteStop(f"route unit payload mismatch: {path}/{key}")


def publish_unit(path: Path, payload: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> str:
    """Publish a private model/prediction unit atomically without clobbering."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        _validate_unit(path, payload, metadata)
        return "SKIP_VERIFIED"
    path.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{path.name}.building-", dir=str(path.parent)))
    try:
        with (building / "payload.npz").open("wb") as handle:
            np.savez_compressed(handle, **{key: np.asarray(value) for key, value in payload.items()})
            handle.flush(); os.fsync(handle.fileno())
        (building / "metadata.json").write_bytes(_canonical(metadata) + b"\n")
        (building / "complete.json").write_bytes(_canonical({
            "schema": "featurelab_caisr20_core_protocol_unit_v1",
            "status": "COMPLETE",
            "payload_digest": _tree_digest(building),
        }) + b"\n")
        _validate_unit(building, payload, metadata)
        if path.exists() or path.is_symlink():
            raise RouteStop(f"route unit target appeared during publish: {path}")
        os.rename(building, path)
        _validate_unit(path, payload, metadata)
        return "BUILD_NEW"
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise


def load_data2_inputs(manifest_path: Path, summary_cache: Path) -> dict[str, object]:
    manifest = pd.read_csv(Path(manifest_path), dtype={"record_id": str, "patient_id": str, "site": str})
    required = {"record_id", "patient_id", "site", "label", "age"}
    if not required.issubset(manifest.columns) or len(manifest) != 6600:
        raise RouteStop("DATA2 manifest schema or row count mismatch")
    if manifest.record_id.nunique() != 6600 or manifest.patient_id.nunique() != 6600:
        raise RouteStop("DATA2 record/patient identity is not unique")
    if int(manifest.label.sum()) != 498:
        raise RouteStop("DATA2 label inventory mismatch")
    counts = manifest.site.value_counts().to_dict()
    positives = manifest.groupby("site").label.sum().astype(int).to_dict()
    if {site: int(counts.get(site, 0)) for site in SITES} != EXPECTED_SITE_COUNTS:
        raise RouteStop(f"DATA2 site inventory mismatch: {counts}")
    if {site: int(positives.get(site, 0)) for site in SITES} != EXPECTED_SITE_POSITIVES:
        raise RouteStop(f"DATA2 positive inventory mismatch: {positives}")
    with np.load(Path(summary_cache), allow_pickle=False) as archive:
        if not {"record_id", "site_static196"}.issubset(archive.files):
            raise RouteStop("DATA2 summary cache lacks identity/static source")
        ids = np.asarray(archive["record_id"]).astype(str)
        sites = np.asarray(archive["site"]).astype(str)
        static = np.asarray(archive["site_static196"], dtype=np.float64)
    if not np.array_equal(ids, manifest.record_id.to_numpy(str)) or not np.array_equal(sites, manifest.site.to_numpy(str)):
        raise RouteStop("DATA2 cache/manifest identity order mismatch")
    if static.shape != (6600, 196) or not np.isfinite(static[:, 0]).all():
        raise RouteStop("DATA2 static source shape/age contract mismatch")
    raw = static[:, CAISR20_X_STATIC_INDICES]
    return {"manifest": manifest, "raw_caisr": raw, "static196": static, "record_id": ids, "site": sites}


def load_typed_rules(rules_path: Path) -> list[dict]:
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    by_index = {int(row["index"]): row for row in rules if row.get("branch") == "x_static"}
    selected = [by_index[index] for index in CAISR20_X_STATIC_INDICES]
    if tuple(row["name"] for row in selected) != CAISR20_FEATURE_IDS:
        raise RouteStop("typed CAISR20 rule order mismatch")
    return selected


def source_value_bridge(*, pool_root: Path, raw_caisr: np.ndarray, manifest: pd.DataFrame) -> dict[str, object]:
    """Compare raw CAISR source values against the immutable old-V5 pool.

    The pool is never rewritten.  A value mismatch is a hard drift; a validity
    mismatch is reported separately because it changes missingness semantics.
    """
    pool_root = Path(pool_root)
    ids = np.load(pool_root / "unique_feature_ids.npy", allow_pickle=False).astype(str)
    old_indices = []
    for name in CAISR20_FEATURE_IDS:
        matches = np.flatnonzero(ids == name)
        if len(matches) != 1:
            raise RouteStop(f"old-V5 CAISR feature missing/duplicated: {name}")
        old_indices.append(int(matches[0]))
    old_values = np.load(pool_root / "X_raw.npy", mmap_mode="r")[:, old_indices]
    old_valid = np.load(pool_root / "valid_mask.npy", mmap_mode="r")[:, old_indices]
    old_sites = np.load(pool_root / "row_sites.npy", allow_pickle=False).astype(str)
    if old_values.shape != raw_caisr.shape or not np.array_equal(old_sites, manifest.site.to_numpy(str)):
        raise RouteStop("old-V5 pool row/source identity mismatch")
    finite_overlap = np.isfinite(old_values) & np.isfinite(raw_caisr)
    value_mismatch = finite_overlap & (np.abs(old_values - raw_caisr) > 1e-12)
    validity_mismatch = old_valid != np.isfinite(raw_caisr)
    by_feature = []
    for j, name in enumerate(CAISR20_FEATURE_IDS):
        by_feature.append({
            "feature_id": name,
            "value_mismatch": int(value_mismatch[:, j].sum()),
            "validity_mismatch": int(validity_mismatch[:, j].sum()),
            "old_valid": int(old_valid[:, j].sum()),
            "new_finite": int(np.isfinite(raw_caisr[:, j]).sum()),
        })
    value_ok = not bool(value_mismatch.any())
    validity_ok = not bool(validity_mismatch.any())
    return {
        "status": "PASS" if value_ok else "DRIFT",
        "finite_overlap_status": "PASS" if value_ok else "DRIFT",
        "validity_policy_status": "EXACT" if validity_ok else "DIAGNOSTIC",
        "value_mismatch_total": int(value_mismatch.sum()),
        "validity_mismatch_total": int(validity_mismatch.sum()),
        "validity_policy_mismatch_blocking": False,
        "features": by_feature,
    }



def historical_bridge(*, root: Path, summary_cache: Path, manifest_path: Path,
                      archive_root: Path) -> dict[str, object]:
    """Reproduce the archived seed-7 pooled CAISR20 route without writing it."""
    import sys
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning

    root = Path(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from feat_input.feat_mody import run_p3_v2 as p3

    manifest = pd.read_csv(Path(manifest_path), dtype={"record_id": str, "site": str})
    with np.load(Path(summary_cache), allow_pickle=False) as archive:
        sites = manifest.site.to_numpy(str)
        matrices = {site: np.asarray(archive[f"p3_{site}_compact30"][:, 10:30], dtype=np.float64) for site in SITES}
    expected = {"I0002": 0.7460233297985154, "I0006": 0.6777064789898571, "S0001": 0.6166620961991992}
    max_diffs = {}; warnings_found = 0
    for holdout in SITES:
        train = sites != holdout; test = ~train
        median, center, scale, _ = p3.fit_imputer_scaler(matrices[holdout][train])
        x_train = p3.apply_imputer_scaler(matrices[holdout][train], median, center, scale)
        x_test = p3.apply_imputer_scaler(matrices[holdout][test], median, center, scale)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(**p3.LR).fit(x_train, manifest.loc[train, "label"].to_numpy(np.int8))
        warnings_found += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        scores = model.decision_function(x_test)
        archived = pd.read_csv(Path(archive_root) / "predictions" / f"holdout_{holdout}.csv", dtype={"record_id": str})
        ids = manifest.loc[test, "record_id"].to_numpy(str)
        if not np.array_equal(ids, archived.record_id.to_numpy(str)):
            raise RouteStop(f"historical bridge record order mismatch: {holdout}")
        max_diffs[holdout] = float(np.max(np.abs(scores - archived.decision_score.to_numpy(np.float64))))
    return {"status": "PASS" if max(max_diffs.values()) <= 1e-10 and warnings_found == 0 else "FAIL",
            "max_abs_diff_by_site": max_diffs, "max_abs_diff": max(max_diffs.values()),
            "warnings": int(warnings_found), "expected_ac": expected,
            "expected_macro_ac": 0.6801306349958572}


def _state_from_historical_parameters(parameters: Mapping[str, Mapping[str, object]]) -> object:
    center = np.empty(20, dtype=np.float64)
    scale = np.empty(20, dtype=np.float64)
    methods = []
    counts = np.empty(20, dtype=np.int64)
    all_missing = np.zeros(20, dtype=bool)
    for j, index in enumerate(CAISR20_X_STATIC_INDICES):
        entry = parameters.get(f"x_static:{index}")
        if not isinstance(entry, Mapping):
            raise RouteStop(f"historical typed parameter missing x_static:{index}")
        center[j] = float(entry["center"])
        scale[j] = float(entry["scale"])
        methods.append(str(entry.get("scale_method", "unknown")))
        counts[j] = int(entry.get("fit_valid_observations", 0))
        all_missing[j] = methods[-1] == "all_missing_fallback"
    from .caisr20_core_protocol import HistoricalTypedState
    return HistoricalTypedState(center, scale, tuple(methods), counts, all_missing)


def historical_typed_pipeline_bridge(*, root: Path, summary_cache: Path,
                                      manifest_path: Path, rules_path: Path) -> dict[str, object]:
    """Rebuild the archived typed_v1 CAISR20 matrices for all three folds."""
    import joblib
    root = Path(root)
    manifest = pd.read_csv(Path(manifest_path), dtype={"record_id": str, "site": str})
    with np.load(Path(summary_cache), allow_pickle=False) as archive:
        static = np.asarray(archive["site_static196"], dtype=np.float64)
        sites = np.asarray(archive["site"], dtype=str)
        targets = {site: np.asarray(archive[f"p3_{site}_compact30"], dtype=np.float64)[:, 10:30] for site in SITES}
    if static.shape != (6600, 196) or not np.array_equal(sites, manifest.site.to_numpy(str)):
        raise RouteStop("historical typed bridge source identity mismatch")
    rules = load_typed_rules(rules_path)
    model_root = root / "feat_input2" / "output" / "data2_lowcost" / "loso_lr" / "compact30"
    per_site = {}
    for holdout in SITES:
        model_path = model_root / f"holdout_{holdout}" / "model.joblib"
        if not model_path.is_file():
            raise RouteStop(f"historical typed bridge checkpoint unavailable: {model_path}")
        archived = joblib.load(model_path)
        state = _state_from_historical_parameters(archived.get("selected_input_parameters", {}))
        rebuilt = apply_historical_typed_caisr20(static, rules, state)
        target = targets[holdout]
        if rebuilt.shape != (6600, 20) or target.shape != (6600, 20) or not np.isfinite(rebuilt).all() or not np.isfinite(target).all():
            raise RouteStop(f"historical typed bridge shape/finite failure: {holdout}")
        diff = np.abs(rebuilt - target)
        per_site[holdout] = {"rows": int(rebuilt.shape[0]), "features": int(rebuilt.shape[1]), "max_abs_diff": float(diff.max())}
    max_diff = max(item["max_abs_diff"] for item in per_site.values())
    return {"status": "PASS" if max_diff <= 1e-12 else "FAIL", "groups": int(sum(item["max_abs_diff"] <= 1e-12 for item in per_site.values())), "rows": 6600, "features": 20, "max_abs_diff": max_diff, "per_site": per_site}

def _sample_path(seed: int, train_site: str, sample_root: Path) -> Path:
    if int(seed) == 42:
        path = Path(sample_root).parent / "python-example-2026-siren-a-g4a-private" / "seed42_smoke_v1" / "samples" / f"{train_site}_seed42" / "selected_indices.npz"
    else:
        path = Path(sample_root) / "seeds43_56_v1" / "seeds" / f"seed{int(seed)}" / "samples" / train_site / "selected_indices.npz"
    if not path.is_file():
        raise RouteStop(f"frozen sample artifact unavailable: {path}")
    return path


def load_frozen_sample(seed: int, train_site: str, sample_root: Path, sites: np.ndarray) -> dict[str, object]:
    path = _sample_path(seed, train_site, sample_root)
    with np.load(path, allow_pickle=False) as archive:
        required = {"local_positions", "global_indices"}
        if set(archive.files) != required:
            raise RouteStop(f"sample artifact fields mismatch: {path}")
        local = np.asarray(archive["local_positions"], dtype=np.int64)
        global_indices = np.asarray(archive["global_indices"], dtype=np.int64)
    expected_global = np.flatnonzero(np.asarray(sites).astype(str) == str(train_site)).astype(np.int64)
    if len(local) != len(global_indices) or len(np.unique(local)) != len(local) or len(np.unique(global_indices)) != len(global_indices):
        raise RouteStop("sample identity is duplicated")
    if np.any(local < 0) or np.any(local >= len(expected_global)) or not np.array_equal(global_indices, expected_global[local]):
        raise RouteStop(f"sample local/global identity mismatch: {path}")
    return {"local_positions": local, "global_indices": global_indices,
            "sample_sha256": array_digest(global_indices), "path": str(path)}


def _model_from_payload(metadata: Mapping[str, object], payload: Mapping[str, np.ndarray]):
    model = LogisticRegression(**model_config(int(metadata["seed"])))
    model.classes_ = np.asarray(payload["classes"], dtype=np.int8)
    model.coef_ = np.asarray(payload["coef"], dtype=np.float64)
    model.intercept_ = np.asarray(payload["intercept"], dtype=np.float64)
    model.n_features_in_ = int(model.coef_.shape[1])
    model.n_iter_ = np.asarray(payload["n_iter"], dtype=np.int32)
    return model


@dataclass(frozen=True)
class _PreprocessingBundle:
    typed: object
    p3: object


def _fit_or_load_model(*, root: Path, train_site: str, seed: int,
                       static196: np.ndarray, rules: Sequence[Mapping[str, object]],
                       labels: np.ndarray, sample: Mapping[str, object],
                       manifest_sha: str) -> tuple[str, dict[str, object], object]:
    path = Path(root) / "models" / train_site / f"seed{int(seed)}"
    selected = np.asarray(sample["global_indices"], dtype=np.int64)
    typed_state = fit_historical_typed_caisr20(np.asarray(static196)[selected], rules)
    typed_selected = apply_historical_typed_caisr20(np.asarray(static196)[selected], rules, typed_state)
    p3_state = fit_selected_preprocessor(typed_selected)
    x_fit = apply_selected_preprocessor(typed_selected, p3_state)
    y_selected = np.asarray(labels[selected], dtype=np.int8)
    cfg = model_config(seed)
    typed_meta = {
        "feature_ids": list(CAISR20_FEATURE_IDS),
        "feature_indices": list(CAISR20_X_STATIC_INDICES),
        "center_sha256": array_digest(typed_state.center),
        "scale_sha256": array_digest(typed_state.scale),
        "scale_method": list(typed_state.scale_method),
        "fit_valid_counts": typed_state.fit_valid_counts.astype(int).tolist(),
    }
    p3_meta = {"median_sha256": array_digest(p3_state.median), "center_sha256": array_digest(p3_state.center), "scale_sha256": array_digest(p3_state.scale)}
    meta = {
        "route_id": ROUTE_ID, "train_site": train_site, "seed": int(seed),
        "feature_manifest_sha256": manifest_sha, "training_sample_sha256": str(sample["sample_sha256"]),
        "preprocessing_identity": {"fit_scope": "selected_training_rows_only", "typed": typed_meta, "p3": p3_meta},
        "model_parameters": cfg,
    }
    if path.exists():
        saved_meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        for key in ("route_id", "train_site", "seed", "feature_manifest_sha256", "training_sample_sha256", "preprocessing_identity", "model_parameters"):
            if saved_meta.get(key) != meta.get(key):
                raise RouteStop(f"model unit identity mismatch: {path}")
        with np.load(path / "payload.npz", allow_pickle=False) as archive:
            payload = {key: np.array(archive[key], copy=True) for key in archive.files}
        _validate_unit(path, payload, saved_meta)
        from .caisr20_core_protocol import HistoricalTypedState
        loaded_typed = HistoricalTypedState(payload["typed_center"], payload["typed_scale"], tuple(saved_meta["preprocessing_identity"]["typed"]["scale_method"]), payload["typed_fit_valid_counts"], payload["typed_all_missing"])
        loaded_p3 = type("State", (), {"median": payload["median"], "center": payload["center"], "scale": payload["scale"], "all_missing": payload["all_missing"]})()
        return "SKIP_VERIFIED", saved_meta, (_model_from_payload(saved_meta, payload), _PreprocessingBundle(loaded_typed, loaded_p3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(**cfg).fit(x_fit, y_selected)
    convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    converged = not convergence_warnings and int(np.max(model.n_iter_)) < int(cfg["max_iter"])
    if not converged:
        raise RouteStop(f"CAISR20 model did not converge: {train_site} seed {seed}")
    meta = dict(meta, n_iter=int(model.n_iter_[0]), converged=True, warnings=convergence_warnings)
    payload = {
        "coef": np.asarray(model.coef_, dtype=np.float64), "intercept": np.asarray(model.intercept_, dtype=np.float64),
        "classes": np.asarray(model.classes_, dtype=np.int8), "n_iter": np.asarray(model.n_iter_, dtype=np.int32),
        "typed_center": typed_state.center, "typed_scale": typed_state.scale,
        "typed_fit_valid_counts": typed_state.fit_valid_counts, "typed_all_missing": typed_state.all_missing.astype(bool),
        "median": p3_state.median, "center": p3_state.center, "scale": p3_state.scale, "all_missing": p3_state.all_missing.astype(bool),
    }
    status = publish_unit(path, payload, meta)
    return status, meta, (model, _PreprocessingBundle(typed_state, p3_state))

def _load_model_unit(path: Path, metadata: Mapping[str, object]):
    with np.load(Path(path) / "payload.npz", allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    _validate_unit(Path(path), payload, metadata)
    from .caisr20_core_protocol import HistoricalTypedState
    typed = HistoricalTypedState(payload["typed_center"], payload["typed_scale"], tuple(metadata["preprocessing_identity"]["typed"]["scale_method"]), payload["typed_fit_valid_counts"], payload["typed_all_missing"])
    p3 = type("State", (), {"median": payload["median"], "center": payload["center"], "scale": payload["scale"], "all_missing": payload["all_missing"]})()
    return _model_from_payload(metadata, payload), _PreprocessingBundle(typed, p3)


def _prediction_unit(root: Path, train_site: str, seed: int, holdout: str, model_path: Path, model_meta: Mapping[str, object], scores: np.ndarray, test_indices: np.ndarray) -> str:
    path = Path(root) / "predictions" / f"seed{int(seed)}" / f"holdout_{holdout}"
    metadata = {"route_id": ROUTE_ID, "train_site": train_site, "seed": int(seed), "holdout_site": holdout, "feature_manifest_sha256": model_meta["feature_manifest_sha256"], "training_sample_sha256": model_meta["training_sample_sha256"], "model_unit_digest": _tree_digest(model_path), "test_indices_sha256": array_digest(test_indices), "score_sha256": array_digest(scores)}
    payload = {"scores": np.asarray(scores, dtype=np.float64), "test_global_indices": np.asarray(test_indices, dtype=np.int64)}
    return publish_unit(path, payload, metadata)


def run_route(*, manifest_path: Path, summary_cache: Path, rules_path: Path, pool_root: Path, sample_root: Path, output_root: Path, strict_source_bridge: bool = True) -> dict[str, object]:
    """Run gates, then (only if all gates pass) 30 models and 45 predictions."""
    started = time.time()
    data = load_data2_inputs(manifest_path, summary_cache)
    manifest = data["manifest"]; raw = data["raw_caisr"]; static196 = data["static196"]; sites = data["site"]
    rules = load_typed_rules(rules_path)
    bridge = source_value_bridge(pool_root=pool_root, raw_caisr=raw, manifest=manifest)
    if strict_source_bridge and bridge["status"] != "PASS":
        raise RouteStop(f"CAISR20 source-value bridge failed: {bridge['value_mismatch_total']} value mismatches")
    historical_root = Path(summary_cache).resolve().parents[4]
    historical_typed = historical_typed_pipeline_bridge(
        root=historical_root, summary_cache=summary_cache, manifest_path=manifest_path, rules_path=rules_path
    )
    if historical_typed["status"] != "PASS":
        raise RouteStop(f"historical typed pipeline bridge failed: {historical_typed['max_abs_diff']}")
    manifest_sha = hashlib.sha256(_canonical(manifest_payload())).hexdigest()
    out = Path(output_root); out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_bytes(_canonical(dict(manifest_payload(), manifest_sha256=manifest_sha)) + b"\n")
    samples = {(site, seed): load_frozen_sample(seed, site, sample_root, sites) for site in ("S0001", "I0006") for seed in SEEDS}
    model_statuses = []; prediction_statuses = []; model_cache = {}
    labels = manifest.label.to_numpy(np.int8)
    for train_site in ("S0001", "I0006"):
        for seed in SEEDS:
            status, meta, model_state = _fit_or_load_model(root=out, train_site=train_site, seed=seed, static196=static196, rules=rules, labels=labels, sample=samples[(train_site, seed)], manifest_sha=manifest_sha)
            model_statuses.append(status); model_cache[(train_site, seed)] = (meta, model_state)
    # No evaluation targets are read until all model units have completed.
    for holdout in SITES:
        train_site = HOLDOUT_TO_TRAIN[holdout]
        test_indices = np.flatnonzero(sites == holdout).astype(np.int64)
        for seed in SEEDS:
            meta, (model, bundle) = model_cache[(train_site, seed)]
            typed_test = apply_historical_typed_caisr20(static196[test_indices], rules, bundle.typed)
            scores = model.decision_function(apply_selected_preprocessor(typed_test, bundle.p3)).astype(np.float64)
            status = _prediction_unit(out, train_site, seed, holdout, out / "models" / train_site / f"seed{seed}", meta, scores, test_indices)
            prediction_statuses.append(status)
    fused = {}
    for holdout in SITES:
        columns = []
        canonical = None
        for seed in SEEDS:
            path = out / "predictions" / f"seed{seed}" / f"holdout_{holdout}"
            with np.load(path / "payload.npz", allow_pickle=False) as archive:
                indices = np.asarray(archive["test_global_indices"], dtype=np.int64); scores = np.asarray(archive["scores"], dtype=np.float64)
            if canonical is None: canonical = indices
            elif not np.array_equal(canonical, indices): raise RouteStop("CAISR holdout identity differs across seeds")
            columns.append(scores)
        fused[holdout] = (canonical, mean_rank_fusion(np.column_stack(columns)))
    return {"bridge": bridge, "manifest_sha256": manifest_sha, "models_built": model_statuses.count("BUILD_NEW"), "models_skipped": model_statuses.count("SKIP_VERIFIED"), "predictions_built": prediction_statuses.count("BUILD_NEW"), "predictions_skipped": prediction_statuses.count("SKIP_VERIFIED"), "fused": fused, "elapsed_seconds": time.time() - started}


def _rank_vector(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("rank input must be a finite non-empty vector")
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    less = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts[:-1])))
    return ((less[inverse] + 0.5 * counts[inverse]) / len(x)).astype(np.float64)


def _scope_value(values: Mapping[str, float], scope: str) -> float:
    if scope in SITES:
        return float(values[scope])
    ordered = np.asarray([values[site] for site in SITES], dtype=np.float64)
    if scope == "Macro":
        return float(ordered.mean())
    if scope == "Worst-site":
        return float(ordered.min())
    if scope == "Small-site mean":
        return float((values["I0002"] + values["I0006"]) / 2.0)
    raise ValueError(f"unknown scope: {scope}")


def _route_scope_delta(core_values: Mapping[str, float], caisr_values: Mapping[str, float], scope: str) -> float:
    """Return the route-level scope score difference, not a scope of site deltas."""
    return float(_scope_value(core_values, scope) - _scope_value(caisr_values, scope))


def _route_scope_delta_batch(core_values: np.ndarray, caisr_values: np.ndarray, scope: str) -> np.ndarray:
    """Apply route-level scope aggregation to paired [draw, site] score arrays."""
    core = np.asarray(core_values, dtype=np.float64)
    caisr = np.asarray(caisr_values, dtype=np.float64)
    if core.ndim != 2 or caisr.shape != core.shape or core.shape[1] != len(SITES):
        raise ValueError(f"paired route score arrays must both have shape [n, {len(SITES)}]")
    if scope in SITES:
        return core[:, SITES.index(scope)] - caisr[:, SITES.index(scope)]
    if scope == "Macro":
        return core.mean(axis=1) - caisr.mean(axis=1)
    if scope == "Worst-site":
        return core.min(axis=1) - caisr.min(axis=1)
    if scope == "Small-site mean":
        return core[:, :2].mean(axis=1) - caisr[:, :2].mean(axis=1)
    raise ValueError(f"unknown scope: {scope}")


def finalize_route_comparison(*, route_root: Path, manifest_path: Path,
                              core_root: Path, core_reference_manifest: Path,
                              public_root: Path, private_root: Path,
                              bootstrap_seed: int = 20261023,
                              n_bootstrap: int = 2000,
                              preserve_frozen_outputs: bool = True) -> dict[str, object]:
    """Score completed CAISR20 units against frozen CORE inductive predictions."""
    from reproduction.siren_a.g5.core import load_eval_targets, score_site
    from .v5_family_loo_analysis import OFFICIAL, _load_columns, _rows_by_key
    from .e0e3_stability import bootstrap_multiplicity, pair_contributions, weighted_ac_batch
    from reproduction.siren_a.pipeline.label_join import MetadataIndex, MetadataRecord

    manifest = pd.read_csv(Path(manifest_path), dtype={"record_id": str, "site": str})
    records = tuple(MetadataRecord(str(row.site), str(row.record_id), str(row.record_id), str(int(row.label)), str(float(row.age)), str(row.record_id), int(i)) for i, row in manifest.iterrows())
    metadata = MetadataIndex(records=records, cache_names=tuple(f"cache_{i:06d}.npz" for i in range(len(records))), relative_paths=tuple(f"{row.site}/{row.record_id}.npz" for _, row in manifest.iterrows()), sites=manifest.site.to_numpy(str), demographics_sha256="private", join_fingerprint="private", audit={})
    old_rows = _rows_by_key(Path(core_reference_manifest), "E0")
    core_groups = {}
    caisr_groups = {}
    for holdout in SITES:
        idx, core_fused, core_matrix, core_keys = _load_columns(old_rows, Path(core_root), "CORE_COH_EMG", holdout, "inductive")
        columns = []
        canonical = None
        for seed in SEEDS:
            path = Path(route_root) / "predictions" / f"seed{seed}" / f"holdout_{holdout}"
            with np.load(path / "payload.npz", allow_pickle=False) as archive:
                scores = np.asarray(archive["scores"], dtype=np.float64)
                indices = np.asarray(archive["test_global_indices"], dtype=np.int64)
            if canonical is None:
                canonical = indices
            elif not np.array_equal(canonical, indices):
                raise RouteStop("CAISR final vector identity differs across seeds")
            columns.append(scores)
        caisr_matrix = np.column_stack(columns).astype(np.float64)
        caisr_groups[holdout] = (canonical, mean_rank_fusion(caisr_matrix), caisr_matrix)
        core_groups[holdout] = (idx, core_fused, core_matrix)
    if preserve_frozen_outputs:
        frozen_site_path = Path(public_root) / "caisr20_vs_core_v2_site_scores.csv"
        if not frozen_site_path.is_file():
            raise RouteStop(f"frozen site-score artifact is required in diagnostic mode: {frozen_site_path}")
        with frozen_site_path.open(newline="", encoding="utf-8") as handle:
            site_rows = list(csv.DictReader(handle))
        required_site_keys = {(route, site) for route in ("CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL", "CORE_COH_EMG_INDUCTIVE") for site in SITES}
        actual_site_keys = {(str(row.get("route")), str(row.get("holdout_site"))) for row in site_rows}
        if actual_site_keys != required_site_keys or len(site_rows) != 6:
            raise RouteStop("frozen site-score artifact coverage mismatch")
        for holdout in SITES:
            if not np.array_equal(caisr_groups[holdout][0], core_groups[holdout][0]):
                raise RouteStop("CAISR/CORE evaluation identity mismatch")
    else:
        site_rows = []
        for holdout in SITES:
            idx, caisr, _ = caisr_groups[holdout]
            labels, ages = load_eval_targets(metadata, idx)
            site_rows.append({"route": "CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL", "holdout_site": holdout, **score_site(labels, caisr, ages, OFFICIAL)})
            idx2, core, _ = core_groups[holdout]
            labels2, ages2 = load_eval_targets(metadata, idx2)
            if not np.array_equal(idx, idx2):
                raise RouteStop("CAISR/CORE evaluation identity mismatch")
            site_rows.append({"route": "CORE_COH_EMG_INDUCTIVE", "holdout_site": holdout, **score_site(labels2, core, ages2, OFFICIAL)})
    by_route = {route: {row["holdout_site"]: float(row["ac_auroc"]) for row in site_rows if row["route"] == route} for route in ("CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL", "CORE_COH_EMG_INDUCTIVE")}
    summary = []
    for route, values in by_route.items():
        summary.append({"route": route, "macro_ac": _scope_value(values, "Macro"), "worst_site_ac": _scope_value(values, "Worst-site"), "small_site_mean_ac": _scope_value(values, "Small-site mean")})
    core_values = by_route["CORE_COH_EMG_INDUCTIVE"]
    caisr_values = by_route["CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL"]
    delta = [{"scope": scope, "core_minus_caisr": _route_scope_delta(core_values, caisr_values, scope)} for scope in (*SITES, "Macro", "Worst-site", "Small-site mean")]

    stability = []
    for seed in SEEDS:
        seed_delta = {}
        core_seed_values = {}
        caisr_seed_values = {}
        for holdout in SITES:
            idx, core_seed, _, _ = _load_columns(old_rows, Path(core_root), "CORE_COH_EMG", holdout, "inductive", seed=seed)
            _, caisr_matrix = caisr_groups[holdout][0], caisr_groups[holdout][2]
            pos = SEEDS.index(seed)
            caisr_seed = _rank_vector(caisr_matrix[:, pos])
            labels, ages = load_eval_targets(metadata, idx)
            core_seed_values[holdout] = float(score_site(labels, core_seed, ages, OFFICIAL)["ac_auroc"])
            caisr_seed_values[holdout] = float(score_site(labels, caisr_seed, ages, OFFICIAL)["ac_auroc"])
            seed_delta[holdout] = core_seed_values[holdout] - caisr_seed_values[holdout]
        values = {}
        for scope in (*SITES, "Macro", "Worst-site", "Small-site mean"):
            # Preserve the five already-frozen scope rows bit-for-bit; only the
            # diagnostic Worst-site row changes to difference-of-route-minima.
            values[scope] = (_route_scope_delta(core_seed_values, caisr_seed_values, scope)
                             if scope == "Worst-site" else _scope_value(seed_delta, scope))
        stability.append({"seed": int(seed), **values})
    stability_summary = []
    for scope in (*SITES, "Macro", "Worst-site", "Small-site mean"):
        values = np.asarray([row[scope] for row in stability], dtype=np.float64)
        stability_summary.append({"scope": scope, "n_seeds": 15, "n_positive": int((values > 0).sum()), "n_zero": int((values == 0).sum()), "n_negative": int((values < 0).sum()), "mean_delta": float(values.mean()), "median_delta": float(np.median(values)), "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)), "min": float(values.min()), "max": float(values.max())})

    rng = np.random.default_rng(int(bootstrap_seed))
    draws = {}
    observed = {}
    for holdout in SITES:
        idx, caisr, _ = caisr_groups[holdout]
        _, core, _ = core_groups[holdout]
        labels, ages = load_eval_targets(metadata, idx)
        mult = np.vstack([bootstrap_multiplicity(len(labels), rng) for _ in range(int(n_bootstrap))])
        draws[holdout] = (weighted_ac_batch(pair_contributions(labels, ages, core), mult), weighted_ac_batch(pair_contributions(labels, ages, caisr), mult))
        if preserve_frozen_outputs:
            observed[holdout] = (core_values[holdout], caisr_values[holdout])
        else:
            observed[holdout] = (float(score_site(labels, core, ages, OFFICIAL)["ac_auroc"]), float(score_site(labels, caisr, ages, OFFICIAL)["ac_auroc"]))
    bootstrap = []
    for scope in (*SITES, "Macro", "Worst-site", "Small-site mean"):
        if scope in SITES:
            series = draws[scope][0] - draws[scope][1]
            observed_delta = observed[scope][0] - observed[scope][1]
        else:
            matrix = np.column_stack([draws[site][0] - draws[site][1] for site in SITES])
            if scope == "Worst-site":
                paired_core = np.column_stack([draws[site][0] for site in SITES])
                paired_caisr = np.column_stack([draws[site][1] for site in SITES])
                series = _route_scope_delta_batch(paired_core, paired_caisr, scope)
                observed_delta = _route_scope_delta(
                    {site: observed[site][0] for site in SITES},
                    {site: observed[site][1] for site in SITES},
                    scope,
                )
            else:
                series = matrix.mean(axis=1) if scope == "Macro" else matrix[:, :2].mean(axis=1)
                observed_delta = _scope_value({site: observed[site][0] - observed[site][1] for site in SITES}, scope)
        valid = np.isfinite(series)
        vals = series[valid]
        bootstrap.append({"scope": scope, "observed": float(observed_delta), "mean": float(np.mean(vals)), "median": float(np.median(vals)), "sd": float(np.std(vals, ddof=1)), "q025": float(np.quantile(vals, .025)), "q975": float(np.quantile(vals, .975)), "fraction_gt_0": float(np.mean(vals > 0)), "fraction_lt_0": float(np.mean(vals < 0)), "valid": int(valid.sum()), "invalid": int((~valid).sum()), "n_bootstrap": int(n_bootstrap), "bootstrap_seed": int(bootstrap_seed)})

    if preserve_frozen_outputs:
        frozen_correlation_path = Path(public_root) / "caisr20_vs_core_v2_score_correlation.csv"
        if not frozen_correlation_path.is_file():
            raise RouteStop(f"frozen score-correlation artifact is required in diagnostic mode: {frozen_correlation_path}")
        with frozen_correlation_path.open(newline="", encoding="utf-8") as handle:
            correlations = list(csv.DictReader(handle))
        if len(correlations) != len(SITES) or {str(row.get("holdout_site")) for row in correlations} != set(SITES):
            raise RouteStop("frozen score-correlation artifact coverage mismatch")
    else:
        correlations = []
        for holdout in SITES:
            core = core_groups[holdout][1]
            caisr = caisr_groups[holdout][1]
            core_rank = _rank_vector(core); caisr_rank = _rank_vector(caisr)
            pearson = float(np.corrcoef(core, caisr)[0, 1])
            spearman = float(np.corrcoef(core_rank, caisr_rank)[0, 1])
            correlations.append({"holdout_site": holdout, "pearson": pearson, "spearman": spearman, "n_records": int(len(core))})

    private_root = Path(private_root); private_root.mkdir(parents=True, exist_ok=True)
    (private_root / "stability.json").write_text(json.dumps(stability, sort_keys=True), encoding="utf-8")
    public_root = Path(public_root); public_root.mkdir(parents=True, exist_ok=True)
    def write(name, rows):
        if not rows: raise ValueError(f"empty public output: {name}")
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (public_root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    if not preserve_frozen_outputs:
        write("caisr20_vs_core_v2_site_scores.csv", site_rows)
        summary_rows = [{"row_type": "route_summary", **row} for row in summary] + [{"row_type": "delta_core_minus_caisr", **row} for row in delta]
        write("caisr20_vs_core_v2_summary.csv", summary_rows)
        write("caisr20_vs_core_v2_score_correlation.csv", correlations)
    write("caisr20_vs_core_v2_seed_stability.csv", stability_summary)
    write("caisr20_vs_core_v2_bootstrap.csv", bootstrap)
    manifest = {"schema": "featurelab_caisr20_vs_core_protocol_v2_manifest_v2", "route": "CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL", "comparator": "CORE_COH_EMG_INDUCTIVE", "site_scores": 6, "summary_rows": len(summary) + len(delta), "seed_stability_rows": len(stability_summary), "bootstrap_rows": len(bootstrap), "correlation_rows": len(correlations), "bootstrap_seed": int(bootstrap_seed), "bootstrap_replicates": int(n_bootstrap), "private_stability": True, "worst_site_semantics": "difference_of_route_minima", "worst_site_seed_fix": "PASS", "worst_site_bootstrap_fix": "PASS", "model_refit_count": 0, "prediction_recompute_count": 0}
    (public_root / "caisr20_vs_core_v2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"site_rows": len(site_rows), "summary_rows": len(summary) + len(delta), "stability_rows": len(stability_summary), "bootstrap_rows": len(bootstrap), "correlation_rows": len(correlations), "delta": delta, "caisr": caisr_values, "core": core_values}


__all__ = ["RouteStop", "array_digest", "historical_bridge", "historical_typed_pipeline_bridge", "finalize_route_comparison", "load_data2_inputs", "load_frozen_sample", "load_typed_rules", "publish_unit", "run_route", "sha256_file", "source_value_bridge", "_route_scope_delta", "_route_scope_delta_batch"]
