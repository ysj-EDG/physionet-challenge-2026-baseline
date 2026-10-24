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
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .caisr20_core_protocol import (
    CAISR20_FEATURE_IDS, CAISR20_X_STATIC_INDICES, EXPECTED_SITE_COUNTS,
    EXPECTED_SITE_POSITIVES, HOLDOUT_TO_TRAIN, ROUTE_ID, SEEDS, SITES,
    apply_selected_preprocessor, fit_selected_preprocessor, manifest_payload,
    mean_rank_fusion, model_config, typed_caisr20,
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
    return {"manifest": manifest, "raw_caisr": raw, "record_id": ids, "site": sites}


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
    status = "EXACT" if value_ok and validity_ok else "DRIFT"
    return {"status": status, "value_mismatch_total": int(value_mismatch.sum()),
            "validity_mismatch_total": int(validity_mismatch.sum()),
            "features": by_feature}



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


def _fit_or_load_model(*, root: Path, train_site: str, seed: int, x_typed: np.ndarray, labels: np.ndarray, sample: Mapping[str, object], manifest_sha: str) -> tuple[str, dict[str, object], object]:
    path = Path(root) / "models" / train_site / f"seed{int(seed)}"
    selected = np.asarray(sample["global_indices"], dtype=np.int64)
    x_selected = x_typed[selected]
    y_selected = np.asarray(labels[selected], dtype=np.int8)
    state = fit_selected_preprocessor(x_selected)
    x_fit = apply_selected_preprocessor(x_selected, state)
    cfg = model_config(seed)
    meta = {
        "route_id": ROUTE_ID, "train_site": train_site, "seed": int(seed),
        "feature_manifest_sha256": manifest_sha, "training_sample_sha256": str(sample["sample_sha256"]),
        "preprocessing_identity": {"fit_scope": "selected_training_rows_only", "median_sha256": array_digest(state.median), "center_sha256": array_digest(state.center), "scale_sha256": array_digest(state.scale)},
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
        loaded_state = type("State", (), {"median": payload["median"], "center": payload["center"], "scale": payload["scale"], "all_missing": payload["all_missing"]})()
        return "SKIP_VERIFIED", saved_meta, (_model_from_payload(saved_meta, payload), loaded_state)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(**cfg).fit(x_fit, y_selected)
    convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    converged = not convergence_warnings and int(np.max(model.n_iter_)) < int(cfg["max_iter"])
    if not converged:
        raise RouteStop(f"CAISR20 model did not converge: {train_site} seed {seed}")
    meta = dict(meta, n_iter=int(model.n_iter_[0]), converged=True, warnings=convergence_warnings)
    payload = {"coef": np.asarray(model.coef_, dtype=np.float64), "intercept": np.asarray(model.intercept_, dtype=np.float64), "classes": np.asarray(model.classes_, dtype=np.int8), "n_iter": np.asarray(model.n_iter_, dtype=np.int32), "median": state.median, "center": state.center, "scale": state.scale, "all_missing": state.all_missing.astype(bool)}
    status = publish_unit(path, payload, meta)
    return status, meta, (model, state)


def _load_model_unit(path: Path, metadata: Mapping[str, object]):
    with np.load(Path(path) / "payload.npz", allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    _validate_unit(Path(path), payload, metadata)
    state = type("State", (), {"median": payload["median"], "center": payload["center"], "scale": payload["scale"], "all_missing": payload["all_missing"]})()
    return _model_from_payload(metadata, payload), state


def _prediction_unit(root: Path, train_site: str, seed: int, holdout: str, model_path: Path, model_meta: Mapping[str, object], scores: np.ndarray, test_indices: np.ndarray) -> str:
    path = Path(root) / "predictions" / f"seed{int(seed)}" / f"holdout_{holdout}"
    metadata = {"route_id": ROUTE_ID, "train_site": train_site, "seed": int(seed), "holdout_site": holdout, "feature_manifest_sha256": model_meta["feature_manifest_sha256"], "training_sample_sha256": model_meta["training_sample_sha256"], "model_unit_digest": _tree_digest(model_path), "test_indices_sha256": array_digest(test_indices), "score_sha256": array_digest(scores)}
    payload = {"scores": np.asarray(scores, dtype=np.float64), "test_global_indices": np.asarray(test_indices, dtype=np.int64)}
    return publish_unit(path, payload, metadata)


def run_route(*, manifest_path: Path, summary_cache: Path, rules_path: Path, pool_root: Path, sample_root: Path, output_root: Path, strict_source_bridge: bool = True) -> dict[str, object]:
    """Run gates, then (only if all gates pass) 30 models and 45 predictions."""
    started = time.time()
    data = load_data2_inputs(manifest_path, summary_cache)
    manifest = data["manifest"]; raw = data["raw_caisr"]; sites = data["site"]
    rules = load_typed_rules(rules_path)
    bridge = source_value_bridge(pool_root=pool_root, raw_caisr=raw, manifest=manifest)
    if strict_source_bridge and bridge["status"] == "DRIFT":
        raise RouteStop(f"CAISR20 source-value bridge failed: {bridge['value_mismatch_total']} value mismatches")
    typed = typed_caisr20(raw, rules)
    manifest_sha = hashlib.sha256(_canonical(manifest_payload())).hexdigest()
    out = Path(output_root); out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_bytes(_canonical(dict(manifest_payload(), manifest_sha256=manifest_sha)) + b"\n")
    samples = {(site, seed): load_frozen_sample(seed, site, sample_root, sites) for site in ("S0001", "I0006") for seed in SEEDS}
    model_statuses = []; prediction_statuses = []; model_cache = {}
    labels = manifest.label.to_numpy(np.int8)
    for train_site in ("S0001", "I0006"):
        for seed in SEEDS:
            status, meta, model_state = _fit_or_load_model(root=out, train_site=train_site, seed=seed, x_typed=typed, labels=labels, sample=samples[(train_site, seed)], manifest_sha=manifest_sha)
            model_statuses.append(status); model_cache[(train_site, seed)] = (meta, model_state)
    # No evaluation targets are read until all model units have completed.
    for holdout in SITES:
        train_site = HOLDOUT_TO_TRAIN[holdout]
        test_indices = np.flatnonzero(sites == holdout).astype(np.int64)
        for seed in SEEDS:
            meta, (model, state) = model_cache[(train_site, seed)]
            scores = model.decision_function(apply_selected_preprocessor(typed[test_indices], state)).astype(np.float64)
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


__all__ = ["RouteStop", "array_digest", "historical_bridge", "load_data2_inputs", "load_frozen_sample", "load_typed_rules", "publish_unit", "run_route", "sha256_file", "source_value_bridge"]
