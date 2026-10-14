"""G4-A Seed 42 training, prediction, no-clobber resume and verification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import joblib
import lightgbm
import numpy as np
import scipy
import sklearn

from reproduction.siren_a.pipeline.input_data import check_cross_protocol_train, load_fold
from reproduction.siren_a.pipeline.run_inputs import (
    array_sha256, atomic_json_no_clobber, atomic_npz_no_clobber,
    canonical_json, default_config as g3b_default_config, preflight as g3b_preflight,
    sha256_file, verify_private_output,
)
from .config import (
    CANONICAL_HOLDOUT, DEMOGRAPHICS_SHA256, ENV_AUDIT_ROOT, ENV_PYTHON,
    EXPECTED_SELECTED, EXPECTED_TRAIN, G2_FINGERPRINT, G3_FINGERPRINT,
    HOLDOUTS, HOLDOUT_TO_TRAIN, MANIFEST_SHA256, MODEL_TYPES, PRIVATE_ROOT,
    PROTOCOLS, SAMPLER_VERSION, SCHEMA, SEED, SPEC_SHA256, TRAIN_SITES,
    VIEW_SIZES, VIEWS,
)
from .models import fit_frozen_model, load_checkpoint, predict_scores, save_checkpoint
from .queries import QueryResult, audit_ac_pair_coverage, build_queries
from .sampler import SampleResult, sample_training_rows
from .verify_g4a import complete_unit, sha256_file as local_sha256, verify_model_parameters, verify_unit


THREAD_ENV = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


@dataclass(frozen=True)
class G4AConfig:
    repo: Path
    private_root: Path
    public_csv: Path
    public_report: Path
    environment_audit: Path


def default_config() -> G4AConfig:
    repo = Path(__file__).resolve().parents[3]
    return G4AConfig(repo, PRIVATE_ROOT,
                     repo / "reproduction/siren_a/audit/g4a_smoke_summary.csv",
                     repo / "reproduction/siren_a/docs/G4A_TRAIN_REPORT.md",
                     ENV_AUDIT_ROOT)


def model_identities() -> Tuple[Tuple[str, int, str, str], ...]:
    return tuple((site, SEED, view, model) for site in TRAIN_SITES
                 for view in VIEWS for model in MODEL_TYPES)


def prediction_identities() -> Tuple[Tuple[str, str, str, str, int], ...]:
    return tuple((holdout, protocol, view, model, SEED) for holdout in HOLDOUTS
                 for protocol in PROTOCOLS for view in VIEWS for model in MODEL_TYPES)


def _identity_sha(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(identity)).hexdigest()


def _environment_identity(config: G4AConfig) -> Mapping[str, object]:
    synthetic = config.environment_audit / "synthetic_api_test.json"
    wheel = config.environment_audit / "lightgbm_wheel.sha256"
    freeze = config.environment_audit / "pip_freeze_after.txt"
    if not synthetic.is_file() or not wheel.is_file() or not freeze.is_file():
        raise ValueError("G4-A environment evidence incomplete")
    api = json.loads(synthetic.read_text(encoding="utf-8"))
    if api.get("status") != "PASS" or api.get("lightgbm") != "4.6.0":
        raise ValueError("G4-A environment API evidence invalid")
    if Path(sys.executable).resolve() != ENV_PYTHON.resolve():
        raise ValueError("G4-A must use the frozen sleepfm_env Python")
    if os.environ.get("PYTHONNOUSERSITE") != "1" or any(os.environ.get(k) != v for k, v in THREAD_ENV.items()):
        raise ValueError("G4-A process environment is not frozen")
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "joblib": joblib.__version__, "lightgbm": lightgbm.__version__}
    expected = {"python": "3.10.20", "numpy": "2.2.6", "scipy": "1.15.3",
                "sklearn": "1.7.2", "joblib": "1.5.3", "lightgbm": "4.6.0"}
    if versions != expected:
        raise ValueError(f"G4-A package drift: {versions}")
    return {"python_executable": str(ENV_PYTHON), "versions": versions,
            "pip_freeze_sha256": local_sha256(freeze),
            "wheel_record_sha256": local_sha256(wheel), "thread_env": THREAD_ENV}


def _source_identity(config: G4AConfig) -> Mapping[str, object]:
    g3b_config = g3b_default_config()
    pre = g3b_preflight(g3b_config)
    verify_private_output(g3b_config.private_root)
    saved = json.loads((g3b_config.private_root / "run_identity.json").read_text(encoding="utf-8"))
    expected = {"g2b1_cache_fingerprint": G2_FINGERPRINT,
                "g3a_artifact_fingerprint": G3_FINGERPRINT,
                "manifest_sha256": MANIFEST_SHA256, "spec_sha256": SPEC_SHA256,
                "demographics_sha256": DEMOGRAPHICS_SHA256}
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("saved G3-B source identity drift")
    code = {}
    for name in ("config.py", "sampler.py", "queries.py", "models.py", "run_g4a.py", "verify_g4a.py"):
        code[name] = local_sha256(Path(__file__).parent / name)
    identity = {"schema": SCHEMA, "seed": SEED, **expected,
                "g3b_run_identity_sha256": local_sha256(g3b_config.private_root / "run_identity.json"),
                "g3b_complete_sha256": local_sha256(g3b_config.private_root / "complete.json"),
                "metadata_join_sha256": saved["metadata_join_sha256"],
                "environment": _environment_identity(config), "code_sha256": code,
                "sampler_version": SAMPLER_VERSION}
    return {"identity": identity, "g3b": g3b_config, "pre": pre}


def _load(config_bundle, holdout: str, protocol: str, view: str):
    g3b = config_bundle["g3b"]
    pre = config_bundle["pre"]
    return load_fold(g3b.g3_root, holdout, protocol, view, pre["metadata"],
                     g3b.manifest_path, pre["identity"])


def _check_train_sharing(bundle) -> None:
    for view in VIEWS:
        s1 = _load(bundle, "I0002", "transductive", view)
        comparisons = [_load(bundle, "I0002", "inductive", view),
                       _load(bundle, "I0006", "transductive", view),
                       _load(bundle, "I0006", "inductive", view)]
        for other in comparisons:
            if other.train_site != "S0001":
                raise ValueError("S0001 sharing identity failed")
            for name in ("X_train", "train_valid_mask", "y_train", "age_train",
                         "train_age_groups", "train_indices"):
                if not np.array_equal(getattr(s1, name), getattr(other, name), equal_nan=True):
                    raise ValueError(f"shared S0001 training mismatch: {view}/{name}")
            if s1.feature_ids != other.feature_ids:
                raise ValueError("shared S0001 feature order mismatch")
        i6a = _load(bundle, "S0001", "transductive", view)
        i6b = _load(bundle, "S0001", "inductive", view)
        check_cross_protocol_train(i6a, i6b)


def preflight(config: G4AConfig) -> Mapping[str, object]:
    bundle = _source_identity(config)
    _check_train_sharing(bundle)
    for site in TRAIN_SITES:
        fold = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", "V1")
        if fold.train_site != site or len(fold.X_train) != EXPECTED_TRAIN[site]:
            raise ValueError("canonical training fold mismatch")
    root = config.private_root
    if root.is_symlink():
        raise ValueError("private root may not be a symlink")
    if root.exists() and (not root.is_dir() or not (root / "run_identity.json").is_file()):
        raise ValueError("existing private root is not a valid G4-A run")
    return bundle


def _publish_building(building: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(target)
    try:
        for directory in sorted((p for p in building.rglob("*") if p.is_dir()), key=lambda p: len(p.parts)):
            (target / directory.relative_to(building)).mkdir()
        files = sorted((p for p in building.rglob("*") if p.is_file()),
                       key=lambda p: (p.name == "complete.json", p.as_posix()))
        for source in files:
            os.link(source, target / source.relative_to(building))
    except Exception:
        raise
    shutil.rmtree(building)


def _build_unit(target: Path, schema: str, identity: Mapping[str, object], writer) -> str:
    if target.exists() or target.is_symlink():
        verify_unit(target, schema, identity)
        return "SKIP_VERIFIED"
    building = target.parent / (target.name + ".building")
    if building.exists() or building.is_symlink():
        raise ValueError(f"incomplete unit evidence present: {building}")
    building.mkdir(parents=True)
    try:
        writer(building)
        complete_unit(building, schema, identity)
        verify_unit(building, schema, identity)
        _publish_building(building, target)
        verify_unit(target, schema, identity)
    except Exception:
        raise
    return "BUILT_NEW"


def _sample_path(root: Path, site: str) -> Path:
    return root / "samples" / f"{site}_seed42"


def _query_path(root: Path, site: str) -> Path:
    return root / "queries" / f"{site}_seed42"


def _model_path(root: Path, site: str, view: str, model: str) -> Path:
    return root / "models" / site / "seed42" / view / model


def _prediction_path(root: Path, holdout: str, protocol: str, view: str, model: str) -> Path:
    return root / "predictions" / f"holdout_{holdout}" / protocol / view / model


def _save_sample(config: G4AConfig, run_sha: str, fold, site: str) -> Tuple[SampleResult, str]:
    sample = sample_training_rows(site, SEED, fold.train_age_groups, fold.y_train, fold.train_indices)
    if sample.target_n != EXPECTED_SELECTED[site]:
        raise ValueError("real sampling target mismatch")
    identity = {"run_identity_sha256": run_sha, "train_site": site, "seed": SEED,
                "sample_sha256": sample.index_sha256, "schema": "g4a_sample_v1"}
    def writer(path):
        atomic_npz_no_clobber(path / "selected_indices.npz",
                              local_positions=sample.local_positions,
                              global_indices=sample.global_indices)
        strata = [{"age_group": key[0], "label": key[1], "original": sample.stratum_sizes[key],
                   "quota": sample.quotas[key]} for key in sorted(sample.quotas)]
        selected_y = fold.y_train[sample.local_positions]
        atomic_json_no_clobber(path / "strata_audit.json",
                               {"train_site": site, "seed": SEED, "train_total": len(fold.y_train),
                                "target": sample.target_n, "actual": len(sample.local_positions),
                                "positive": int(np.sum(selected_y == 1)),
                                "negative": int(np.sum(selected_y == 0)), "strata": strata,
                                "index_sha256": sample.index_sha256,
                                "sampler_version": SAMPLER_VERSION,
                                "numpy_version": np.__version__})
    action = _build_unit(_sample_path(config.private_root, site), "g4a_sample_v1", identity, writer)
    with np.load(_sample_path(config.private_root, site) / "selected_indices.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["local_positions"], sample.local_positions) or not np.array_equal(archive["global_indices"], sample.global_indices):
            raise ValueError("saved sample differs from recomputation")
    return sample, action


def _save_query(config: G4AConfig, run_sha: str, fold, site: str,
                sample: SampleResult) -> Tuple[QueryResult, Mapping[str, object], str]:
    local = sample.local_positions
    query = build_queries(fold.age_train[local], fold.y_train[local], local, sample.global_indices)
    coverage = audit_ac_pair_coverage(fold.age_train[local], fold.y_train[local],
                                      fold.train_age_groups[local])
    identity = {"run_identity_sha256": run_sha, "train_site": site, "seed": SEED,
                "sample_sha256": sample.index_sha256, "query_sha256": query.fingerprint,
                "schema": "g4a_query_v1"}
    def writer(path):
        atomic_npz_no_clobber(path / "query.npz", permutation=query.permutation,
                              inverse_permutation=query.inverse_permutation,
                              sorted_local_positions=query.sorted_local_positions,
                              sorted_global_indices=query.sorted_global_indices,
                              sorted_group_ids=query.sorted_group_ids,
                              sorted_labels=query.sorted_labels, group_ids=query.group_ids,
                              group_sizes=query.group_sizes, group_positive=query.group_positive,
                              group_negative=query.group_negative)
        atomic_json_no_clobber(path / "query_audit.json",
                               {"train_site": site, "query_count": len(query.group_sizes),
                                "maximum_group_size": query.truncation_level,
                                "truncation_level": query.truncation_level,
                                "query_fingerprint": query.fingerprint, **coverage})
    action = _build_unit(_query_path(config.private_root, site), "g4a_query_v1", identity, writer)
    return query, coverage, action


def _model_identity(run_sha: str, site: str, view: str, model: str,
                    sample: SampleResult, query: QueryResult, feature_sha: str) -> Mapping[str, object]:
    return {"run_identity_sha256": run_sha, "train_site": site, "seed": SEED,
            "view": view, "model_type": model, "sample_sha256": sample.index_sha256,
            "query_sha256": query.fingerprint, "feature_ids_sha256": feature_sha,
            "schema": "g4a_model_v1"}


def _checkpoint_files(path: Path, model_type: str) -> Mapping[str, str]:
    model_name = "model.txt" if model_type == "lambdarank" else "model.joblib"
    return {model_name: local_sha256(path / model_name),
            "adapter_state.npz": local_sha256(path / "adapter_state.npz")}


def _train_model(config: G4AConfig, run_sha: str, bundle, site: str, view: str, model_type: str,
                 sample: SampleResult, query: QueryResult) -> Tuple[Mapping[str, object], str]:
    fold = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", view)
    X = np.asarray(fold.X_train[sample.local_positions], dtype=np.float64)
    y = np.asarray(fold.y_train[sample.local_positions], dtype=np.int8)
    feature_sha = hashlib.sha256("\n".join(fold.feature_ids).encode()).hexdigest()
    identity = _model_identity(run_sha, site, view, model_type, sample, query, feature_sha)
    target = _model_path(config.private_root, site, view, model_type)
    def writer(path):
        start = time.perf_counter()
        before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trained = fit_frozen_model(model_type, X, y, fold.feature_ids,
                                       sample.global_indices, query)
        fit_seconds = time.perf_counter() - start
        checkpoint_files = save_checkpoint(path, trained)
        loaded = load_checkpoint(path, model_type, checkpoint_files)
        replay = predict_scores(model_type, loaded.model, loaded.adapter, X)
        difference = float(np.max(np.abs(replay - trained.train_score)))
        if difference > 1e-12 or np.ptp(replay) == 0:
            raise ValueError("checkpoint reload prediction mismatch or constant")
        iterations = (int(np.max(np.asarray(trained.model.n_iter_))) if hasattr(trained.model, "n_iter_")
                      else int(trained.model.booster_.num_trees()))
        metadata = {**identity, "parameters": trained.parameters, "train_n": len(X),
                    "positive": int(np.sum(y == 1)), "negative": int(np.sum(y == 0)),
                    "feature_count": X.shape[1], "missing_count": int(np.isnan(X).sum()),
                    "fit_seconds": fit_seconds, "peak_rss_kib": int(max(before_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
                    "iterations": iterations, "warning_count": len(caught),
                    "warnings": [str(item.message) for item in caught],
                    "checkpoint_files": checkpoint_files,
                    "adapter_state_sha256": checkpoint_files["adapter_state.npz"],
                    "reload_max_abs_diff": difference,
                    "train_score_min": float(replay.min()), "train_score_max": float(replay.max()),
                    "train_score_std": float(replay.std()),
                    "score_type": "positive_probability" if model_type in {"lr", "hgb"} else "raw_ranking_score"}
        atomic_json_no_clobber(path / "metadata.json", metadata)
    action = _build_unit(target, "g4a_model_v1", identity, writer)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    files = _checkpoint_files(target, model_type)
    loaded = load_checkpoint(target, model_type, files)
    replay = predict_scores(model_type, loaded.model, loaded.adapter, X)
    if np.ptp(replay) == 0 or not np.isfinite(replay).all():
        raise ValueError("verified model produces invalid training score")
    return metadata, action


def _predict(config: G4AConfig, run_sha: str, bundle, holdout: str, protocol: str,
             view: str, model_type: str, model_meta: Mapping[str, object]) -> Tuple[Mapping[str, object], str]:
    site = HOLDOUT_TO_TRAIN[holdout]
    fold = _load(bundle, holdout, protocol, view)
    model_path = _model_path(config.private_root, site, view, model_type)
    files = _checkpoint_files(model_path, model_type)
    loaded = load_checkpoint(model_path, model_type, files)
    scores = predict_scores(model_type, loaded.model, loaded.adapter, fold.X_test)
    if np.ptp(scores) == 0:
        raise ValueError("strictly constant test prediction")
    test_sha = array_sha256(fold.test_indices)
    prediction_sha = array_sha256(scores)
    model_complete_sha = local_sha256(model_path / "complete.json")
    identity = {"run_identity_sha256": run_sha, "holdout_site": holdout,
                "train_site": site, "protocol": protocol, "view": view,
                "model_type": model_type, "seed": SEED, "test_indices_sha256": test_sha,
                "model_complete_sha256": model_complete_sha,
                "prediction_sha256": prediction_sha, "schema": "g4a_prediction_v1"}
    target = _prediction_path(config.private_root, holdout, protocol, view, model_type)
    def writer(path):
        atomic_npz_no_clobber(path / "prediction.npz", scores=scores,
                              test_global_indices=fold.test_indices)
        atomic_json_no_clobber(path / "metadata.json",
                               {**identity, "test_count": len(scores),
                                "feature_count": len(fold.feature_ids),
                                "feature_ids_sha256": model_meta["feature_ids_sha256"],
                                "adapter_state_sha256": model_meta["adapter_state_sha256"],
                                "checkpoint_sha256": files["model.txt" if model_type == "lambdarank" else "model.joblib"],
                                "score_type": model_meta["score_type"], "finite": True,
                                "nonconstant": True, "score_min": float(scores.min()),
                                "score_max": float(scores.max()), "score_std": float(scores.std())})
    action = _build_unit(target, "g4a_prediction_v1", identity, writer)
    with np.load(target / "prediction.npz", allow_pickle=False) as archive:
        saved = np.asarray(archive["scores"])
        indices = np.asarray(archive["test_global_indices"])
    replay = predict_scores(model_type, loaded.model, loaded.adapter, fold.X_test)
    if not np.array_equal(indices, fold.test_indices) or np.max(np.abs(saved - replay)) > 1e-12:
        raise ValueError("saved prediction replay mismatch")
    return json.loads((target / "metadata.json").read_text(encoding="utf-8")), action


def _public_rows(config: G4AConfig) -> Sequence[Mapping[str, object]]:
    rows = []
    for holdout, protocol, view, model, seed in prediction_identities():
        pred = json.loads((_prediction_path(config.private_root, holdout, protocol, view, model) / "metadata.json").read_text())
        site = HOLDOUT_TO_TRAIN[holdout]
        model_meta = json.loads((_model_path(config.private_root, site, view, model) / "metadata.json").read_text())
        sample = json.loads((_sample_path(config.private_root, site) / "strata_audit.json").read_text())
        query = json.loads((_query_path(config.private_root, site) / "query_audit.json").read_text())
        rows.append({"holdout_site": holdout, "train_site": site, "protocol": protocol,
                     "view": view, "model_type": model, "seed": seed,
                     "train_total": sample["train_total"], "train_selected": sample["actual"],
                     "train_positive": sample["positive"], "train_negative": sample["negative"],
                     "feature_count": model_meta["feature_count"], "age_group_count": query["query_count"],
                     "sampler_verified": "PASS", "query_verified": "PASS",
                     "checkpoint_verified": "PASS", "test_count": pred["test_count"],
                     "prediction_finite": "PASS", "prediction_nonconstant": "PASS",
                     "prediction_reload_match": "PASS", "source_verified": "PASS", "status": "PASS"})
    return rows


def _write_public_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if len(rows) != 90:
        raise ValueError("public summary must have 90 rows")
    fields = list(rows[0])
    output = tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False)
    temporary = Path(output.name)
    try:
        with output:
            writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows); output.flush(); os.fsync(output.fileno())
        if path.exists():
            if path.read_bytes() != temporary.read_bytes():
                raise ValueError("existing public report differs")
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def _initialize_root(config: G4AConfig, identity: Mapping[str, object]) -> None:
    root = config.private_root
    root_sha = _identity_sha(identity)
    if root.exists():
        saved = json.loads((root / "run_identity.json").read_text())
        if saved != identity:
            raise ValueError("G4-A private run identity drift")
        return
    root.mkdir(parents=True)
    atomic_json_no_clobber(root / "run_identity.json", identity)
    atomic_json_no_clobber(root / "config_frozen.json",
                           {"schema": SCHEMA, "seed": SEED, "models": list(MODEL_TYPES),
                            "views": list(VIEWS), "protocols": list(PROTOCOLS),
                            "run_identity_sha256": root_sha})


def build(config: G4AConfig) -> Mapping[str, object]:
    started = time.perf_counter()
    bundle = preflight(config)
    identity = bundle["identity"]
    run_sha = _identity_sha(identity)
    _initialize_root(config, identity)
    samples: Dict[str, SampleResult] = {}
    queries: Dict[str, QueryResult] = {}
    sample_actions = {}; query_actions = {}
    for site in TRAIN_SITES:
        fold = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", "V1")
        samples[site], sample_actions[site] = _save_sample(config, run_sha, fold, site)
        queries[site], _, query_actions[site] = _save_query(config, run_sha, fold, site, samples[site])
    model_meta = {}; model_actions = {}
    ordered = list(model_identities())
    ordered.sort(key=lambda item: (item[2] != "V1", TRAIN_SITES.index(item[0]),
                                   VIEWS.index(item[2]), MODEL_TYPES.index(item[3])))
    for site, _, view, model in ordered:
        model_meta[(site, view, model)], model_actions[(site, view, model)] = _train_model(
            config, run_sha, bundle, site, view, model, samples[site], queries[site])
    prediction_actions = {}
    for holdout, protocol, view, model, _ in prediction_identities():
        site = HOLDOUT_TO_TRAIN[holdout]
        _, prediction_actions[(holdout, protocol, view, model)] = _predict(
            config, run_sha, bundle, holdout, protocol, view, model,
            model_meta[(site, view, model)])
    rows = _public_rows(config)
    _write_public_csv(config.public_csv, rows)
    summary = {"schema": SCHEMA, "status": "PASS", "unique_models": 30,
               "prediction_units": 90, "sample_units": 2, "query_units": 2,
               "elapsed_seconds": time.perf_counter() - started,
               "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
               "sample_actions": sample_actions, "query_actions": query_actions,
               "models_built": sum(value == "BUILT_NEW" for value in model_actions.values()),
               "models_skipped": sum(value == "SKIP_VERIFIED" for value in model_actions.values()),
               "predictions_built": sum(value == "BUILT_NEW" for value in prediction_actions.values()),
               "predictions_skipped": sum(value == "SKIP_VERIFIED" for value in prediction_actions.values())}
    report_path = config.private_root / "reports" / "run_summary.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if {k: existing[k] for k in ("schema", "status", "unique_models", "prediction_units")} != {k: summary[k] for k in ("schema", "status", "unique_models", "prediction_units")}:
            raise ValueError("existing run summary invalid")
    else:
        atomic_json_no_clobber(report_path, summary)
    if not (config.private_root / "complete.json").exists():
        complete_unit(config.private_root, SCHEMA, {"run_identity_sha256": run_sha})
    verify_result = verify_all(config, bundle=bundle)
    return {**summary, "verification": verify_result}


def verify_all(config: G4AConfig, bundle=None) -> Mapping[str, object]:
    bundle = preflight(config) if bundle is None else bundle
    identity = bundle["identity"]
    run_sha = _identity_sha(identity)
    verify_unit(config.private_root, SCHEMA, {"run_identity_sha256": run_sha})
    if json.loads((config.private_root / "run_identity.json").read_text()) != identity:
        raise ValueError("root identity mismatch")
    samples = {}; queries = {}
    for site in TRAIN_SITES:
        fold = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", "V1")
        sample = sample_training_rows(site, SEED, fold.train_age_groups, fold.y_train, fold.train_indices)
        sample_identity = {"run_identity_sha256": run_sha, "train_site": site, "seed": SEED,
                           "sample_sha256": sample.index_sha256, "schema": "g4a_sample_v1"}
        verify_unit(_sample_path(config.private_root, site), "g4a_sample_v1", sample_identity)
        with np.load(_sample_path(config.private_root, site) / "selected_indices.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["local_positions"], sample.local_positions) or not np.array_equal(archive["global_indices"], sample.global_indices):
                raise ValueError("sample replay mismatch")
        query = build_queries(fold.age_train[sample.local_positions], fold.y_train[sample.local_positions],
                              sample.local_positions, sample.global_indices)
        query_identity = {"run_identity_sha256": run_sha, "train_site": site, "seed": SEED,
                          "sample_sha256": sample.index_sha256, "query_sha256": query.fingerprint,
                          "schema": "g4a_query_v1"}
        verify_unit(_query_path(config.private_root, site), "g4a_query_v1", query_identity)
        samples[site] = sample; queries[site] = query
    model_meta = {}
    for site, _, view, model_type in model_identities():
        fold = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", view)
        feature_sha = hashlib.sha256("\n".join(fold.feature_ids).encode()).hexdigest()
        model_identity = _model_identity(run_sha, site, view, model_type,
                                         samples[site], queries[site], feature_sha)
        path = _model_path(config.private_root, site, view, model_type)
        verify_unit(path, "g4a_model_v1", model_identity)
        meta = json.loads((path / "metadata.json").read_text())
        if not verify_model_parameters(model_type, meta["parameters"], queries[site].truncation_level):
            raise ValueError("model metadata invalid")
        loaded = load_checkpoint(path, model_type, _checkpoint_files(path, model_type))
        X = fold.X_train[samples[site].local_positions]
        score = predict_scores(model_type, loaded.model, loaded.adapter, X)
        if not np.isfinite(score).all() or np.ptp(score) == 0:
            raise ValueError("model replay invalid")
        model_meta[(site, view, model_type)] = meta
    for holdout, protocol, view, model_type, _ in prediction_identities():
        site = HOLDOUT_TO_TRAIN[holdout]
        fold = _load(bundle, holdout, protocol, view)
        model_path = _model_path(config.private_root, site, view, model_type)
        loaded = load_checkpoint(model_path, model_type, _checkpoint_files(model_path, model_type))
        replay = predict_scores(model_type, loaded.model, loaded.adapter, fold.X_test)
        target = _prediction_path(config.private_root, holdout, protocol, view, model_type)
        with np.load(target / "prediction.npz", allow_pickle=False) as archive:
            saved = np.asarray(archive["scores"]); indices = np.asarray(archive["test_global_indices"])
        prediction_sha = array_sha256(replay)
        pred_identity = {"run_identity_sha256": run_sha, "holdout_site": holdout,
                         "train_site": site, "protocol": protocol, "view": view,
                         "model_type": model_type, "seed": SEED,
                         "test_indices_sha256": array_sha256(fold.test_indices),
                         "model_complete_sha256": local_sha256(model_path / "complete.json"),
                         "prediction_sha256": prediction_sha, "schema": "g4a_prediction_v1"}
        verify_unit(target, "g4a_prediction_v1", pred_identity)
        if not np.array_equal(indices, fold.test_indices) or np.max(np.abs(saved - replay)) > 1e-12:
            raise ValueError("prediction replay mismatch")
    rows = _public_rows(config)
    if len(rows) != 90:
        raise ValueError("public closure failed")
    return {"status": "PASS", "samples": 2, "queries": 2, "models": 30,
            "predictions": 90, "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen G4-A Seed 42 training smoke")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = default_config()
    if args.preflight_only:
        pre = preflight(config)
        result = {"status": "PASS", "source_identity_sha256": _identity_sha(pre["identity"])}
    elif args.verify_only:
        result = verify_all(config)
    else:
        result = build(config)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "verification"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
