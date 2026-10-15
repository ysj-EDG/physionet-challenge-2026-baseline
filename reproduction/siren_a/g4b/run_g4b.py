"""Incremental G4-B multi-seed training, prediction, and recovery runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
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

from reproduction.siren_a.g4.models import load_checkpoint, predict_scores, save_checkpoint
from reproduction.siren_a.g4.queries import audit_ac_pair_coverage, build_queries
from reproduction.siren_a.g4.run_g4a import (
    _identity_sha,
    _load,
    _source_identity,
    default_config as g4a_default_config,
    verify_all as verify_g4a_all,
)
from reproduction.siren_a.g4.sampler import sample_training_rows
from reproduction.siren_a.g4.verify_g4a import complete_unit, sha256_file, verify_unit
from reproduction.siren_a.pipeline.run_inputs import (
    array_sha256,
    atomic_json_no_clobber,
    atomic_npz_no_clobber,
)

from .config import (
    CANONICAL_HOLDOUT,
    DEMOGRAPHICS_SHA256,
    ENV_PYTHON,
    EXPECTED_SELECTED,
    EXPECTED_TRAIN,
    G2_FINGERPRINT,
    G3_FINGERPRINT,
    G3B_COMPLETE_SHA256,
    G3B_RUN_IDENTITY_SHA256,
    G4A_ROOT,
    G4A_RUN_IDENTITY_SHA256,
    HOLDOUTS,
    HOLDOUT_TO_TRAIN,
    MANIFEST_SHA256,
    MODEL_TYPES,
    NEW_SEEDS,
    PRIVATE_ROOT,
    PROTOCOLS,
    SAMPLER_VERSION,
    SCHEMA,
    SPEC_SHA256,
    THREAD_ENV,
    TRAIN_SITES,
    VIEW_SIZES,
    VIEWS,
    new_model_identities,
    new_prediction_identities,
    parameters_for_seed,
)
from .models import fit_seed_model, verify_runtime_parameters
from .manifest import build_g5_ready, verify_g5_ready
from .verify_g4b import verify_adapter_artifact, verify_query_artifact


APPROVED_PARENT_SHA = "bcbe09354b898fb19556235d67d90bfc0aa085f1"
LEGACY_RUN_IDENTITY_SHA256 = "544d4c192ed0a3b5fdbd5452d36edf436af6ed52ece5d88077f327e13470b412"
LEGACY_CODE_SHA256 = {
    "config.py": "b561e7cbdce8f55a3694ac1029b44f7396350fd59343ff2137d4eda031b515bf",
    "manifest.py": "a9b78d762027797d8622038989d09347b88706c694521ac74e56badc92ae829c",
    "models.py": "71e81088f4765511749df36afd13c215a4c9aad1b3310f423fad0df6278179c0",
    "run_g4b.py": "91b549596b5f1a46ddf86250bd564447386f8a215083e1bb22013bb367e21564",
    "verify_g4b.py": "97b71f200ee3e24a378619a586966d2baaa0ea5d0af0ff7e1ada8e5c3dfdbb3c",
}
APPROVED_TRACKED_PATHS = {
    "reproduction/siren_a/audit/g4b_model_summary.csv",
    "reproduction/siren_a/audit/g4b_prediction_summary.csv",
    "reproduction/siren_a/docs/G4B_IMPLEMENTATION_PLAN.md",
    "reproduction/siren_a/docs/G4B_TRAIN_REPORT.md",
    "reproduction/siren_a/docs/G4B_TRAIN_SPEC.md",
    *{
        f"reproduction/siren_a/g4b/{name}"
        for name in (
            "__init__.py", "config.py", "manifest.py", "models.py", "run_g4b.py",
            "verify_g4b.py", "test_manifest.py", "test_models.py", "test_run_g4b.py",
            "test_seed_schedule.py", "test_verify_g4b.py",
        )
    },
}


@dataclass(frozen=True)
class G4BConfig:
    repo: Path
    private_root: Path
    g4a_root: Path
    public_model_csv: Path
    public_prediction_csv: Path


def default_config() -> G4BConfig:
    repo = Path(__file__).resolve().parents[3]
    return G4BConfig(
        repo=repo,
        private_root=PRIVATE_ROOT,
        g4a_root=G4A_ROOT,
        public_model_csv=repo / "reproduction/siren_a/audit/g4b_model_summary.csv",
        public_prediction_csv=repo / "reproduction/siren_a/audit/g4b_prediction_summary.csv",
    )


def smoke_model_schedule():
    return tuple(item for item in new_model_identities() if item[1] == 43 and item[2] == "V1")


def smoke_prediction_schedule():
    return tuple(item for item in new_prediction_identities() if item[4] == 43 and item[2] == "V1")


def full_model_schedule():
    return new_model_identities()


def full_prediction_schedule():
    return new_prediction_identities()


def sample_path(root: Path, seed: int, site: str) -> Path:
    return Path(root) / "seeds" / f"seed{int(seed)}" / "samples" / site


def query_path(root: Path, seed: int, site: str) -> Path:
    return Path(root) / "seeds" / f"seed{int(seed)}" / "queries" / site


def model_path(root: Path, seed: int, site: str, view: str, model_type: str) -> Path:
    return Path(root) / "seeds" / f"seed{int(seed)}" / "models" / site / view / model_type


def prediction_path(root: Path, seed: int, holdout: str, protocol: str,
                    view: str, model_type: str) -> Path:
    return (Path(root) / "seeds" / f"seed{int(seed)}" / "predictions"
            / f"holdout_{holdout}" / protocol / view / model_type)


def _publish_building(building: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(target)
    try:
        for directory in sorted((path for path in building.rglob("*") if path.is_dir()),
                                key=lambda path: len(path.parts)):
            (target / directory.relative_to(building)).mkdir()
        files = sorted((path for path in building.rglob("*") if path.is_file()),
                       key=lambda path: (path.name == "complete.json", path.as_posix()))
        for source in files:
            os.link(source, target / source.relative_to(building))
    except Exception:
        raise
    shutil.rmtree(building)


def build_unit(target: Path, schema: str, identity: Mapping[str, object], writer) -> str:
    target = Path(target)
    if target.exists() or target.is_symlink():
        verify_unit(target, schema, identity)
        return "SKIP_VERIFIED"
    building = target.parent / f"{target.name}.building"
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


def check_shared_training(left, right) -> bool:
    if left.train_site != right.train_site:
        raise ValueError("shared training site mismatch")
    for name in ("X_train", "train_valid_mask", "y_train", "age_train",
                 "train_age_groups", "train_indices"):
        if not np.array_equal(getattr(left, name), getattr(right, name), equal_nan=True):
            raise ValueError(f"shared training mismatch: {name}")
    if tuple(left.feature_ids) != tuple(right.feature_ids):
        raise ValueError("shared training feature order mismatch")
    return True


def _environment_identity() -> Mapping[str, object]:
    if Path(sys.executable).resolve() != ENV_PYTHON.resolve():
        raise ValueError("G4-B must use frozen sleepfm_env Python")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ValueError("PYTHONNOUSERSITE must equal 1")
    if any(os.environ.get(key) != value for key, value in THREAD_ENV.items()):
        raise ValueError("G4-B thread environment drift")
    versions = {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "joblib": joblib.__version__, "lightgbm": lightgbm.__version__,
    }
    expected = {"python": "3.10.20", "numpy": "2.2.6", "scipy": "1.15.3",
                "sklearn": "1.7.2", "joblib": "1.5.3", "lightgbm": "4.6.0"}
    if versions != expected:
        raise ValueError(f"G4-B environment drift: {versions}")
    prefixes = {name: str(Path(module.__file__).resolve()) for name, module in {
        "numpy": np, "scipy": scipy, "sklearn": sklearn,
        "joblib": joblib, "lightgbm": lightgbm,
    }.items()}
    environment_root = ENV_PYTHON.parent.parent.resolve()
    if any(environment_root not in Path(path).parents for path in prefixes.values()):
        raise ValueError("a core package was imported outside the frozen environment")
    return {"python_executable": str(ENV_PYTHON), "versions": versions,
            "imports": prefixes, "thread_env": dict(THREAD_ENV)}


def _code_hashes() -> Mapping[str, str]:
    directory = Path(__file__).parent
    names = ("config.py", "models.py", "run_g4b.py", "verify_g4b.py", "manifest.py")
    hashes = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"G4-B code file missing: {name}")
        hashes[name] = sha256_file(path)
    return hashes


def _git_preflight(repo: Path) -> Mapping[str, str]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    top = Path(git("rev-parse", "--show-toplevel")).resolve()
    if top != Path(repo).resolve():
        raise ValueError("G4-B Git worktree mismatch")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "research/siren-a-stagepool-v1":
        raise ValueError("G4-B Git branch mismatch")
    origin = git("remote", "get-url", "origin")
    if "ysj-EDG/physionet-challenge-2026-baseline" not in origin:
        raise ValueError("G4-B Git origin mismatch")
    head = git("rev-parse", "HEAD")
    if head != APPROVED_PARENT_SHA:
        parent = git("rev-parse", "HEAD^")
        if parent != APPROVED_PARENT_SHA:
            raise ValueError("G4-B approved parent SHA mismatch")
    remote_line = git("ls-remote", "--heads", "origin", branch)
    fields = remote_line.split()
    if len(fields) != 2 or fields[1] != f"refs/heads/{branch}":
        raise ValueError("G4-B remote branch is missing")
    remote = fields[0]
    if head != remote:
        raise ValueError("G4-B local/remote tracking SHA mismatch")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    unexpected = []
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in APPROVED_TRACKED_PATHS:
            unexpected.append(path)
    if unexpected:
        raise ValueError(f"G4-B worktree contains out-of-scope changes: {unexpected}")
    return {"branch": branch, "head": head, "remote": remote, "origin": origin}


def verify_g4a_source_code(
    source_bundle: Mapping[str, object],
    frozen_identity: Mapping[str, object],
) -> bool:
    current = source_bundle.get("identity")
    if not isinstance(current, Mapping):
        raise ValueError("G4-A current source identity missing")
    current_code = current.get("code_sha256")
    frozen_code = frozen_identity.get("code_sha256")
    if not isinstance(current_code, Mapping) or current_code != frozen_code:
        raise ValueError("G4-A executable source code drift")
    return True


def _select_execution_identity(root: Path, current: Mapping[str, object]):
    if not root.exists():
        return current
    path = root / "run_identity.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("existing G4-B run identity missing")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved == current:
        return saved
    if (_identity_sha(saved) != LEGACY_RUN_IDENTITY_SHA256
            or saved.get("code_sha256") != LEGACY_CODE_SHA256):
        raise ValueError("existing G4-B execution code identity is not approved")
    saved_contract = {key: value for key, value in saved.items() if key != "code_sha256"}
    current_contract = {key: value for key, value in current.items() if key != "code_sha256"}
    if saved_contract != current_contract:
        raise ValueError("existing G4-B execution contract drift")
    return saved


def _load_g4a_identity(config: G4BConfig) -> Mapping[str, object]:
    path = config.g4a_root / "run_identity.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("G4-A run identity missing")
    identity = json.loads(path.read_text(encoding="utf-8"))
    if _identity_sha(identity) != G4A_RUN_IDENTITY_SHA256:
        raise ValueError("G4-A run identity drift")
    verify_unit(config.g4a_root, "g4a_seed42_v1",
                {"run_identity_sha256": G4A_RUN_IDENTITY_SHA256})
    return identity


def preflight(config: G4BConfig) -> Mapping[str, object]:
    git_identity = _git_preflight(config.repo)
    environment = _environment_identity()
    g4a_config = g4a_default_config()
    if g4a_config.private_root.resolve() != config.g4a_root.resolve():
        raise ValueError("G4-A source root mismatch")
    source_bundle = _source_identity(g4a_config)
    g4a_identity = _load_g4a_identity(config)
    verify_g4a_source_code(source_bundle, g4a_identity)
    expected = {
        "g2b1_cache_fingerprint": G2_FINGERPRINT,
        "g3a_artifact_fingerprint": G3_FINGERPRINT,
        "g3b_run_identity_sha256": G3B_RUN_IDENTITY_SHA256,
        "g3b_complete_sha256": G3B_COMPLETE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "spec_sha256": SPEC_SHA256,
        "demographics_sha256": DEMOGRAPHICS_SHA256,
    }
    for key, value in expected.items():
        if g4a_identity.get(key) != value:
            raise ValueError(f"G4-A upstream identity drift: {key}")
    for view in VIEWS:
        s1 = _load(source_bundle, "I0002", "transductive", view)
        for holdout, protocol in (("I0002", "inductive"),
                                  ("I0006", "transductive"),
                                  ("I0006", "inductive")):
            check_shared_training(s1, _load(source_bundle, holdout, protocol, view))
        check_shared_training(_load(source_bundle, "S0001", "transductive", view),
                              _load(source_bundle, "S0001", "inductive", view))
    for site in TRAIN_SITES:
        fold = _load(source_bundle, CANONICAL_HOLDOUT[site], "transductive", "V1")
        if fold.train_site != site or len(fold.X_train) != EXPECTED_TRAIN[site]:
            raise ValueError("canonical training fold mismatch")
    root = config.private_root
    if root.is_symlink():
        raise ValueError("private root may not be a symlink")
    if root.exists() and (not root.is_dir() or not (root / "run_identity.json").is_file()):
        raise ValueError("existing G4-B private root is invalid")
    identity = {
        "schema": SCHEMA,
        "seeds": list(NEW_SEEDS),
        **expected,
        "g4a_run_identity_sha256": G4A_RUN_IDENTITY_SHA256,
        "g4a_complete_sha256": sha256_file(config.g4a_root / "complete.json"),
        "environment": environment,
        "code_sha256": _code_hashes(),
        "sampler_version": SAMPLER_VERSION,
    }
    identity = _select_execution_identity(root, identity)
    return {"identity": identity, "source_bundle": source_bundle,
            "g4a_identity": g4a_identity, "git_identity": git_identity}


def _initialize_root(config: G4BConfig, identity: Mapping[str, object]) -> str:
    run_sha = _identity_sha(identity)
    root = config.private_root
    if root.exists():
        saved = json.loads((root / "run_identity.json").read_text(encoding="utf-8"))
        if saved != identity:
            raise ValueError("G4-B private run identity drift")
        return run_sha
    root.mkdir(parents=True)
    atomic_json_no_clobber(root / "run_identity.json", identity)
    atomic_json_no_clobber(root / "config_frozen.json", {
        "schema": SCHEMA, "seeds": list(NEW_SEEDS), "models": list(MODEL_TYPES),
        "views": list(VIEWS), "protocols": list(PROTOCOLS),
        "run_identity_sha256": run_sha,
    })
    return run_sha


def _save_sample(config: G4BConfig, run_sha: str, bundle, seed: int, site: str):
    fold = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", "V1")
    sample = sample_training_rows(site, seed, fold.train_age_groups, fold.y_train,
                                  fold.train_indices)
    if sample.target_n != EXPECTED_SELECTED[site]:
        raise ValueError("sampling target mismatch")
    identity = {"run_identity_sha256": run_sha, "train_site": site, "seed": seed,
                "sample_sha256": sample.index_sha256, "schema": "g4b_sample_v1"}

    def writer(path):
        atomic_npz_no_clobber(path / "selected_indices.npz",
                              local_positions=sample.local_positions,
                              global_indices=sample.global_indices)
        selected = fold.y_train[sample.local_positions]
        strata = [{"age_group": key[0], "label": key[1],
                   "original": sample.stratum_sizes[key], "quota": sample.quotas[key]}
                  for key in sorted(sample.quotas)]
        atomic_json_no_clobber(path / "strata_audit.json", {
            "train_site": site, "seed": seed, "train_total": len(fold.y_train),
            "target": sample.target_n, "actual": len(sample.local_positions),
            "positive": int(np.sum(selected == 1)), "negative": int(np.sum(selected == 0)),
            "strata": strata, "index_sha256": sample.index_sha256,
            "sampler_version": SAMPLER_VERSION, "numpy_version": np.__version__,
        })

    action = build_unit(sample_path(config.private_root, seed, site), "g4b_sample_v1",
                        identity, writer)
    with np.load(sample_path(config.private_root, seed, site) / "selected_indices.npz",
                 allow_pickle=False) as archive:
        if (not np.array_equal(archive["local_positions"], sample.local_positions)
                or not np.array_equal(archive["global_indices"], sample.global_indices)):
            raise ValueError("saved sampling indices differ from deterministic replay")
    return sample, action


def _save_query(config: G4BConfig, run_sha: str, bundle, seed: int, site: str, sample):
    fold = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", "V1")
    local = sample.local_positions
    ages = np.asarray(fold.age_train[local], dtype=np.float64)
    labels = np.asarray(fold.y_train[local], dtype=np.int8)
    query = build_queries(ages, labels, local, sample.global_indices)
    coverage = audit_ac_pair_coverage(ages, labels, fold.train_age_groups[local])
    identity = {"run_identity_sha256": run_sha, "train_site": site, "seed": seed,
                "sample_sha256": sample.index_sha256, "query_sha256": query.fingerprint,
                "schema": "g4b_query_v1"}

    def writer(path):
        atomic_npz_no_clobber(
            path / "query.npz", permutation=query.permutation,
            inverse_permutation=query.inverse_permutation,
            sorted_local_positions=query.sorted_local_positions,
            sorted_global_indices=query.sorted_global_indices,
            sorted_group_ids=query.sorted_group_ids, sorted_labels=query.sorted_labels,
            group_ids=query.group_ids, group_sizes=query.group_sizes,
            group_positive=query.group_positive, group_negative=query.group_negative,
        )
        atomic_json_no_clobber(path / "query_audit.json", {
            "train_site": site, "seed": seed, "query_count": len(query.group_sizes),
            "maximum_group_size": query.truncation_level,
            "truncation_level": query.truncation_level,
            "query_fingerprint": query.fingerprint,
            "sample_sha256": sample.index_sha256, **coverage,
        })

    target = query_path(config.private_root, seed, site)
    action = build_unit(target, "g4b_query_v1", identity, writer)
    verify_query_artifact(target, "g4b_query_v1", identity, ages, labels, local,
                          sample.global_indices, sample_sha256=sample.index_sha256)
    return query, coverage, action


def _checkpoint_files(path: Path, model_type: str) -> Mapping[str, str]:
    model_name = "model.txt" if model_type == "lambdarank" else "model.joblib"
    return {model_name: sha256_file(path / model_name),
            "adapter_state.npz": sha256_file(path / "adapter_state.npz")}


def _feature_sha(feature_ids) -> str:
    return hashlib.sha256("\n".join(feature_ids).encode("utf-8")).hexdigest()


def _model_identity(run_sha: str, seed: int, site: str, view: str, model_type: str,
                    sample, query, feature_sha: str) -> Mapping[str, object]:
    return {"run_identity_sha256": run_sha, "train_site": site, "seed": seed,
            "view": view, "model_type": model_type, "sample_sha256": sample.index_sha256,
            "query_sha256": query.fingerprint, "feature_ids_sha256": feature_sha,
            "schema": "g4b_model_v1"}


def _verify_loaded_parameters(model_type: str, model: object,
                              expected: Mapping[str, object]) -> bool:
    if model_type in {"lr", "hgb"}:
        observed = model.get_params(deep=False)
        aliases = {key: key for key in expected}
    else:
        observed = dict(model.params)
        aliases = {
            "objective": "objective", "n_estimators": "num_iterations",
            "learning_rate": "learning_rate", "num_leaves": "num_leaves",
            "max_depth": "max_depth", "min_child_samples": "min_data_in_leaf",
            "reg_lambda": "lambda_l2", "colsample_bytree": "feature_fraction",
            "subsample": "bagging_fraction", "subsample_freq": "bagging_freq",
            "label_gain": "label_gain", "lambdarank_norm": "lambdarank_norm",
            "random_state": "seed", "n_jobs": "num_threads", "verbosity": "verbosity",
            "deterministic": "deterministic", "force_col_wise": "force_col_wise",
            "lambdarank_truncation_level": "lambdarank_truncation_level",
        }
    for key, value in expected.items():
        observed_value = observed.get(aliases[key])
        if observed_value != value:
            raise ValueError(f"loaded model parameter mismatch: {key}={observed_value!r}")
    return True


def _verify_model_unit(config: G4BConfig, run_sha: str, bundle, seed: int, site: str,
                       view: str, model_type: str, sample, query):
    fold = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", view)
    X = np.asarray(fold.X_train[sample.local_positions], dtype=np.float64)
    y = np.asarray(fold.y_train[sample.local_positions], dtype=np.int8)
    feature_sha = _feature_sha(fold.feature_ids)
    identity = _model_identity(run_sha, seed, site, view, model_type, sample, query,
                               feature_sha)
    target = model_path(config.private_root, seed, site, view, model_type)
    verify_unit(target, "g4b_model_v1", identity)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    expected_params = parameters_for_seed(model_type, seed, query.truncation_level)
    if metadata.get("parameters") != expected_params:
        raise ValueError("saved model parameter metadata mismatch")
    files = _checkpoint_files(target, model_type)
    if metadata.get("checkpoint_files") != files:
        raise ValueError("checkpoint file hashes differ from metadata")
    loaded = load_checkpoint(target, model_type, files)
    _verify_loaded_parameters(model_type, loaded.model, expected_params)
    verify_adapter_artifact(target / "adapter_state.npz", model_type, X,
                            fold.feature_ids, sample.global_indices)
    replay = predict_scores(model_type, loaded.model, loaded.adapter, X)
    if not np.isfinite(replay).all() or np.ptp(replay) == 0:
        raise ValueError("loaded model training replay invalid")
    return metadata, loaded, replay


def _train_model(config: G4BConfig, run_sha: str, bundle, seed: int, site: str,
                 view: str, model_type: str, sample, query):
    fold = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", view)
    X = np.asarray(fold.X_train[sample.local_positions], dtype=np.float64)
    y = np.asarray(fold.y_train[sample.local_positions], dtype=np.int8)
    feature_sha = _feature_sha(fold.feature_ids)
    identity = _model_identity(run_sha, seed, site, view, model_type, sample, query,
                               feature_sha)
    target = model_path(config.private_root, seed, site, view, model_type)

    def writer(path):
        start = time.perf_counter()
        before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trained = fit_seed_model(model_type, seed, X, y, fold.feature_ids,
                                     sample.global_indices, query)
        verify_runtime_parameters(model_type, trained.model, dict(trained.parameters))
        fit_seconds = time.perf_counter() - start
        files = save_checkpoint(path, trained)
        loaded = load_checkpoint(path, model_type, files)
        _verify_loaded_parameters(model_type, loaded.model, trained.parameters)
        replay = predict_scores(model_type, loaded.model, loaded.adapter, X)
        difference = float(np.max(np.abs(replay - trained.train_score)))
        if difference > 1e-12 or np.ptp(replay) == 0:
            raise ValueError("checkpoint reload mismatch or constant training output")
        iterations = (int(np.max(np.asarray(trained.model.n_iter_)))
                      if hasattr(trained.model, "n_iter_")
                      else int(trained.model.booster_.num_trees()))
        atomic_json_no_clobber(path / "metadata.json", {
            **identity, "parameters": dict(trained.parameters), "train_n": len(X),
            "positive": int(np.sum(y == 1)), "negative": int(np.sum(y == 0)),
            "feature_count": X.shape[1], "missing_count": int(np.isnan(X).sum()),
            "fit_seconds": fit_seconds,
            "peak_rss_kib": int(max(before_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
            "iterations": iterations, "warning_count": len(caught),
            "warnings": [str(item.message) for item in caught],
            "checkpoint_files": files, "adapter_state_sha256": files["adapter_state.npz"],
            "reload_max_abs_diff": difference,
            "train_score_min": float(replay.min()), "train_score_max": float(replay.max()),
            "train_score_std": float(replay.std()),
            "score_type": "positive_probability" if model_type in {"lr", "hgb"}
                          else "raw_ranking_score",
        })

    action = build_unit(target, "g4b_model_v1", identity, writer)
    metadata, _, _ = _verify_model_unit(config, run_sha, bundle, seed, site, view,
                                        model_type, sample, query)
    return metadata, action


def _prediction_identity(run_sha: str, seed: int, holdout: str, protocol: str,
                         view: str, model_type: str, test_sha: str,
                         model_complete_sha: str, prediction_sha: str):
    return {"run_identity_sha256": run_sha, "holdout_site": holdout,
            "train_site": HOLDOUT_TO_TRAIN[holdout], "protocol": protocol,
            "view": view, "model_type": model_type, "seed": seed,
            "test_indices_sha256": test_sha,
            "model_complete_sha256": model_complete_sha,
            "prediction_sha256": prediction_sha, "schema": "g4b_prediction_v1"}


def _predict(config: G4BConfig, run_sha: str, bundle, seed: int, holdout: str,
             protocol: str, view: str, model_type: str, sample, query):
    site = HOLDOUT_TO_TRAIN[holdout]
    fold = _load(bundle, holdout, protocol, view)
    model_meta, loaded, _ = _verify_model_unit(config, run_sha, bundle, seed, site,
                                                view, model_type, sample, query)
    scores = predict_scores(model_type, loaded.model, loaded.adapter, fold.X_test)
    if np.ptp(scores) == 0:
        raise ValueError("strictly constant test prediction")
    model_root = model_path(config.private_root, seed, site, view, model_type)
    files = _checkpoint_files(model_root, model_type)
    test_sha = array_sha256(fold.test_indices)
    prediction_sha = array_sha256(scores)
    identity = _prediction_identity(run_sha, seed, holdout, protocol, view, model_type,
                                    test_sha, sha256_file(model_root / "complete.json"),
                                    prediction_sha)
    target = prediction_path(config.private_root, seed, holdout, protocol, view, model_type)

    def writer(path):
        atomic_npz_no_clobber(path / "prediction.npz", scores=scores,
                              test_global_indices=fold.test_indices)
        atomic_json_no_clobber(path / "metadata.json", {
            **identity, "test_count": len(scores), "feature_count": len(fold.feature_ids),
            "feature_ids_sha256": model_meta["feature_ids_sha256"],
            "adapter_state_sha256": model_meta["adapter_state_sha256"],
            "checkpoint_sha256": files["model.txt" if model_type == "lambdarank"
                                       else "model.joblib"],
            "score_type": model_meta["score_type"], "finite": True,
            "nonconstant": True, "score_min": float(scores.min()),
            "score_max": float(scores.max()), "score_std": float(scores.std()),
        })

    action = build_unit(target, "g4b_prediction_v1", identity, writer)
    _verify_prediction_unit(config, run_sha, bundle, seed, holdout, protocol, view,
                            model_type, sample, query)
    return json.loads((target / "metadata.json").read_text(encoding="utf-8")), action


def _verify_prediction_unit(config: G4BConfig, run_sha: str, bundle, seed: int,
                            holdout: str, protocol: str, view: str, model_type: str,
                            sample, query):
    site = HOLDOUT_TO_TRAIN[holdout]
    fold = _load(bundle, holdout, protocol, view)
    model_meta, loaded, _ = _verify_model_unit(config, run_sha, bundle, seed, site,
                                                view, model_type, sample, query)
    replay = predict_scores(model_type, loaded.model, loaded.adapter, fold.X_test)
    model_root = model_path(config.private_root, seed, site, view, model_type)
    target = prediction_path(config.private_root, seed, holdout, protocol, view, model_type)
    identity = _prediction_identity(run_sha, seed, holdout, protocol, view, model_type,
                                    array_sha256(fold.test_indices),
                                    sha256_file(model_root / "complete.json"),
                                    array_sha256(replay))
    verify_unit(target, "g4b_prediction_v1", identity)
    with np.load(target / "prediction.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"scores", "test_global_indices"}:
            raise ValueError("prediction array keys mismatch")
        saved = np.asarray(archive["scores"])
        indices = np.asarray(archive["test_global_indices"])
    if saved.dtype != np.float64 or saved.shape != replay.shape:
        raise ValueError("prediction schema mismatch")
    if not np.array_equal(indices, fold.test_indices):
        raise ValueError("prediction test-index mismatch")
    difference = float(np.max(np.abs(saved - replay)))
    if difference > 1e-12 or not np.isfinite(saved).all() or np.ptp(saved) == 0:
        raise ValueError("prediction replay mismatch")
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != model_meta["checkpoint_files"][
            "model.txt" if model_type == "lambdarank" else "model.joblib"]:
        raise ValueError("prediction checkpoint provenance mismatch")
    return metadata, difference


def _bridge_identity(config: G4BConfig, run_sha: str) -> Mapping[str, object]:
    return {"run_identity_sha256": run_sha,
            "g4a_run_identity_sha256": G4A_RUN_IDENTITY_SHA256,
            "g4a_complete_sha256": sha256_file(config.g4a_root / "complete.json"),
            "schema": "g4b_seed42_bridge_v1"}


def verify_seed42_bridge(config: G4BConfig, run_sha: str, bundle,
                         *, publish: bool = True) -> Mapping[str, object]:
    result = verify_g4a_all(g4a_default_config(), bundle=bundle)
    if result.get("models") != 30 or result.get("predictions") != 90:
        raise ValueError("G4-A closure count mismatch")
    query_rows = []
    adapter_rows = []
    for site in TRAIN_SITES:
        fold_v1 = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", "V1")
        sample = sample_training_rows(site, 42, fold_v1.train_age_groups,
                                      fold_v1.y_train, fold_v1.train_indices)
        g4a_sample = config.g4a_root / "samples" / f"{site}_seed42"
        sample_identity = {"run_identity_sha256": G4A_RUN_IDENTITY_SHA256,
                           "train_site": site, "seed": 42,
                           "sample_sha256": sample.index_sha256,
                           "schema": "g4a_sample_v1"}
        verify_unit(g4a_sample, "g4a_sample_v1", sample_identity)
        query = build_queries(fold_v1.age_train[sample.local_positions],
                              fold_v1.y_train[sample.local_positions],
                              sample.local_positions, sample.global_indices)
        query_identity = {"run_identity_sha256": G4A_RUN_IDENTITY_SHA256,
                          "train_site": site, "seed": 42,
                          "sample_sha256": sample.index_sha256,
                          "query_sha256": query.fingerprint,
                          "schema": "g4a_query_v1"}
        verify_query_artifact(
            config.g4a_root / "queries" / f"{site}_seed42", "g4a_query_v1",
            query_identity, np.asarray(fold_v1.age_train[sample.local_positions], dtype=np.float64),
            np.asarray(fold_v1.y_train[sample.local_positions], dtype=np.int8),
            sample.local_positions, sample.global_indices,
            sample_sha256=sample.index_sha256,
        )
        query_rows.append({"train_site": site, "status": "PASS",
                           "query_sha256": query.fingerprint,
                           "array_count": 10, "sample_sha256": sample.index_sha256})
        for view in VIEWS:
            fold = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", view)
            X = np.asarray(fold.X_train[sample.local_positions], dtype=np.float64)
            for model_type in MODEL_TYPES:
                path = config.g4a_root / "models" / site / "seed42" / view / model_type
                verify_adapter_artifact(path / "adapter_state.npz", model_type, X,
                                        fold.feature_ids, sample.global_indices)
                adapter_rows.append({"train_site": site, "view": view,
                                     "model_type": model_type, "status": "PASS"})
    bridge = {"status": "PASS", "models": 30, "predictions": 90,
              "queries_verified": len(query_rows), "lr_adapters_verified": 10,
              "native_adapters_verified": 20, "query_rows": query_rows,
              "adapter_rows": adapter_rows, "g4a_refit_count": 0,
              "g4a_overwrite_count": 0}
    if publish:
        target = config.private_root / "seed42_bridge"
        identity = _bridge_identity(config, run_sha)

        def writer(path):
            atomic_json_no_clobber(path / "source_identity.json", {
                **identity, "status": "PASS", "models": 30, "predictions": 90})
            atomic_json_no_clobber(path / "query_semantic_audit.json", query_rows)
            atomic_json_no_clobber(path / "adapter_semantic_audit.json", adapter_rows)

        bridge["action"] = build_unit(target, "g4b_seed42_bridge_v1", identity, writer)
    return bridge


def _seed_complete(config: G4BConfig, run_sha: str, seed: int) -> None:
    root = config.private_root / "seeds" / f"seed{seed}"
    identity = {"run_identity_sha256": run_sha, "seed": seed,
                "models": 30, "predictions": 90, "schema": "g4b_seed_v1"}
    if (root / "complete.json").exists():
        verify_unit(root, "g4b_seed_v1", identity)
    else:
        complete_unit(root, "g4b_seed_v1", identity)
        verify_unit(root, "g4b_seed_v1", identity)


def _write_public_csv(path: Path, rows) -> None:
    if not rows:
        raise ValueError("public CSV rows may not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if path.exists():
            if path.read_bytes() != temporary.read_bytes():
                raise ValueError(f"existing public CSV differs: {path.name}")
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _public_rows(config: G4BConfig):
    model_rows = []
    for site, seed, view, model_type in full_model_schedule():
        metadata = json.loads((model_path(config.private_root, seed, site, view, model_type)
                               / "metadata.json").read_text(encoding="utf-8"))
        sample = json.loads((sample_path(config.private_root, seed, site)
                             / "strata_audit.json").read_text(encoding="utf-8"))
        query = json.loads((query_path(config.private_root, seed, site)
                            / "query_audit.json").read_text(encoding="utf-8"))
        model_rows.append({
            "train_site": site, "seed": seed, "view": view, "model_type": model_type,
            "train_full_n": sample["train_total"], "selected_n": sample["actual"],
            "selected_positive": sample["positive"], "selected_negative": sample["negative"],
            "feature_count": metadata["feature_count"], "query_count": query["query_count"],
            "truncation_level": query["truncation_level"] if model_type == "lambdarank" else "",
            "fit_seconds": metadata["fit_seconds"], "iterations_or_trees": metadata["iterations"],
            "model_parameters_verified": "PASS", "sampling_verified": "PASS",
            "query_verified": "PASS", "adapter_verified": "PASS",
            "checkpoint_verified": "PASS", "model_reload_verified": "PASS", "status": "PASS",
        })
    prediction_rows = []
    for holdout, protocol, view, model_type, seed in full_prediction_schedule():
        site = HOLDOUT_TO_TRAIN[holdout]
        metadata = json.loads((prediction_path(config.private_root, seed, holdout, protocol,
                                               view, model_type) / "metadata.json").read_text(encoding="utf-8"))
        sample = json.loads((sample_path(config.private_root, seed, site)
                             / "strata_audit.json").read_text(encoding="utf-8"))
        prediction_rows.append({
            "holdout_site": holdout, "train_site": site, "protocol": protocol,
            "view": view, "model_type": model_type, "seed": seed,
            "train_selected": sample["actual"], "test_count": metadata["test_count"],
            "score_type": metadata["score_type"], "prediction_finite": "PASS",
            "prediction_nonconstant": "PASS", "prediction_replay_verified": "PASS",
            "test_indices_verified": "PASS", "model_source_verified": "PASS", "status": "PASS",
        })
    if len(model_rows) != 420 or len(prediction_rows) != 1260:
        raise ValueError("public summary row closure failed")
    return model_rows, prediction_rows


def _run(config: G4BConfig, *, smoke: bool) -> Mapping[str, object]:
    started = time.perf_counter()
    pre = preflight(config)
    run_sha = _initialize_root(config, pre["identity"])
    bundle = pre["source_bundle"]
    bridge = verify_seed42_bridge(config, run_sha, bundle, publish=True)
    seeds = (43,) if smoke else NEW_SEEDS
    model_schedule = smoke_model_schedule() if smoke else full_model_schedule()
    prediction_schedule = smoke_prediction_schedule() if smoke else full_prediction_schedule()
    samples = {}
    queries = {}
    sample_actions = {}
    query_actions = {}
    for seed in seeds:
        for site in TRAIN_SITES:
            samples[(seed, site)], sample_actions[(seed, site)] = _save_sample(
                config, run_sha, bundle, seed, site)
            queries[(seed, site)], _, query_actions[(seed, site)] = _save_query(
                config, run_sha, bundle, seed, site, samples[(seed, site)])
    model_actions = {}
    model_metadata = {}
    ordered = sorted(model_schedule, key=lambda item: (
        item[1], item[2] != "V1", TRAIN_SITES.index(item[0]),
        VIEWS.index(item[2]), MODEL_TYPES.index(item[3])))
    for site, seed, view, model_type in ordered:
        model_metadata[(site, seed, view, model_type)], model_actions[(site, seed, view, model_type)] = _train_model(
            config, run_sha, bundle, seed, site, view, model_type,
            samples[(seed, site)], queries[(seed, site)])
    prediction_actions = {}
    for holdout, protocol, view, model_type, seed in prediction_schedule:
        site = HOLDOUT_TO_TRAIN[holdout]
        _, prediction_actions[(holdout, protocol, view, model_type, seed)] = _predict(
            config, run_sha, bundle, seed, holdout, protocol, view, model_type,
            samples[(seed, site)], queries[(seed, site)])
    g5 = None
    if not smoke:
        for seed in NEW_SEEDS:
            _seed_complete(config, run_sha, seed)
        g5 = build_g5_ready(config, run_sha, build_unit)
        model_rows, prediction_rows = _public_rows(config)
        _write_public_csv(config.public_model_csv, model_rows)
        _write_public_csv(config.public_prediction_csv, prediction_rows)
    summary = {
        "status": "PASS", "mode": "smoke" if smoke else "build",
        "bridge": bridge["status"],
        "samples_built": sum(value == "BUILT_NEW" for value in sample_actions.values()),
        "samples_skipped": sum(value == "SKIP_VERIFIED" for value in sample_actions.values()),
        "queries_built": sum(value == "BUILT_NEW" for value in query_actions.values()),
        "queries_skipped": sum(value == "SKIP_VERIFIED" for value in query_actions.values()),
        "models_built": sum(value == "BUILT_NEW" for value in model_actions.values()),
        "models_skipped": sum(value == "SKIP_VERIFIED" for value in model_actions.values()),
        "predictions_built": sum(value == "BUILT_NEW" for value in prediction_actions.values()),
        "predictions_skipped": sum(value == "SKIP_VERIFIED" for value in prediction_actions.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    if g5 is not None:
        summary["g5_ready"] = g5["status"]
        report = config.private_root / "reports" / "build_summary.json"
        if not report.exists():
            atomic_json_no_clobber(report, summary)
        root_identity = {"run_identity_sha256": run_sha}
        if (config.private_root / "complete.json").exists():
            verify_unit(config.private_root, SCHEMA, root_identity)
        else:
            complete_unit(config.private_root, SCHEMA, root_identity)
            verify_unit(config.private_root, SCHEMA, root_identity)
    return summary


def verify_new(config: G4BConfig, pre=None, *, require_full: bool = True) -> Mapping[str, object]:
    pre = preflight(config) if pre is None else pre
    identity = pre["identity"]
    run_sha = _identity_sha(identity)
    saved = json.loads((config.private_root / "run_identity.json").read_text(encoding="utf-8"))
    if saved != identity:
        raise ValueError("G4-B root identity mismatch")
    bundle = pre["source_bundle"]
    bridge = verify_seed42_bridge(config, run_sha, bundle, publish=False)
    samples = {}
    queries = {}
    for seed in NEW_SEEDS:
        seed_root = config.private_root / "seeds" / f"seed{seed}"
        if not seed_root.exists():
            if require_full:
                raise ValueError(f"missing seed: {seed}")
            continue
        for site in TRAIN_SITES:
            fold = _load(bundle, CANONICAL_HOLDOUT[site], "transductive", "V1")
            sample = sample_training_rows(site, seed, fold.train_age_groups,
                                          fold.y_train, fold.train_indices)
            sample_identity = {"run_identity_sha256": run_sha, "train_site": site,
                               "seed": seed, "sample_sha256": sample.index_sha256,
                               "schema": "g4b_sample_v1"}
            verify_unit(sample_path(config.private_root, seed, site), "g4b_sample_v1",
                        sample_identity)
            query = build_queries(fold.age_train[sample.local_positions],
                                  fold.y_train[sample.local_positions],
                                  sample.local_positions, sample.global_indices)
            query_identity = {"run_identity_sha256": run_sha, "train_site": site,
                              "seed": seed, "sample_sha256": sample.index_sha256,
                              "query_sha256": query.fingerprint,
                              "schema": "g4b_query_v1"}
            verify_query_artifact(
                query_path(config.private_root, seed, site), "g4b_query_v1", query_identity,
                np.asarray(fold.age_train[sample.local_positions], dtype=np.float64),
                np.asarray(fold.y_train[sample.local_positions], dtype=np.int8),
                sample.local_positions, sample.global_indices,
                sample_sha256=sample.index_sha256,
            )
            samples[(seed, site)] = sample
            queries[(seed, site)] = query
        for site in TRAIN_SITES:
            for view in VIEWS:
                for model_type in MODEL_TYPES:
                    _verify_model_unit(config, run_sha, bundle, seed, site, view,
                                       model_type, samples[(seed, site)], queries[(seed, site)])
        for holdout in HOLDOUTS:
            site = HOLDOUT_TO_TRAIN[holdout]
            for protocol in PROTOCOLS:
                for view in VIEWS:
                    for model_type in MODEL_TYPES:
                        _verify_prediction_unit(
                            config, run_sha, bundle, seed, holdout, protocol, view,
                            model_type, samples[(seed, site)], queries[(seed, site)])
        if require_full:
            _seed_complete(config, run_sha, seed)
    found_seeds = sorted({seed for seed, _ in samples})
    if require_full and found_seeds != list(NEW_SEEDS):
        raise ValueError("G4-B seed closure mismatch")
    if require_full:
        verify_g5_ready(config, run_sha)
        verify_unit(config.private_root, SCHEMA, {"run_identity_sha256": run_sha})
    return {"status": "PASS", "bridge": bridge["status"],
            "seeds": len(found_seeds), "samples": len(samples),
            "queries": len(queries), "models": len(found_seeds) * 30,
            "predictions": len(found_seeds) * 90}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen G4-B Seeds 43-56 execution")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--build", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = default_config()
    if args.preflight_only:
        pre = preflight(config)
        result = {"status": "PASS", "run_identity_sha256": _identity_sha(pre["identity"])}
    elif args.smoke:
        result = _run(config, smoke=True)
    elif args.build or args.resume:
        result = _run(config, smoke=False)
    else:
        result = verify_new(config, require_full=True)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
