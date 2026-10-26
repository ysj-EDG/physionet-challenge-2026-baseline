"""Inductive V5-only runner for the frozen CORE_NO_CAISR20 subset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .core_no_caisr20 import ROUTE_ID, build_manifest_payload, build_no_caisr20_rows, load_caisr20_feature_ids, load_core_rows_from_manifest
from .v5_family_loo import load_v5_manifest, _sha
from .v5_family_loo_runner import (
    UnitIdentityError, _array_sha, _canonical, _file_sha, _load_model, _publish_model,
    _sample_query_bridge, _validate_prediction_unit_identity, project_fold_for_config,
    publish_npz_unit,
)

TRAIN_SITES = ("S0001", "I0006")
HOLDOUTS = ("I0002", "I0006", "S0001")
SEEDS = tuple(range(42, 57))
MODEL_TYPES = ("lr", "hgb", "lambdarank")
PROTOCOL = "inductive"


class CoreNoCAISRRunnerStop(RuntimeError):
    """Frozen route identity, checkpoint, or prediction contract failed."""


def model_identities_no_caisr20():
    return tuple((site, seed, "V5", ROUTE_ID, model) for site in TRAIN_SITES for seed in SEEDS for model in MODEL_TYPES)


def prediction_identities_no_caisr20():
    return tuple((holdout, seed, "V5", ROUTE_ID, PROTOCOL, model) for holdout in HOLDOUTS for seed in SEEDS for model in MODEL_TYPES)


def validate_no_caisr_run_contract(models, predictions):
    models = tuple(models); predictions = tuple(predictions)
    if len(models) != 90 or len(set(models)) != 90 or set(models) != set(model_identities_no_caisr20()):
        raise CoreNoCAISRRunnerStop("V5 model identity/cardinality contract mismatch")
    if len(predictions) != 135 or len(set(predictions)) != 135 or set(predictions) != set(prediction_identities_no_caisr20()):
        raise CoreNoCAISRRunnerStop("V5 prediction identity/cardinality contract mismatch")
    return {"models": 90, "predictions": 135}


def validate_prediction_route(metadata: Mapping[str, object]) -> bool:
    if metadata.get("route_id") != ROUTE_ID or metadata.get("view") != "V5":
        raise CoreNoCAISRRunnerStop("prediction is not a CORE_NO_CAISR20 V5 unit")
    return True


def assemble_final_columns(frozen_v1_v4: np.ndarray, new_v5: np.ndarray) -> np.ndarray:
    left = np.asarray(frozen_v1_v4, dtype=np.float64); right = np.asarray(new_v5, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0] or left.shape[1] != 180 or right.shape[1] != 45:
        raise CoreNoCAISRRunnerStop("final ensemble requires 180 frozen and 45 new columns")
    return np.column_stack((left, right)).astype(np.float64, copy=False)


def _subset_fold(full_fold, rows):
    return project_fold_for_config(full_fold, rows)


def run_v5_no_caisr20(*, rank_root: Path, metadata, manifest_path: Path, caisr_manifest_path: Path, source_fingerprints: Mapping[str, str], output_root: Path, old_root: Path, run_identity: Mapping[str, object]) -> dict[str, object]:
    """Train only the new V5 subset and emit only its inductive predictions."""
    from .train_inputs_v2 import load_v2_fold
    from reproduction.siren_a.g4.queries import build_queries
    from reproduction.siren_a.g4.sampler import sample_training_rows
    from reproduction.siren_a.g4b.models import fit_seed_model, verify_runtime_parameters
    from reproduction.siren_a.g4b.config import parameters_for_seed
    from reproduction.siren_a.g4.models import fit_frozen_model, predict_scores

    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    run_file = output_root / "run_identity.json"
    canonical_identity = json.loads(_canonical(run_identity).decode("utf-8"))
    if run_file.exists() and json.loads(run_file.read_text(encoding="utf-8")) != canonical_identity:
        raise CoreNoCAISRRunnerStop("run identity drift")
    if not run_file.exists():
        run_file.write_bytes(_canonical(canonical_identity) + b"\n")
    identity_sha = hashlib.sha256(_canonical(canonical_identity)).hexdigest()
    manifest = load_v5_manifest(Path(manifest_path))
    core_rows = load_core_rows_from_manifest(manifest)
    caisr_ids = load_caisr20_feature_ids(Path(caisr_manifest_path))
    retained_rows = build_no_caisr20_rows(core_rows, caisr_ids)
    manifest_payload = build_manifest_payload(core_rows, retained_rows, caisr_ids)
    feature_manifest_sha = hashlib.sha256(_canonical(manifest_payload)).hexdigest()
    fold_cache = {}
    def get_fold(holdout, protocol):
        key=(holdout, protocol)
        if key not in fold_cache:
            full = load_v2_fold(Path(rank_root), holdout, protocol, "V5", metadata, Path(manifest_path), source_fingerprints)
            fold_cache[key] = _subset_fold(full, retained_rows)
        return fold_cache[key]
    models_built = models_skipped = predictions_built = predictions_skipped = sample_matches = 0
    replay_max = 0.0
    for train_site, representative in (("S0001", "I0002"), ("I0006", "S0001")):
        fold = get_fold(representative, "transductive")
        for seed in SEEDS:
            sample = sample_training_rows(train_site, seed, fold.train_age_groups, fold.y_train, fold.train_indices)
            query = build_queries(fold.age_train[sample.local_positions], fold.y_train[sample.local_positions], sample.local_positions, sample.global_indices)
            prepared = type("Prepared", (), {"fold": fold, "seed": seed, "sample": sample, "query": query})()
            _sample_query_bridge(prepared, Path(old_root)); sample_matches += 1
            for model_type in MODEL_TYPES:
                model_path = output_root / "models" / train_site / f"seed{seed}" / model_type
                metadata_model = {"route_id": ROUTE_ID, "config_id": ROUTE_ID, "train_site": train_site, "seed": int(seed), "view":"V5", "model_type": model_type, "feature_count": len(retained_rows), "feature_ids_sha256": _sha([row.feature_id for row in retained_rows]), "feature_manifest_sha256": feature_manifest_sha, "sample_sha256": sample.index_sha256, "sample_identity_sha256": sample.index_sha256, "query_sha256": query.fingerprint, "run_identity_sha256": identity_sha, "fit_scope":"seed_subset", "rank256_rebuild_count":0}
                if model_path.exists():
                    status = _publish_model(model_path, None, metadata_model); models_skipped += 1
                else:
                    if seed == 42:
                        trained = fit_frozen_model(model_type, fold.X_train[sample.local_positions], fold.y_train[sample.local_positions], fold.feature_ids, sample.global_indices, query)
                    else:
                        trained = fit_seed_model(model_type, seed, fold.X_train[sample.local_positions], fold.y_train[sample.local_positions], fold.feature_ids, sample.global_indices, query)
                        verify_runtime_parameters(model_type, trained.model, parameters_for_seed(model_type, seed, query.truncation_level))
                    from .v5_family_loo_runner import _publish_model as publish_model
                    import tempfile, shutil
                    replay_dir = Path(tempfile.mkdtemp(prefix="core-no-caisr-replay-", dir=str(output_root)))
                    try:
                        from reproduction.siren_a.g4.models import save_checkpoint, load_checkpoint
                        files = save_checkpoint(replay_dir, trained)
                        reloaded = load_checkpoint(replay_dir, model_type, files)
                        replay = float(np.max(np.abs(predict_scores(model_type, reloaded.model, reloaded.adapter, fold.X_train[sample.local_positions]) - trained.train_score)))
                        replay_max = max(replay_max, replay)
                    finally:
                        shutil.rmtree(replay_dir, ignore_errors=True)
                    metadata_model.update({"parameters":dict(trained.parameters), "score_type":"raw_ranking_score" if model_type == "lambdarank" else "positive_probability", "reload_max_abs_diff": replay})
                    publish_model(model_path, trained, metadata_model); models_built += 1
    # All models must be complete before any prediction is built.
    for holdout in HOLDOUTS:
        train_site = {"I0002":"S0001", "I0006":"S0001", "S0001":"I0006"}[holdout]
        fold = get_fold(holdout, PROTOCOL)
        for seed in SEEDS:
            for model_type in MODEL_TYPES:
                model_path = output_root / "models" / train_site / f"seed{seed}" / model_type
                prediction_path = output_root / "predictions" / f"seed{seed}" / f"holdout_{holdout}" / PROTOCOL / model_type
                expected = {"route_id": ROUTE_ID, "config_id": ROUTE_ID, "holdout_site": holdout, "train_site": train_site, "protocol": PROTOCOL, "seed": int(seed), "view":"V5", "model_type": model_type, "feature_count": len(retained_rows), "feature_ids_sha256": _sha([row.feature_id for row in retained_rows]), "feature_manifest_sha256": feature_manifest_sha, "model_complete_sha256": _file_sha(model_path / "complete.json"), "test_indices_sha256": _array_sha(np.asarray(fold.test_indices, dtype=np.int64)), "score_type":"raw_ranking_score" if model_type == "lambdarank" else "positive_probability", "run_identity_sha256": identity_sha, "rank256_rebuild_count":0}
                if prediction_path.exists():
                    _validate_prediction_unit_identity(prediction_path, expected); predictions_skipped += 1; continue
                model, _ = _load_model(model_path, model_type)
                scores = np.asarray(predict_scores(model_type, model.model, model.adapter, fold.X_test), dtype=np.float64)
                if not np.isfinite(scores).all() or np.ptp(scores) == 0:
                    raise CoreNoCAISRRunnerStop("new V5 prediction is nonfinite or constant")
                payload = {"test_global_indices": np.asarray(fold.test_indices, dtype=np.int64), "scores": scores}
                metadata_pred = dict(expected); metadata_pred["prediction_sha256"] = _array_sha(scores)
                publish_npz_unit(prediction_path, payload, metadata_pred); predictions_built += 1
    validate_no_caisr_run_contract(model_identities_no_caisr20(), prediction_identities_no_caisr20())
    summary = {"schema":"featurelab_core_no_caisr20_runner_summary_v1","route_id":ROUTE_ID,"protocol":PROTOCOL,"models_built":models_built,"models_skipped":models_skipped,"predictions_built":predictions_built,"predictions_skipped":predictions_skipped,"sample_identity_matches":sample_matches,"sample_identity_expected":30,"new_v5_model_count":90,"new_v5_prediction_count":135,"reload_max_abs_diff":float(replay_max),"rank256_rebuild_count":0,"v1_v4_refit_count":0,"v1_v4_prediction_recompute_count":0,"feature_manifest_sha256":feature_manifest_sha}
    (output_root / "summary.json").write_bytes(_canonical(summary) + b"\n")
    return summary


__all__ = ["CoreNoCAISRRunnerStop", "HOLDOUTS", "MODEL_TYPES", "PROTOCOL", "ROUTE_ID", "SEEDS", "TRAIN_SITES", "assemble_final_columns", "model_identities_no_caisr20", "prediction_identities_no_caisr20", "run_v5_no_caisr20", "validate_no_caisr_run_contract", "validate_prediction_route"]
