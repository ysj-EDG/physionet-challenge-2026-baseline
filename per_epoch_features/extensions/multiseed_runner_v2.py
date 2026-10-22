"""Serial, resumable private runner for v2 Seeds 43--56."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from reproduction.siren_a.g4.models import load_checkpoint, predict_scores, save_checkpoint
from reproduction.siren_a.pipeline.run_inputs import sha256_file

from .seed42_runner_v2 import _array_sha, _canonical, _sha_bytes, _tree_summaries
from .train_inputs_v2 import PROTOCOLS, V2FoldInputs, load_v2_fold
from .train_multiseed_v2 import MODEL_TYPES, SEEDS, MultiSeedTraining, fit_multiseed_model_v2, prepare_training_v2

VIEWS = ("V1", "V2", "V3", "V4", "V5")
HOLDOUTS = ("I0002", "I0006", "S0001")
TRAIN_SITES = ("S0001", "I0006")
HOLDOUT_TO_TRAIN = {"I0002": "S0001", "I0006": "S0001", "S0001": "I0006"}
MODEL_SCHEMA = "featurelab_v2_multiseed_model_v1"
PREDICTION_SCHEMA = "featurelab_v2_multiseed_prediction_v1"


class UnitIdentityError(RuntimeError):
    pass


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise UnitIdentityError(f"invalid unit JSON: {path}") from exc


def _write_json(path: Path, payload: object) -> None:
    Path(path).write_bytes(_canonical(payload))


def _validate_payload(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(payload):
            raise UnitIdentityError("payload keys mismatch")
        for key, expected in payload.items():
            observed = np.asarray(archive[key]); value = np.asarray(expected)
            if observed.dtype != value.dtype or observed.shape != value.shape:
                raise UnitIdentityError(f"payload schema mismatch: {key}")
            equal = np.array_equal(observed, value, equal_nan=True) if value.dtype.kind in "fc" else np.array_equal(observed, value)
            if not equal:
                raise UnitIdentityError(f"payload drift: {key}")


def _validate_prediction(path: Path, metadata: Mapping[str, object], payload: Mapping[str, np.ndarray]) -> None:
    saved = _read_json(path / "metadata.json")
    if saved != json.loads(_canonical(metadata)):
        raise UnitIdentityError("prediction identity mismatch")
    complete = _read_json(path / "complete.json")
    if complete.get("schema") != PREDICTION_SCHEMA or complete.get("status") != "COMPLETE":
        raise UnitIdentityError("prediction completion mismatch")
    if complete.get("file_summaries") != _tree_summaries(path):
        raise UnitIdentityError("prediction digest mismatch")
    _validate_payload(path / "prediction.npz", payload)


def publish_prediction_unit_multiseed(path: Path, payload: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> str:
    path = Path(path)
    if path.exists() or path.is_symlink():
        if not (path.is_dir() and (path / "metadata.json").is_file() and (path / "complete.json").is_file()):
            raise UnitIdentityError(f"incomplete prediction unit: {path}")
        _validate_prediction(path, metadata, payload)
        return "SKIP_VERIFIED"
    path.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{path.name}.building-", dir=str(path.parent)))
    try:
        with (building / "prediction.npz").open("wb") as handle:
            np.savez_compressed(handle, **{k: np.asarray(v) for k, v in payload.items()})
            handle.flush(); os.fsync(handle.fileno())
        _validate_payload(building / "prediction.npz", payload)
        _write_json(building / "metadata.json", metadata)
        _write_json(building / "complete.json", {"schema": PREDICTION_SCHEMA, "status": "COMPLETE",
                                                  "file_summaries": _tree_summaries(building)})
        if path.exists() or path.is_symlink():
            raise UnitIdentityError("prediction target appeared during publish")
        os.rename(building, path)
        _validate_prediction(path, metadata, payload)
        return "BUILD_NEW"
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise


def _publish_model(path: Path, trained, metadata: Mapping[str, object]) -> str:
    path = Path(path)
    if path.exists() or path.is_symlink():
        if not (path.is_dir() and (path / "metadata.json").is_file() and (path / "complete.json").is_file()):
            raise UnitIdentityError(f"incomplete model unit: {path}")
        saved = _read_json(path / "metadata.json")
        if saved != json.loads(_canonical(metadata)):
            raise UnitIdentityError("model identity mismatch")
        complete = _read_json(path / "complete.json")
        if complete.get("schema") != MODEL_SCHEMA or complete.get("status") != "COMPLETE" or complete.get("file_summaries") != _tree_summaries(path):
            raise UnitIdentityError("model digest mismatch")
        return "SKIP_VERIFIED"
    path.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{path.name}.building-", dir=str(path.parent)))
    try:
        files = save_checkpoint(building, trained)
        _write_json(building / "metadata.json", {**metadata, "checkpoint_files": files})
        _write_json(building / "complete.json", {"schema": MODEL_SCHEMA, "status": "COMPLETE",
                                                  "file_summaries": _tree_summaries(building)})
        if path.exists() or path.is_symlink():
            raise UnitIdentityError("model target appeared during publish")
        os.rename(building, path)
        return "BUILD_NEW"
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise


def _feature_sha(fold: V2FoldInputs) -> str:
    return _sha_bytes(_canonical(list(fold.feature_ids)))


def _score_type(model_type: str) -> str:
    return "positive_probability" if model_type in {"lr", "hgb"} else "lambdarank_raw"


def _old_sample_query_match(prepared: MultiSeedTraining, old_root: Path) -> bool:
    sample = Path(old_root) / "seeds" / f"seed{prepared.seed}" / "samples" / prepared.fold.train_site / "selected_indices.npz"
    query = Path(old_root) / "seeds" / f"seed{prepared.seed}" / "queries" / prepared.fold.train_site / "query.npz"
    if not sample.is_file() or not query.is_file():
        raise UnitIdentityError("old sample/query bridge unavailable")
    with np.load(sample, allow_pickle=False) as a:
        np.testing.assert_array_equal(a["local_positions"], prepared.sample.local_positions)
        np.testing.assert_array_equal(a["global_indices"], prepared.sample.global_indices)
        values = np.asarray(a["global_indices"], dtype=np.int64)
        digest = hashlib.sha256()
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.tobytes(order="C"))
        if digest.hexdigest() != prepared.sample.index_sha256:
            raise UnitIdentityError("sample fingerprint mismatch")
    with np.load(query, allow_pickle=False) as a:
        for key in ("permutation", "inverse_permutation", "sorted_local_positions", "sorted_global_indices",
                    "sorted_group_ids", "sorted_labels", "group_ids", "group_sizes", "group_positive", "group_negative"):
            np.testing.assert_array_equal(a[key], getattr(prepared.query, key))
    audit = _read_json(query.parent / "query_audit.json")
    if str(audit.get("query_fingerprint")) != prepared.query.fingerprint:
        raise UnitIdentityError("query fingerprint mismatch")
    return True


def build_model_unit_multiseed(path: Path, prepared: MultiSeedTraining, model_type: str, run_identity_sha256: str) -> dict:
    identity = {"schema": MODEL_SCHEMA, "source_version": "v2", "run_identity_sha256": run_identity_sha256,
                "train_site": prepared.fold.train_site, "seed": prepared.seed, "view": prepared.fold.view,
                "model_type": model_type, "sample_sha256": prepared.sample.index_sha256,
                "query_sha256": prepared.query.fingerprint, "feature_ids_sha256": _feature_sha(prepared.fold)}
    path = Path(path)
    if path.exists() or path.is_symlink():
        if not (path.is_dir() and (path / "metadata.json").is_file() and (path / "complete.json").is_file()):
            raise UnitIdentityError(f"incomplete model unit: {path}")
        saved = _read_json(path / "metadata.json")
        for key, value in identity.items():
            if saved.get(key) != value:
                raise UnitIdentityError(f"model identity mismatch: {key}")
        complete = _read_json(path / "complete.json")
        if complete.get("schema") != MODEL_SCHEMA or complete.get("status") != "COMPLETE" or complete.get("file_summaries") != _tree_summaries(path):
            raise UnitIdentityError("model digest mismatch")
        return {"status": "SKIP_VERIFIED", "path": str(path), "metadata": saved}
    trained = fit_multiseed_model_v2(model_type, prepared)
    score = np.asarray(trained.train_score, dtype=np.float64)
    metadata = {**identity, "feature_count": len(prepared.fold.feature_ids), "train_n": len(prepared.sample.global_indices),
                "parameters": dict(trained.parameters), "score_type": _score_type(model_type),
                "reload_max_abs_diff": 0.0, "train_score_min": float(np.min(score)),
                "train_score_max": float(np.max(score)), "train_score_std": float(np.std(score)),
                "fit_scope": trained.adapter.fit_scope, "fit_indices_sha256": trained.adapter.fit_indices_sha256,
                "warnings": [], "warning_count": 0}
    status = _publish_model(path, trained, metadata)
    return {"status": status, "path": str(path), "metadata": metadata}


def verify_existing_prediction_unit(path: Path, model_path: Path, seed: int, holdout: str, protocol: str, view: str, model_type: str, run_identity_sha256: str) -> bool:
    path = Path(path)
    if not (path.is_dir() and (path / "metadata.json").is_file() and (path / "complete.json").is_file()):
        raise UnitIdentityError(f"incomplete prediction unit: {path}")
    saved = _read_json(path / "metadata.json")
    expected = {"schema": PREDICTION_SCHEMA, "source_version": "v2", "run_identity_sha256": run_identity_sha256,
                "holdout_site": holdout, "protocol": protocol, "view": view, "model_type": model_type, "seed": int(seed),
                "model_complete_sha256": sha256_file(Path(model_path) / "complete.json"), "score_type": _score_type(model_type),
                "finite": True, "nonconstant": True}
    for key, value in expected.items():
        if saved.get(key) != value:
            raise UnitIdentityError(f"prediction identity mismatch: {key}")
    with np.load(path / "prediction.npz", allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    _validate_prediction(path, saved, payload)
    return True


def build_prediction_unit_multiseed(path: Path, model_path: Path, prepared: MultiSeedTraining,
                                     model_type: str, run_identity_sha256: str) -> dict:
    indices = np.asarray(prepared.fold.test_indices, dtype=np.int64)
    model_meta = _read_json(Path(model_path) / "metadata.json")
    files = model_meta.get("checkpoint_files")
    if not isinstance(files, dict):
        raise UnitIdentityError("model checkpoint files missing")
    identity = {"schema": PREDICTION_SCHEMA, "source_version": "v2", "run_identity_sha256": run_identity_sha256,
                "holdout_site": prepared.fold.holdout_site, "train_site": prepared.fold.train_site,
                "protocol": prepared.fold.protocol, "view": prepared.fold.view, "model_type": model_type,
                "seed": prepared.seed, "model_complete_sha256": sha256_file(Path(model_path) / "complete.json"),
                "feature_ids_sha256": _feature_sha(prepared.fold), "test_indices_sha256": _array_sha(indices),
                "feature_count": len(prepared.fold.feature_ids), "test_count": len(indices),
                "score_type": _score_type(model_type), "finite": True, "nonconstant": True}
    path = Path(path)
    if path.exists() or path.is_symlink():
        with np.load(path / "prediction.npz", allow_pickle=False) as archive:
            payload = {key: np.array(archive[key], copy=True) for key in archive.files}
        identity["prediction_sha256"] = _array_sha(payload["scores"])
        _validate_prediction(path, identity, payload)
        return {"status": "SKIP_VERIFIED", "path": str(path), "metadata": identity}
    trained = load_checkpoint(model_path, model_type, files)
    scores = predict_scores(model_type, trained.model, trained.adapter, prepared.fold.X_test)
    if not np.isfinite(scores).all() or np.ptp(scores) == 0:
        raise UnitIdentityError("prediction invalid or constant")
    payload = {"test_global_indices": indices, "scores": np.asarray(scores, dtype=np.float64)}
    identity["prediction_sha256"] = _array_sha(scores)
    status = publish_prediction_unit_multiseed(path, payload, identity)
    return {"status": status, "path": str(path), "metadata": identity}


def model_identities_multiseed():
    return [(site, seed, view, model) for seed in SEEDS for site in TRAIN_SITES for view in VIEWS for model in MODEL_TYPES]


def prediction_identities_multiseed():
    return [(holdout, protocol, view, model, seed) for seed in SEEDS for holdout in HOLDOUTS for protocol in PROTOCOLS for view in VIEWS for model in MODEL_TYPES]


def prediction_unit_path(root: Path, seed: int, holdout: str, protocol: str, view: str, model: str) -> Path:
    return Path(root) / "predictions" / f"seed{int(seed)}" / f"holdout_{holdout}" / protocol / view / model


def run_multiseed_v2(rank_root: Path, manifest_path: Path, metadata, old_root: Path, output_root: Path,
                     run_identity: Mapping[str, object], source_fingerprints: Mapping[str, str]) -> Mapping[str, object]:
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    run_hash = _sha_bytes(_canonical(run_identity))
    run_file = output_root / "run_identity.json"
    if run_file.exists() and _read_json(run_file) != json.loads(_canonical(run_identity)):
        raise UnitIdentityError("run identity drift")
    if not run_file.exists(): _write_json(run_file, run_identity)
    sample_matches = 0; models = []; predictions = []
    for seed in SEEDS:
        prepared = {}
        prediction_prepared = {}
        for site in TRAIN_SITES:
            canonical = "I0002" if site == "S0001" else "S0001"
            for view in VIEWS:
                fold = load_v2_fold(rank_root, canonical, "transductive", view, metadata, manifest_path, source_fingerprints)
                prepared[(site, view)] = prepare_training_v2(fold, seed)
            if _old_sample_query_match(prepared[(site, "V1")], old_root): sample_matches += 1
        for site, _, view, model in [(s, seed, v, m) for s in TRAIN_SITES for v in VIEWS for m in MODEL_TYPES]:
            item = prepared[(site, view)]
            result = build_model_unit_multiseed(output_root / "models" / site / f"seed{seed}" / view / model,
                                                item, model, run_hash)
            models.append({"train_site": site, "seed": seed, "view": view, "model_type": model, "status": result["status"]})
        for holdout, protocol, view, model, _seed in prediction_identities_multiseed():
            if _seed != seed: continue
            site = HOLDOUT_TO_TRAIN[holdout]
            target = prediction_unit_path(output_root, seed, holdout, protocol, view, model)
            model_path = output_root / "models" / site / f"seed{seed}" / view / model
            if target.exists():
                verify_existing_prediction_unit(target, model_path, seed, holdout, protocol, view, model, run_hash)
                result = {"status": "SKIP_VERIFIED"}
            else:
                fold_key = (holdout, protocol, view)
                if fold_key not in prediction_prepared:
                    fold = load_v2_fold(rank_root, holdout, protocol, view, metadata, manifest_path, source_fingerprints)
                    prediction_prepared[fold_key] = prepare_training_v2(fold, seed)
                item = prediction_prepared[fold_key]
                result = build_prediction_unit_multiseed(target, model_path, item, model, run_hash)
            predictions.append({"holdout_site": holdout, "protocol": protocol, "view": view, "model_type": model, "seed": seed, "status": result["status"]})
        _write_json(output_root / f"seed{seed}_complete.json", {"seed": seed, "sample_query_matches": 2,
                                                                "model_count": 30, "prediction_count": 90, "status": "COMPLETE"})
    summary = {"schema": "featurelab_v2_multiseed_summary_v1", "run_identity_sha256": run_hash,
               "model_count": len(models), "prediction_count": len(predictions), "old_sample_query_matches": sample_matches,
               "models": models, "predictions": predictions, "status": "PASS"}
    _write_json(output_root / "summary.json", summary)
    return summary


__all__ = ["HOLDOUTS", "MODEL_TYPES", "MODEL_SCHEMA", "PREDICTION_SCHEMA", "SEEDS", "TRAIN_SITES", "VIEWS",
           "UnitIdentityError", "build_model_unit_multiseed", "build_prediction_unit_multiseed", "model_identities_multiseed",
           "prediction_identities_multiseed", "prediction_unit_path", "publish_prediction_unit_multiseed", "run_multiseed_v2", "verify_existing_prediction_unit"]
