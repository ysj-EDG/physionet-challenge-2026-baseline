"""Private Seed-42 v2 model and prediction execution.

This runner is deliberately separate from the frozen G4-A entry point.  It
loads only v2 Rank256 view arrays, fits the approved estimator factory, and
publishes each model/prediction unit with a no-clobber completion record.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from reproduction.siren_a.g4.config import HGB_PARAMS, LGBM_PARAMS, LR_PARAMS
from reproduction.siren_a.g4.models import (
    TrainedModel,
    fit_frozen_model,
    load_checkpoint,
    predict_scores,
    save_checkpoint,
)
from reproduction.siren_a.pipeline.run_inputs import sha256_file

from .train_inputs_v2 import PROTOCOLS, V2FoldInputs, V2_VIEW_DIMS, load_v2_fold
from .train_seed42_v2 import (MODEL_TYPES, SEED, Seed42Training, prepare_seed42_training,
                              compare_old_sample, compare_old_query)


MODEL_SCHEMA = "featurelab_v2_seed42_model_v1"
PREDICTION_SCHEMA = "featurelab_v2_seed42_prediction_v1"
VIEWS = ("V1", "V2", "V3", "V4", "V5")
HOLDOUTS = ("I0002", "I0006", "S0001")
TRAIN_SITES = ("S0001", "I0006")
HOLDOUT_TO_TRAIN = {"I0002": "S0001", "I0006": "S0001", "S0001": "I0006"}
CANONICAL_HOLDOUT = {"S0001": "I0002", "I0006": "S0001"}


class UnitIdentityError(RuntimeError):
    """An existing private unit cannot be safely reused or replaced."""


@dataclass(frozen=True)
class UnitResult:
    status: str
    path: Path
    metadata: Mapping[str, object]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _tree_summaries(root: Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(Path(root).iterdir()) if p.is_file() and p.name != "complete.json"}


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise UnitIdentityError(f"invalid JSON: {path}") from exc


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    Path(path).write_bytes(_canonical(payload))


def _validate_payload(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    with np.load(Path(path), allow_pickle=False) as arrays:
        if set(arrays.files) != set(payload):
            raise UnitIdentityError("prediction payload keys mismatch")
        for key, expected in payload.items():
            observed = np.asarray(arrays[key]); value = np.asarray(expected)
            if observed.dtype != value.dtype or observed.shape != value.shape:
                raise UnitIdentityError(f"prediction payload schema mismatch: {key}")
            equal = np.array_equal(observed, value, equal_nan=True) if value.dtype.kind in "fc" else np.array_equal(observed, value)
            if not equal:
                raise UnitIdentityError(f"prediction payload drift: {key}")


def _validate_unit(path: Path, metadata: Mapping[str, object], payload: Mapping[str, np.ndarray], schema: str) -> bool:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise UnitIdentityError(f"unit unavailable: {path}")
    saved = _read_json(path / "metadata.json")
    if saved != json.loads(_canonical(metadata)):
        raise UnitIdentityError("unit identity mismatch")
    complete = _read_json(path / "complete.json")
    if complete.get("schema") != schema or complete.get("status") != "COMPLETE":
        raise UnitIdentityError("unit completion schema mismatch")
    if complete.get("file_summaries") != _tree_summaries(path):
        raise UnitIdentityError("unit file digest mismatch")
    _validate_payload(path / ("prediction.npz" if schema == PREDICTION_SCHEMA else "_unused.npz"), payload) if schema == PREDICTION_SCHEMA else None
    return True


def _publish_payload_unit(path: Path, payload: Mapping[str, np.ndarray], metadata: Mapping[str, object], schema: str, filename: str) -> UnitResult:
    path = Path(path)
    if path.exists() or path.is_symlink():
        if path.is_dir() and (path / "metadata.json").is_file() and (path / "complete.json").is_file():
            _validate_unit(path, metadata, payload, schema)
            return UnitResult("SKIP_VERIFIED", path, metadata)
        raise UnitIdentityError(f"existing incomplete or invalid unit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{path.name}.building-", dir=str(path.parent)))
    try:
        with (building / filename).open("wb") as handle:
            np.savez_compressed(handle, **{key: np.asarray(value) for key, value in payload.items()})
            handle.flush(); os.fsync(handle.fileno())
        _validate_payload(building / filename, payload)
        _write_json(building / "metadata.json", metadata)
        complete = {"schema": schema, "status": "COMPLETE", "file_summaries": _tree_summaries(building)}
        _write_json(building / "complete.json", complete)
        if path.exists() or path.is_symlink():
            raise UnitIdentityError(f"target appeared during publication: {path}")
        os.rename(building, path)
        _validate_unit(path, metadata, payload, schema)
        return UnitResult("BUILD_NEW", path, metadata)
    except Exception:
        if building.exists():
            for child in building.iterdir():
                if child.is_file() or child.is_symlink(): child.unlink()
            building.rmdir()
        raise


def publish_prediction_unit_v2(path: Path, payload: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> Mapping[str, object]:
    """No-clobber public helper used by tests and the private runner."""
    result = _publish_payload_unit(path, payload, metadata, PREDICTION_SCHEMA, "prediction.npz")
    return {"status": result.status, "path": result.path}


def verify_prediction_unit_v2(path: Path, metadata: Mapping[str, object]) -> bool:
    path = Path(path)
    with np.load(path / "prediction.npz", allow_pickle=False) as arrays:
        payload = {key: np.array(arrays[key], copy=True) for key in arrays.files}
    return _validate_unit(path, metadata, payload, PREDICTION_SCHEMA)


def _publish_model_unit(path: Path, trained: TrainedModel, metadata: Mapping[str, object]) -> UnitResult:
    path = Path(path)
    if path.exists() or path.is_symlink():
        if path.is_dir() and (path / "metadata.json").is_file() and (path / "complete.json").is_file():
            saved = _read_json(path / "metadata.json")
            if saved != json.loads(_canonical(metadata)):
                raise UnitIdentityError("model unit identity mismatch")
            complete = _read_json(path / "complete.json")
            if complete.get("schema") != MODEL_SCHEMA or complete.get("status") != "COMPLETE" or complete.get("file_summaries") != _tree_summaries(path):
                raise UnitIdentityError("model unit digest mismatch")
            return UnitResult("SKIP_VERIFIED", path, metadata)
        raise UnitIdentityError(f"existing incomplete model unit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{path.name}.building-", dir=str(path.parent)))
    try:
        checkpoint_files = save_checkpoint(building, trained)
        _write_json(building / "metadata.json", {**metadata, "checkpoint_files": checkpoint_files})
        complete = {"schema": MODEL_SCHEMA, "status": "COMPLETE", "file_summaries": _tree_summaries(building)}
        _write_json(building / "complete.json", complete)
        if path.exists() or path.is_symlink():
            raise UnitIdentityError(f"model target appeared during publication: {path}")
        os.rename(building, path)
        return UnitResult("BUILD_NEW", path, {**metadata, "checkpoint_files": checkpoint_files})
    except Exception:
        if building.exists():
            for child in building.iterdir():
                if child.is_file() or child.is_symlink(): child.unlink()
            building.rmdir()
        raise


def _model_parameters(model_type: str) -> Mapping[str, object]:
    if model_type == "lr": return dict(LR_PARAMS)
    if model_type == "hgb": return dict(HGB_PARAMS)
    if model_type == "lambdarank": return dict(LGBM_PARAMS)
    raise ValueError(model_type)


def model_identities_v2() -> list[tuple[str, int, str, str]]:
    return [(site, SEED, view, model) for site in TRAIN_SITES for view in VIEWS for model in MODEL_TYPES]


def prediction_identities_v2() -> list[tuple[str, str, str, str, int]]:
    return [(holdout, protocol, view, model, SEED) for holdout in HOLDOUTS for protocol in PROTOCOLS for view in VIEWS for model in MODEL_TYPES]


def _score_type(model_type: str) -> str:
    return "positive_probability" if model_type in {"lr", "hgb"} else "lambdarank_raw"


def build_model_unit_v2(path: Path, prepared: Seed42Training, model_type: str, run_identity_sha256: str) -> Mapping[str, object]:
    if model_type not in MODEL_TYPES: raise ValueError(model_type)
    sample = prepared.sample
    identity = {
        "schema": MODEL_SCHEMA, "run_identity_sha256": run_identity_sha256,
        "train_site": prepared.fold.train_site, "seed": SEED, "view": prepared.fold.view,
        "model_type": model_type, "sample_sha256": sample.index_sha256,
        "query_sha256": prepared.query.fingerprint,
        "feature_ids_sha256": _sha_bytes(_canonical(list(prepared.fold.feature_ids))),
    }
    existing = Path(path)
    if existing.exists() or existing.is_symlink():
        saved = _read_json(existing / "metadata.json")
        if any(saved.get(key) != value for key, value in identity.items()):
            raise UnitIdentityError("model unit identity mismatch")
        complete = _read_json(existing / "complete.json")
        if complete.get("schema") != MODEL_SCHEMA or complete.get("status") != "COMPLETE" or complete.get("file_summaries") != _tree_summaries(existing):
            raise UnitIdentityError("model unit digest mismatch")
        return {"status": "SKIP_VERIFIED", "path": existing, "metadata": saved}
    trained = fit_frozen_model(model_type, prepared.X_sample, prepared.y_sample, prepared.fold.feature_ids, sample.global_indices, prepared.query)
    replay = predict_scores(model_type, trained.model, trained.adapter, prepared.X_sample)
    replay_diff = float(np.max(np.abs(replay - trained.train_score)))
    if replay_diff > 1e-12: raise UnitIdentityError(f"model replay drift: {replay_diff}")
    metadata = {**identity, "feature_count": len(prepared.fold.feature_ids), "train_n": len(sample.global_indices),
                "parameters": _model_parameters(model_type), "score_type": _score_type(model_type),
                "reload_max_abs_diff": replay_diff, "train_score_min": float(np.min(trained.train_score)),
                "train_score_max": float(np.max(trained.train_score)), "train_score_std": float(np.std(trained.train_score)),
                "warnings": [], "warning_count": 0, "fit_scope": trained.adapter.fit_scope,
                "fit_indices_sha256": trained.adapter.fit_indices_sha256}
    result = _publish_model_unit(path, trained, metadata)
    return {"status": result.status, "path": result.path, "metadata": result.metadata}


def build_prediction_unit_v2(path: Path, model_path: Path, prepared: Seed42Training, model_type: str, run_identity_sha256: str, model_complete_sha256: str) -> Mapping[str, object]:
    indices = np.asarray(prepared.fold.test_indices, dtype=np.int64)
    base_identity = {"schema": PREDICTION_SCHEMA, "run_identity_sha256": run_identity_sha256,
                     "holdout_site": prepared.fold.holdout_site, "train_site": prepared.fold.train_site,
                     "protocol": prepared.fold.protocol, "view": prepared.fold.view, "model_type": model_type,
                     "seed": SEED, "model_complete_sha256": model_complete_sha256,
                     "feature_ids_sha256": _sha_bytes(_canonical(list(prepared.fold.feature_ids))),
                     "test_indices_sha256": _array_sha(indices)}
    existing = Path(path)
    if existing.exists() or existing.is_symlink():
        saved = _read_json(existing / "metadata.json")
        if any(saved.get(key) != value for key, value in base_identity.items()):
            raise UnitIdentityError("prediction unit identity mismatch")
        complete = _read_json(existing / "complete.json")
        if complete.get("schema") != PREDICTION_SCHEMA or complete.get("status") != "COMPLETE" or complete.get("file_summaries") != _tree_summaries(existing):
            raise UnitIdentityError("prediction unit digest mismatch")
        return {"status": "SKIP_VERIFIED", "path": existing, "metadata": saved}
    model_metadata = _read_json(Path(model_path) / "metadata.json")
    files = model_metadata.get("checkpoint_files")
    if not isinstance(files, dict): raise UnitIdentityError("model checkpoint files missing")
    trained = load_checkpoint(model_path, model_type, files)
    scores = predict_scores(model_type, trained.model, trained.adapter, prepared.fold.X_test)
    if not np.isfinite(scores).all() or np.ptp(scores) == 0: raise UnitIdentityError("prediction is invalid or constant")
    payload = {"test_global_indices": indices, "scores": np.asarray(scores, dtype=np.float64)}
    identity = {**base_identity, "prediction_sha256": _array_sha(scores)}
    metadata = {**identity, "feature_count": len(prepared.fold.feature_ids), "test_count": len(indices),
                "score_type": _score_type(model_type), "finite": True, "nonconstant": True,
                "score_min": float(np.min(scores)), "score_max": float(np.max(scores)), "score_std": float(np.std(scores))}
    result = _publish_payload_unit(path, payload, metadata, PREDICTION_SCHEMA, "prediction.npz")
    return {"status": result.status, "path": result.path, "metadata": result.metadata}


def build_reference_manifest_v2(output_path: Path, old_root: Path, new_root: Path, run_identity_sha256: str) -> Mapping[str, object]:
    entries = {
        "E0": {"V1": "old", "V2": "old", "V3": "old", "V4": "old", "V5": "old"},
        "E1": {"V1": "v2", "V2": "v2", "V3": "v2", "V4": "v2", "V5": "old"},
        "E2": {"V1": "old", "V2": "old", "V3": "old", "V4": "old", "V5": "v2"},
        "E3": {"V1": "v2", "V2": "v2", "V3": "v2", "V4": "v2", "V5": "v2"},
    }
    rows = []
    for experiment, sources in entries.items():
        for holdout, protocol, view, model, seed in prediction_identities_v2():
            source = sources[view]
            root = Path(new_root) if source == "v2" else Path(old_root)
            rows.append({"experiment": experiment, "holdout_site": holdout, "protocol": protocol, "view": view,
                         "model_type": model, "seed": seed, "source": source,
                         "prediction_unit": str(root / "predictions" / f"holdout_{holdout}" / protocol / view / model),
                         "source_identity_only": True})
    payload = {"schema": "featurelab_v2_seed42_reference_manifest_v1", "run_identity_sha256": run_identity_sha256,
               "experiments": entries, "units": rows, "status": "REFERENCE_ONLY_NO_SCORING"}
    _write_json(Path(output_path), payload)
    return payload



def run_seed42_v2(
    rank_root: Path,
    manifest_path: Path,
    metadata,
    old_root: Path,
    output_root: Path,
    run_identity: Mapping[str, object],
    source_fingerprints: Mapping[str, str],
) -> Mapping[str, object]:
    """Build/resume all 30 models and 90 prediction units serially."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_identity_sha256 = _sha_bytes(_canonical(run_identity))
    run_path = output_root / "run_identity.json"
    if run_path.exists():
        if _read_json(run_path) != json.loads(_canonical(run_identity)):
            raise UnitIdentityError("Seed42 run identity drift")
    else:
        _write_json(run_path, run_identity)
    folds = {}
    prepared = {}
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            for view in VIEWS:
                fold = load_v2_fold(rank_root, holdout, protocol, view, metadata, manifest_path, source_fingerprints)
                item = prepare_seed42_training(fold)
                folds[(holdout, protocol, view)] = fold
                prepared[(holdout, protocol, view)] = item
    old_bridge = {}
    for train_site in TRAIN_SITES:
        canonical = CANONICAL_HOLDOUT[train_site]
        item = prepared[(canonical, "transductive", "V1")]
        old_bridge[train_site] = {
            "sample_match": bool(compare_old_sample(item, old_root)),
            "query_match": bool(compare_old_query(item, old_root)),
        }
    model_results = []
    for train_site, _seed, view, model_type in model_identities_v2():
        item = prepared[(CANONICAL_HOLDOUT[train_site], "transductive", view)]
        target = output_root / "models" / train_site / f"seed{SEED}" / view / model_type
        result = build_model_unit_v2(target, item, model_type, run_identity_sha256)
        model_results.append({"train_site": train_site, "view": view, "model_type": model_type, "status": result["status"]})
    prediction_results = []
    for holdout, protocol, view, model_type, _seed in prediction_identities_v2():
        train_site = HOLDOUT_TO_TRAIN[holdout]
        item = prepared[(holdout, protocol, view)]
        model_path = output_root / "models" / train_site / f"seed{SEED}" / view / model_type
        model_complete_sha256 = sha256_file(model_path / "complete.json")
        target = output_root / "predictions" / f"holdout_{holdout}" / protocol / view / model_type
        result = build_prediction_unit_v2(target, model_path, item, model_type, run_identity_sha256, model_complete_sha256)
        prediction_results.append({"holdout_site": holdout, "protocol": protocol, "view": view, "model_type": model_type, "status": result["status"]})
    reference_path = output_root / "reference_manifest.json"
    reference = build_reference_manifest_v2(reference_path, old_root, output_root, run_identity_sha256)
    summary = {
        "schema": "featurelab_v2_seed42_private_summary_v1", "run_identity_sha256": run_identity_sha256,
        "models": model_results, "predictions": prediction_results, "old_bridge": old_bridge,
        "model_count": len(model_results), "prediction_count": len(prediction_results),
        "reference_manifest": str(reference_path), "status": "PASS",
    }
    _write_json(output_root / "summary.json", summary)
    return summary
